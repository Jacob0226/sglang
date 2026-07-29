# SPDX-License-Identifier: Apache-2.0

import fnmatch
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, List, Optional, cast

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import (  # noqa: E501
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.quark.schemes import (
    QuarkLinearScheme,
    QuarkMoEScheme,
    QuarkW4A4MXFP4,
    QuarkW4A4MXFp4MoE,
    QuarkW4A8MXFp4MoE,
    QuarkW8A8Fp8,
    QuarkW8A8FP8MoE,
)
from sglang.srt.layers.quantization.quark.utils import deep_compare, should_ignore_layer
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import get_device_capability

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput

__all__ = ["QuarkLinearMethod", "QuarkFusedMoEMethod"]

logger = logging.getLogger(__name__)

# MLA-attention projection linears that Quark MXFP4 checkpoints leave in the
# `exclude` list (i.e. bf16). SGLANG_DSA_FP8_PROJ_GEMM re-quantizes them to FP8
# at load time so they match ATOM, which runs the same weights in FP8.
#
# Three FP8 flavours are available and they are far from equivalent. Measured on
# GLM-5.2-MXFP4 i8k/conc64 prefill (MI355X), summed over the 78 calls one
# projection makes per EXTEND forward, GEMM plus whatever activation quant it
# needs:
#
#   o_proj (M=16368, N=6144, K=4096)
#     bf16       Tensile Cijk_BBS  46.3 + flatten (a free view)         = 46.3ms
#     pertensor  Tensile Cijk_F8BS 21.9 + amax 6.6 + scaled_quant 3.2   = 31.7ms
#     ptpc       aiter bpreshuffle 34.4 + per_token_group_quant 3.1     = 37.5ms
#     block      CK blockscale     57.0 + fused_flatten_fp8_quant 13.3  = 70.3ms
#   q_b_proj
#     bf16       aiter bf16gemm    16.0 + fused_qk_rmsnorm 2.6          = 18.6ms
#     block      CK blockscale     15.4 + fused_rms_fp8_group_quant 3.1 = 18.5ms
#
# Tensile's FP8 GEMM is roughly twice as fast as either aiter/CK kernel on the
# o_proj shape, so the per-tensor route is the fastest of the three: it cut the
# prefill EXTEND forward by 11.3ms. What it must avoid is aiter's
# dynamic_per_tensor_quant: that is 3 kernels whose amax atomicMax'es every
# block into one global float, 17.4ms where a plain reduction takes 6.6ms.
#
# All three are nevertheless off by default, because end to end none of them
# pays: per-tensor o_proj measured GSM8K 0.922 vs 0.936 baseline and output
# throughput 1930 vs 2049 tok/s at i1k/conc64. The GEMM win is a prefill-only
# effect while the activation quant is pure overhead at decode-sized M, and a
# single scale over the whole activation is too coarse once real inputs have
# outlier tokens (a synthetic normal input hides this: relative L2 vs bf16 is
# 0.0374 per-tensor, 0.0375 per-token).
#
# q_b_proj is break-even in every flavour, so it stays bf16. kv_b_proj is out of
# scope: on the absorb path it is folded into w_kc/w_vc and consumed by a bmm,
# not a standard Linear.apply.
#
# pertensor and ptpc quantize the activation inside apply() and so need a bf16
# input; those layers are marked and the MLA forward skips its fused pre-quant.
_FP8_PROJ_GEMM_BLOCK_SUFFIXES = ()
_FP8_PROJ_GEMM_PTPC_SUFFIXES = ()
_FP8_PROJ_GEMM_PER_TENSOR_SUFFIXES = ()

_FP8_PROJ_GEMM_BLOCK_SIZE = [128, 128]


def _fp8_proj_gemm_mode(prefix: str) -> Optional[str]:
    if "self_attn" not in prefix:
        return None
    for mode, suffixes in (
        ("block", _FP8_PROJ_GEMM_BLOCK_SUFFIXES),
        ("ptpc", _FP8_PROJ_GEMM_PTPC_SUFFIXES),
        ("pertensor", _FP8_PROJ_GEMM_PER_TENSOR_SUFFIXES),
    ):
        if suffixes and prefix.endswith(suffixes):
            return mode
    return None


def _block_cast_to_fp8(
    weight: torch.Tensor, block_size: List[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [N, K] weight to FP8 with one fp32 scale per block_n x block_k tile."""
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max

    block_n, block_k = block_size
    n, k = weight.shape
    assert n % block_n == 0 and k % block_k == 0, (
        f"SGLANG_DSA_FP8_PROJ_GEMM needs a weight divisible by {block_size}, "
        f"got {(n, k)}. Drop this projection from _FP8_PROJ_GEMM_BLOCK_SUFFIXES."
    )
    tiles = weight.view(n // block_n, block_n, k // block_k, block_k).float()
    scale = tiles.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4) / fp8_max
    qweight = (tiles / scale).clamp(min=-fp8_max, max=fp8_max).to(fp8_dtype)
    return (
        qweight.view(n, k).contiguous(),
        scale.view(n // block_n, k // block_k).contiguous(),
    )


@lru_cache(maxsize=1)
def _online_fp8_linear_classes() -> dict:
    # Imported lazily: fp8.py pulls in a large chunk of the quantization stack.
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

    def _online_config() -> Fp8Config:
        return Fp8Config(
            is_checkpoint_fp8_serialized=False, activation_scheme="dynamic"
        )

    class OnlinePerTensorFp8LinearMethod(Fp8LinearMethod):
        """Per-tensor activation x per-tensor weight FP8, via Tensile _scaled_mm.

        Weight setup is the stock non-block online path. Only apply() is
        replaced, to compute the activation scale with a plain reduction instead
        of aiter's dynamic_per_tensor_quant (3 kernels, contended atomicMax).
        """

        def __init__(self):
            super().__init__(quant_config=_online_config())

        def apply(self, layer, x, bias=None):
            import aiter

            from sglang.kernels.ops.quantization.fp8_kernel import fp8_max

            x_2d = x.view(-1, x.shape[-1])
            amax = torch.linalg.vector_norm(x_2d, ord=torch.inf).float()
            x_scale = (amax / fp8_max).clamp(min=1e-12).view(1)
            q_x, _ = aiter.per_tensor_quant_hip(
                x_2d, scale=x_scale, quant_dtype=aiter.dtypes.fp8
            )
            out = torch._scaled_mm(
                q_x,
                layer.weight,
                out_dtype=x.dtype,
                scale_a=x_scale,
                scale_b=layer.weight_scale,
                bias=bias,
            )
            return out.view(*x.shape[:-1], out.shape[-1])

    class OnlinePtpcFp8LinearMethod(Fp8LinearMethod):
        """Per-token activation x per-channel weight FP8, via aiter bpreshuffle.

        The stock non-block path already implements this; it just needs
        use_aiter_fp8_per_token, which is otherwise driven by a global env var.
        """

        def __init__(self):
            super().__init__(quant_config=_online_config())
            self.use_aiter_fp8_per_token = True

    class OnlineBlockFp8LinearMethod(Fp8LinearMethod):
        """Block-FP8 linear for bf16 weights that the checkpoint left unquantized.

        Fp8Config rejects weight_block_size unless the checkpoint is fp8-serialized,
        so the config is built without it and block_quant is switched on afterwards.
        create_weights then allocates a bf16 weight (which is what the checkpoint
        holds) and no scale parameter, and we quantize into weight_scale_inv here
        before the stock block-FP8 post-processing runs.
        """

        def __init__(self):
            config = _online_config()
            config.weight_block_size = _FP8_PROJ_GEMM_BLOCK_SIZE
            super().__init__(quant_config=config)

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            qweight, weight_scale = _block_cast_to_fp8(
                layer.weight.data, _FP8_PROJ_GEMM_BLOCK_SIZE
            )
            layer.weight = torch.nn.Parameter(qweight, requires_grad=False)
            layer.register_parameter(
                "weight_scale_inv",
                torch.nn.Parameter(weight_scale, requires_grad=False),
            )
            layer.input_scale = None
            super().process_weights_after_loading(layer)

    return {
        "block": OnlineBlockFp8LinearMethod,
        "ptpc": OnlinePtpcFp8LinearMethod,
        "pertensor": OnlinePerTensorFp8LinearMethod,
    }

_MOE_SHARED_EXPERT_QUANT_LAYER0_BASES: tuple[str, ...] = (
    "model.layers.0",
    "model.language_model.layers.0",
)

_SHARED_EXPERT_BODY_PROJ_SUFFIXES: tuple[str, ...] = (
    "gate_proj",
    "up_proj",
    "gate_up_proj",
    "down_proj",
)


class QuarkConfig(QuantizationConfig):

    def __init__(
        self,
        quant_config: Optional[dict[str, Any]] = None,
        hf_config: "PretrainedConfig | None" = None,
        kv_cache_group: Optional[list[str]] = None,
        kv_cache_config: Optional[dict[str, Any]] = None,
        pack_method: str = "reorder",
        is_prequantized: bool = False,
        online_scheme: Optional[str] = None,
    ):
        super().__init__()
        if kv_cache_group is None:
            kv_cache_group = []

        if online_scheme is not None:
            assert not is_prequantized
            if online_scheme == "quark_mxfp4":
                quant_config = self._create_online_mxfp4_config(
                    model_type=hf_config.model_type
                )
            else:
                raise ValueError(f"Unsupported online_scheme: {online_scheme}")

        if quant_config is None:
            raise ValueError("Either quant_config or online_scheme must be provided")

        self.quant_config = quant_config
        self.kv_cache_group = kv_cache_group
        self.kv_cache_config = kv_cache_config
        self.pack_method = pack_method
        self.exclude_layers = cast(list[str], self.quant_config.get("exclude", []))
        self.is_prequantized = is_prequantized

        self.packed_modules_mapping = self.quant_config["packed_modules_mapping"]
        self._quantized_layers = set()

    @property
    def quantized_layers(self) -> tuple[list[str], int]:
        # Extract unique layer types (last part after ".")
        layer_types = sorted(
            set(name.split(".")[-1] for name in self._quantized_layers)
        )
        return layer_types, len(self._quantized_layers)

    def get_linear_method(self) -> "QuarkLinearMethod":
        return QuarkLinearMethod(self)

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_name(self) -> str:
        return "quark"

    def apply_weight_name_mapper(self, hf_to_sglang_mapper):
        mapped = hf_to_sglang_mapper.apply_list(self.exclude_layers)
        expanded = []
        for name in mapped:
            expanded.append(name)
            if name.startswith("language_model."):
                expanded.append(name.removeprefix("language_model."))
        self.exclude_layers = list(dict.fromkeys(expanded))

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        # Check if the layer is skipped for quantization.

        if should_ignore_layer(
            prefix,
            ignore=self.exclude_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            if isinstance(layer, LinearBase):
                mode = (
                    _fp8_proj_gemm_mode(prefix)
                    if envs.SGLANG_DSA_FP8_PROJ_GEMM.get()
                    else None
                )
                if mode is not None:
                    logger.info_once(
                        "SGLANG_DSA_FP8_PROJ_GEMM: %s FP8 for MLA proj %s", mode, prefix
                    )
                    # These modes quantize the activation themselves and need
                    # bf16 in; the MLA forward checks this before emitting its
                    # pre-quant tuple.
                    layer._sglang_online_fp8_needs_bf16_input = mode != "block"
                    return _online_fp8_linear_classes()[mode]()
                return UnquantizedLinearMethod()
            elif isinstance(layer, RadixAttention):
                return QuarkKVCacheMethod(self)
            return None

        if isinstance(layer, LinearBase):
            scheme = self.get_linear_scheme(layer=layer, layer_name=prefix)
            layer.scheme = scheme
            self._quantized_layers.add(prefix)
            return QuarkLinearMethod(self)

        if isinstance(layer, RadixAttention):
            self._quantized_layers.add(prefix)
            return QuarkKVCacheMethod(self)

        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        if isinstance(layer, FusedMoE):
            self._quantized_layers.add(prefix)
            layer.scheme = self.get_moe_scheme(layer, prefix)
            return QuarkFusedMoEMethod(self)

        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QuarkConfig":
        export_config = config.get("export")
        if export_config is None:
            raise ValueError(
                "The export key should be included in "
                "the configurations of Quark quantized model"
            )

        kv_cache_group = cast(list[str], export_config.get("kv_cache_group"))
        pack_method = cast(str, export_config.get("pack_method"))

        # In the export model of quark, the quantization configuration
        # of kv_cache is stored in layer_quant_config. First, it is
        # judged whether kv_cache_group exists, and then it is judged
        # whether layer_quant_config has a quantization configuration
        # that matches kv_cache.
        if len(kv_cache_group) == 0:
            kv_cache_config = None
        else:
            kv_cache_set = set(kv_cache_group)
            layer_quant_config = cast(dict[str, Any], config.get("layer_quant_config"))
            layer_quant_names = list(layer_quant_config.keys())
            layer_quant_set = set(layer_quant_names)

            if not kv_cache_set.issubset(layer_quant_set):
                raise ValueError(
                    "The Quark quantized model has the "
                    "kv_cache_group parameter setting, "
                    "but no kv_cache quantization settings "
                    "were found in the quantization "
                    "configuration."
                )

            q_configs = [
                cast(dict[str, Any], layer_quant_config.get(name))
                for name in kv_cache_group
            ]
            if not all(deep_compare(q_config, q_configs[0]) for q_config in q_configs):
                raise ValueError(
                    "The quantization method used for kv_cache should "
                    "be the same, but the quantization method for the "
                    "kv_cache layer in the config is different."
                )
            kv_cache_config = q_configs[0].get("output_tensors")
            if kv_cache_config is None:
                raise ValueError("The kv_cache quantization configuration is empty.")

            # Since we have already set kv_cache quantization configurations,
            # we will remove the quantization configuration for the
            # output_tensors corresponding to the kv_cache layer.
            for q_config in q_configs:
                q_config["output_tensors"] = None

            # In case q_proj output is also quantized, remove the configuration
            # to keep qkv consistency.
            q_proj_q_config = cast(dict[str, Any], layer_quant_config.get("*q_proj"))
            if q_proj_q_config is not None:
                q_proj_q_config["output_tensors"] = None

        return cls(
            quant_config=config,
            kv_cache_group=kv_cache_group,
            kv_cache_config=kv_cache_config,
            pack_method=pack_method,
            is_prequantized=True,
        )

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @staticmethod
    def _create_online_mxfp4_config(model_type: str) -> dict[str, Any]:
        """
        Create a synthetic quant_config for online MXFP4 quantization.
        """
        # MOE gate/router is typically implemented as a ReplicatedLinear, and skipped for quantization for accuracy reasons.
        # lm_head/embed_tokens is also skipped for accuracy reasons, normally not handled by `QuarkConfig` in any case, but adding them here for safety.
        exclude = [
            "re:.*gate$",
            "re:.*router",
            "re:.*lm_head",
            "re:.*embed_tokens",
        ]

        if model_type == "qwen3_5_moe":
            # Exclusion for accuracy adapted from
            # https://huggingface.co/amd/Qwen3.5-397B-A17B-MXFP4/blob/main/config.json
            exclude.extend(
                [
                    "re:.*n_proj_a",
                    "re:.*in_proj_b",
                    "re:.*in_proj_qkv",
                    "re:.*in_proj_z",
                    "re:.*o_proj",
                    "re:.*out_proj",
                    "re:.*qkv_proj",
                    "re:.*shared_expert",
                ]
            )

        return {
            "packed_modules_mapping": {},
            "exclude": exclude,
            "global_quant_config": {
                "weight": {
                    "dtype": "fp4",
                    "qscheme": "per_group",
                    "group_size": 32,
                    "is_dynamic": False,
                    "scale_format": "e8m0",
                },
                "input_tensors": {
                    "dtype": "fp4",
                    "qscheme": "per_group",
                    "group_size": 32,
                    "is_dynamic": True,
                    "scale_format": "e8m0",
                },
                "output_tensors": None,
                "bias": None,
            },
            "layer_quant_config": {},
            "layer_type_quant_config": {},
            "export": {
                "kv_cache_group": [],
                "pack_method": "reorder",
            },
        }

    def _check_scheme_supported(self, min_capability: int, error: bool = True) -> bool:
        capability_tuple = get_device_capability()

        if capability_tuple is not None:
            assert 0 <= capability_tuple[1] < 10
            capability = capability_tuple[0] * 10 + capability_tuple[1]

            supported = capability >= min_capability
            if error and not supported:
                # Pass a single joined message; RuntimeError stringifies
                # multiple positional args as a tuple repr.
                raise RuntimeError(
                    "Quantization scheme is not supported for "
                    f"the current GPU. Min capability: {min_capability}. "
                    f"Current capability: {capability}."
                )
            return supported
        else:
            return False

    def _is_fp8_w8a8(
        self,
        weight_quant: Optional[dict[str, Any]],
        input_quant: Optional[dict[str, Any]],
    ) -> bool:
        # Confirm weights and input quantized.
        if weight_quant is None or input_quant is None:
            return False

        # Confirm weight scheme is supported
        is_fp8_dtype = (
            weight_quant.get("dtype") == "fp8_e4m3"
            and input_quant.get("dtype") == "fp8_e4m3"
        )
        is_static_weight = not weight_quant.get("is_dynamic")
        is_per_tensor_or_channel_weight = weight_quant.get("qscheme") in [
            "per_tensor",
            "per_channel",
        ]

        if not (is_fp8_dtype and is_static_weight and is_per_tensor_or_channel_weight):
            return False

        # Dynamic quantization is always supported if weights supported.
        if input_quant.get("is_dynamic"):
            return True

        # Confirm activation scheme is supported.
        is_per_tensor_activation = input_quant.get("qscheme") == "per_tensor"
        return is_per_tensor_activation

    def _is_mx_fp4(
        self,
        weight_quant: Optional[dict[str, Any]],
        input_quant: Optional[dict[str, Any]],
    ) -> bool:
        # Confirm weights and input quantized.
        if weight_quant is None or input_quant is None:
            logger.debug(
                "Quark model is not in MX-FP4 format: "
                "weight_quant or input_quant not set"
            )
            return False

        # Input and weight dtype needs to be fp4.
        if weight_quant.get("dtype") != "fp4" or input_quant.get("dtype") != "fp4":
            logger.debug("Quark model is not in MX-FP4 format: dtype not fp4")
            return False

        # Input and weight qscheme needs to be per group.
        if (
            weight_quant.get("qscheme") != "per_group"
            or input_quant.get("qscheme") != "per_group"
        ):
            logger.debug("Quark model is not in MX-FP4 format: not per_group")
            return False

        # Input and weight group size needs to be 32.
        if weight_quant.get("group_size") != 32 or input_quant.get("group_size") != 32:
            logger.debug("Quark model is not in MX-FP4 format: not group_size=32")
            return False

        # Weights need to use static quantization.
        if weight_quant.get("is_dynamic") is True:
            logger.debug("Quark model is not in MX-FP4 format: not weight static")
            return False

        # Activations need to use dynamic quantization.
        if input_quant.get("is_dynamic") is False:
            logger.debug("Quark model is not in MX-FP4 format: not activation dynamic")
            return False

        # Activations and weight scales need to be in e8m0 format.
        if (
            weight_quant.get("scale_format") != "e8m0"
            or input_quant.get("scale_format") != "e8m0"
        ):
            logger.debug("Quark model is not in MX-FP4 format: not scale_format e8m0")
            return False

        return True

    def _is_mx_w4a8(
        self,
        weight_quant: Optional[dict[str, Any]],
        input_quant: Optional[dict[str, Any]],
    ) -> bool:
        if weight_quant is None or input_quant is None:
            return False

        is_mx_fp4_weight = (
            weight_quant.get("dtype") == "fp4"
            and weight_quant.get("qscheme") == "per_group"
            and weight_quant.get("group_size") == 32
            and not weight_quant.get("is_dynamic")
            and weight_quant.get("scale_format") == "e8m0"
        )
        is_static_fp8_activation = (
            input_quant.get("dtype") in ("fp8_e4m3", "fp8_e4m3fn")
            and input_quant.get("qscheme") == "per_tensor"
            and not input_quant.get("is_dynamic")
        )
        return is_mx_fp4_weight and is_static_fp8_activation

    def _find_matched_config(
        self, layer_name: str, module: torch.nn.Module
    ) -> dict[str, Any]:

        proj_name = layer_name.split(".")[-1]
        if proj_name in self.packed_modules_mapping:
            shard_proj_names = self.packed_modules_mapping[proj_name]

            # Convert fused_name --> [shard_names]
            shard_names = [
                layer_name.replace(proj_name, shard_proj_name)
                for shard_proj_name in shard_proj_names
            ]
            shard_configs = [
                self._find_matched_config(shard_name, module)
                for shard_name in shard_names
            ]
            if not all(
                deep_compare(q_config, shard_configs[0]) for q_config in shard_configs
            ):
                raise ValueError(
                    f"Found a different quantization configuration for "
                    f"{shard_proj_names} in {layer_name}. SGLang "
                    "requires all to use the same scheme."
                )
            return shard_configs[0]
        else:
            layer_quant_config = cast(
                dict[str, Any], self.quant_config.get("layer_quant_config")
            )
            for name_pattern in layer_quant_config:
                if fnmatch.fnmatch(layer_name, name_pattern):
                    return layer_quant_config[name_pattern]

            layer_type = type(module).__name__
            layer_type_quant_config = cast(
                dict[str, Any], self.quant_config.get("layer_type_quant_config")
            )
            if layer_type in layer_type_quant_config:
                return layer_type_quant_config[layer_type]

            global_quant_config = cast(
                dict[str, Any], self.quant_config.get("global_quant_config")
            )
            return global_quant_config

    def _get_scheme_from_config(self, config: dict[str, Any]) -> "QuarkLinearScheme":
        if config.get("output_tensors") or config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with output_tensors "
                "and bias quantized are not supported"
            )
        weight_config = cast(dict[str, Any], config.get("weight"))
        input_config = cast(dict[str, Any], config.get("input_tensors"))

        if self._is_mx_fp4(weight_config, input_config):
            return QuarkW4A4MXFP4(
                weight_config,
                input_config,
                is_checkpoint_mxfp4_serialized=self.is_prequantized,
            )
        if self._is_fp8_w8a8(weight_config, input_config):
            is_fp8_w8a8_supported = self._check_scheme_supported(
                QuarkW8A8Fp8.get_min_capability(), error=False
            )
            if is_fp8_w8a8_supported:
                return QuarkW8A8Fp8(weight_config, input_config)

        raise NotImplementedError(
            "No quark compatible scheme was found. "
            f"Weight config: {weight_config}, "
            f"Input config: {input_config}"
        )

    def get_linear_scheme(
        self, layer: torch.nn.Module, layer_name: str
    ) -> "QuarkLinearScheme":

        layer_quant_config = self._find_matched_config(layer_name, layer)

        # Find the quant_scheme
        scheme = self._get_scheme_from_config(layer_quant_config)

        # Raise error if device does not support the scheme
        # (e.g. fp8 needs ada lovelace)
        self._check_scheme_supported(scheme.get_min_capability())

        return scheme

    def get_moe_scheme(
        self,
        module: torch.nn.Module,
        layer_name: str,
    ) -> "QuarkMoEScheme":
        layer_quant_config = self._find_matched_config(layer_name, module)

        if layer_quant_config.get("output_tensors") or layer_quant_config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with "
                "output_tensors and bias "
                "quantized are not supported"
            )
        weight_config = layer_quant_config.get("weight")
        input_config = layer_quant_config.get("input_tensors")

        if self._is_mx_fp4(weight_config, input_config):
            return QuarkW4A4MXFp4MoE(
                weight_config,
                input_config,
                is_checkpoint_mxfp4_serialized=self.is_prequantized,
            )
        elif self._is_mx_w4a8(weight_config, input_config):
            logger.info_once("Using Quark MXFP4-W/FP8-A MoE scheme")
            return QuarkW4A8MXFp4MoE(weight_config, input_config)
        elif self._is_fp8_w8a8(weight_config, input_config):
            return QuarkW8A8FP8MoE(weight_config, input_config)
        else:
            raise RuntimeError("Unsupported FusedMoe scheme")

    def get_scaled_act_names(self) -> List[str]:
        return []

    def can_fuse_shared_expert(self) -> bool:
        # Shared-expert body excluded from quant; the gate must not veto fusion.
        if any(
            "shared_expert" in layer
            and "shared_expert_gate" not in layer
            and not layer.startswith("mtp.")
            for layer in self.exclude_layers
        ):
            return False

        # No per-layer config -> uniform spec, nothing to compare.
        layer_quant_config = self.quant_config.get("layer_quant_config") or {}
        if not layer_quant_config:
            return True

        # Compare routed vs shared specs at layer 0 (stub module needed by
        # _find_matched_config; an unmatched name -> ValueError -> cannot fuse).
        lookup_stub = torch.nn.Module()
        try:
            for base in _MOE_SHARED_EXPERT_QUANT_LAYER0_BASES:
                moe_name = f"{base}.mlp.experts"
                moe_cfg = self._find_matched_config(moe_name, lookup_stub)
                for suffix in _SHARED_EXPERT_BODY_PROJ_SUFFIXES:
                    shared_name = f"{base}.mlp.shared_expert.{suffix}"
                    shared_cfg = self._find_matched_config(shared_name, lookup_stub)
                    if not deep_compare(moe_cfg, shared_cfg):
                        return False
        except ValueError:
            return False

        return True


class QuarkLinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: QuarkConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Use the QuarkLinearScheme associated with the layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Use the output of create_weights and the QuarkLinearScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, x, bias=bias)


class QuarkFusedMoEMethod(FusedMoEMethodBase):

    def __init__(self, quantization_config: QuarkConfig):
        self.quantization_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """
        Use the QuarkMoEScheme associated with the layer to create
        the necessary parameters for the layer. See FusedMoEMethodBase for param
        details
        """
        layer.scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        layer.scheme.create_moe_runner(layer, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ):
        """
        Use the output of create_weights and the QuarkMoEScheme
        associated with the layer to apply the forward pass with the
        fused MoE layer. See FusedMoEMethodBase for param details

        """
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        return scheme.apply_weights(layer, dispatch_output)


class QuarkKVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from quark checkpoints.
    """

    def __init__(self, quant_config: QuarkConfig):
        self.validate_kv_cache_config(quant_config.kv_cache_config)
        super().__init__(quant_config)

    @staticmethod
    def validate_kv_cache_config(kv_cache_config: Optional[dict[str, Any]]):
        """
        Validator for the kv cache configuration. Useful for controlling the
        kv cache quantization schemes, that are being supported in vLLM
        :param kv_cache_config: the quark kv cache scheme
        """
        if kv_cache_config is None:
            return

        dtype = kv_cache_config.get("dtype")
        if dtype != "fp8_e4m3":
            raise NotImplementedError(
                "Currently supported kv cache quantization is "
                f"dtype=fp8_e4m3, however received {dtype}"
            )

        qscheme = kv_cache_config.get("qscheme")
        if qscheme != "per_tensor":
            raise NotImplementedError(
                "Only support per-tensor scaling factor "
                "for quark KV cache. "
                f"Expected qscheme: per_tensor, found qscheme: {qscheme}"
            )

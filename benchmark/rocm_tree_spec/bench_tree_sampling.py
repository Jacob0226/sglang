"""Microbenchmark for the target-only tree verifier, Triton vs CUDA AOT.

Runs on ROCm (Triton only) and on CUDA (both, so the ratio isolates the port from
the hardware -- timings from two different GPUs are not comparable).

Two things this is careful about:

- Real tree topologies. A chain placeholder leaves `retrive_next_sibling` all -1,
  so the sibling walk that the tree verifier exists for never executes.
- Scratch state. The kernel writes rejected candidates back into `draft_probs`, so
  a second call on the same buffer takes different branches. The buffer is reset
  between iterations, outside the timed region -- folding a vocabulary-sized memset
  into the measurement would swamp the kernel itself.

Usage:  python3 bench_tree_sampling.py [--vocab 151936] [--iters 20]
"""

from __future__ import annotations

import argparse
import statistics

import torch

from sglang.kernels.ops.speculative.tree_sampling import (
    tree_speculative_sampling_target_only_triton,
)

DEVICE = torch.device("cuda")


def load_aot_kernel():
    """The python wrapper imports fine on ROCm; only the torch op is missing, so
    probe with a real call rather than trusting the import."""
    try:
        from sgl_kernel import tree_speculative_sampling_target_only as aot
    except (ImportError, AttributeError):
        return None
    try:
        args = make_inputs(batch_size=1, num_draft_tokens=2, width=1, vocab_size=8)
        aot(**args, threshold_single=1.0, threshold_acc=1.0, deterministic=True)
    except (AttributeError, NotImplementedError, RuntimeError):
        return None
    return aot


class Topology:
    """First-child / next-sibling encoding of a breadth-first k-ary tree.

    A uniform tree is a proxy: EAGLE-2 prunes globally by confidence, so a real
    draft tree is ragged. Width still bounds the branching, which is the axis that
    decides whether the sibling walk does any work.

    ``parent`` and ``rank`` are kept because the verifier scores a child against
    the *parent's* target row, so candidates have to be built per parent.
    """

    def __init__(self, num_nodes: int, width: int):
        children: list[list[int]] = [[] for _ in range(num_nodes)]
        self.parent = [0] * num_nodes
        self.rank = [0] * num_nodes
        node_depth = [0] * num_nodes

        frontier = [0]
        next_id = 1
        while frontier and next_id < num_nodes:
            p = frontier.pop(0)
            for slot in range(width):
                if next_id >= num_nodes:
                    break
                children[p].append(next_id)
                self.parent[next_id] = p
                self.rank[next_id] = slot
                node_depth[next_id] = node_depth[p] + 1
                frontier.append(next_id)
                next_id += 1

        self.next_token = [-1] * num_nodes
        self.next_sibling = [-1] * num_nodes
        for p, kids in enumerate(children):
            if kids:
                self.next_token[p] = kids[0]
                for older, younger in zip(kids, kids[1:]):
                    self.next_sibling[older] = younger

        self.depth = max(node_depth) + 1
        self.width = width


def make_candidates(
    target_probs: torch.Tensor,
    topology: Topology,
    agreement: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draft candidates the target mostly agrees with.

    Uniformly random token ids sit at p ~ 1/vocab, so every candidate is rejected at
    the root and the traversal the verifier exists for never runs. The verifier
    scores a child against its parent's target row, so siblings are taken as that
    row's top-`width` tokens -- which is also how EAGLE builds them -- and a
    fraction is corrupted to model draft/target drift.
    """
    batch_size, num_draft_tokens, vocab_size = target_probs.shape
    top_tokens = target_probs.topk(max(topology.width, 1), dim=-1).indices

    parent = torch.tensor(topology.parent, dtype=torch.int64, device=DEVICE)
    rank = torch.tensor(topology.rank, dtype=torch.int64, device=DEVICE)
    agreed = top_tokens[:, parent, rank]

    corrupted = torch.randint(
        0,
        vocab_size,
        (batch_size, num_draft_tokens),
        dtype=torch.int64,
        device=DEVICE,
        generator=generator,
    )
    agrees = (
        torch.rand((batch_size, num_draft_tokens), device=DEVICE, generator=generator)
        < agreement
    )
    return torch.where(agrees, agreed, corrupted)


def make_inputs(
    *,
    batch_size: int,
    num_draft_tokens: int,
    width: int,
    vocab_size: int,
    seed: int = 0,
    agreement: float = 0.7,
    logit_scale: float = 6.0,
) -> dict:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    tree = Topology(num_draft_tokens, width)

    def per_request(values):
        return (
            torch.tensor(values, dtype=torch.int64, device=DEVICE)
            .expand(batch_size, -1)
            .contiguous()
        )

    # Peaked, like a trained model's next-token distribution. softmax(randn) over a
    # 150K vocabulary is nearly uniform and would make every acceptance test fail.
    target_probs = torch.softmax(
        torch.randn(
            (batch_size, num_draft_tokens, vocab_size),
            dtype=torch.float32,
            device=DEVICE,
            generator=generator,
        )
        * logit_scale,
        dim=-1,
    )

    return dict(
        predicts=torch.full(
            (batch_size * num_draft_tokens,), -1, dtype=torch.int32, device=DEVICE
        ),
        accept_index=torch.full(
            (batch_size, tree.depth), -1, dtype=torch.int32, device=DEVICE
        ),
        accept_token_num=torch.zeros(batch_size, dtype=torch.int32, device=DEVICE),
        candidates=make_candidates(target_probs, tree, agreement, generator),
        retrive_index=torch.arange(
            batch_size * num_draft_tokens, dtype=torch.int64, device=DEVICE
        ).view(batch_size, num_draft_tokens),
        retrive_next_token=per_request(tree.next_token),
        retrive_next_sibling=per_request(tree.next_sibling),
        uniform_samples=torch.rand(
            (batch_size, num_draft_tokens),
            dtype=torch.float32,
            device=DEVICE,
            generator=generator,
        ),
        uniform_samples_for_final_sampling=torch.rand(
            (batch_size,), dtype=torch.float32, device=DEVICE, generator=generator
        ),
        target_probs=target_probs,
        draft_probs=torch.zeros(
            (batch_size, num_draft_tokens, vocab_size),
            dtype=torch.float32,
            device=DEVICE,
        ),
    )


def time_kernel(kernel, args: dict, iters: int) -> tuple[float, float]:
    """Median milliseconds, and the mean accepted draft length."""

    def call():
        kernel(**args, threshold_single=1.0, threshold_acc=1.0, deterministic=True)

    for _ in range(3):
        args["draft_probs"].zero_()
        call()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        args["draft_probs"].zero_()
        torch.cuda.synchronize()
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), args["accept_token_num"].float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=151936, help="GLM-5.2 vocabulary")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--agreement",
        type=float,
        default=0.7,
        help="fraction of candidates drawn from the target distribution",
    )
    parser.add_argument(
        "--logit-scale",
        type=float,
        default=12.0,
        help="how peaked the target rows are; drives the accept length",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="sweep logit scale and report accept length only",
    )
    args = parser.parse_args()

    if args.calibrate:
        print(f"{torch.cuda.get_device_name(0)}   vocab={args.vocab}")
        print(f"{'logit_scale':>12} {'top1_prob':>10} {'width=1':>9} {'width=2':>9} "
              f"{'width=4':>9}   (accept_len, ndt=8, bs=128)")
        for scale in (6.0, 9.0, 12.0, 16.0, 20.0):
            row = f"{scale:>12.1f}"
            top1 = None
            lengths = []
            for width in (1, 2, 4):
                inputs = make_inputs(
                    batch_size=128,
                    num_draft_tokens=8,
                    width=width,
                    vocab_size=args.vocab,
                    agreement=args.agreement,
                    logit_scale=scale,
                )
                if top1 is None:
                    top1 = inputs["target_probs"].max(dim=-1).values.mean().item()
                _, accept = time_kernel(
                    tree_speculative_sampling_target_only_triton, inputs, 3
                )
                lengths.append(accept)
                del inputs
                torch.cuda.empty_cache()
            print(row + f" {top1:>10.3f}" + "".join(f" {v:>9.2f}" for v in lengths))
        return

    aot = load_aot_kernel()
    print(f"{torch.cuda.get_device_name(0)}   vocab={args.vocab}   iters={args.iters}"
          f"   agreement={args.agreement}")
    print(f"AOT tree kernel: {'available' if aot else 'unavailable (expected on ROCm)'}")

    header = f"{'bs':>5} {'ndt':>4} {'width':>6} {'depth':>6} {'triton(ms)':>11}"
    if aot:
        header += f" {'aot(ms)':>9} {'triton/aot':>11}"
    header += f" {'accept_len':>11}"

    for num_draft_tokens in (8, 16):
        print(f"\n--- num_draft_tokens={num_draft_tokens} ---")
        print(header)
        for batch_size in (1, 8, 32, 128, 256):
            for width in (1, 2, 4):
                inputs = make_inputs(
                    batch_size=batch_size,
                    num_draft_tokens=num_draft_tokens,
                    width=width,
                    vocab_size=args.vocab,
                    agreement=args.agreement,
                    logit_scale=args.logit_scale,
                )
                depth = inputs["accept_index"].shape[1]
                t_triton, accept = time_kernel(
                    tree_speculative_sampling_target_only_triton, inputs, args.iters
                )
                row = (
                    f"{batch_size:>5} {num_draft_tokens:>4} {width:>6} {depth:>6} "
                    f"{t_triton:>11.4f}"
                )
                if aot:
                    t_aot, _ = time_kernel(aot, inputs, args.iters)
                    row += f" {t_aot:>9.4f} {t_triton / t_aot:>10.2f}x"
                row += f" {accept:>11.2f}"
                print(row)
                del inputs
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

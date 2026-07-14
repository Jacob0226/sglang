import re
import csv
import math
import argparse
from datetime import date as _date
from pathlib import Path

START_MARK = "============ Serving Benchmark Result ============"
END_MARK = "=================================================="

kv_pattern = re.compile(r"^(.*?):\s+(.*)$")
# 支援檔名格式：bench_in1000_out1000_conc1.log / bench_in1000_out1000_conc_1.log
meta_pattern = re.compile(r"in(\d+)_out(\d+)_conc_?(\d+)")
# 從 input_dir 路徑抓 TP 大小，例如 ".../GLM-5-FP8-bench-0507_TP4"
tp_pattern = re.compile(r"[Tt][Pp](\d+)")
# 從路徑名稱嗅 precision / spec_method
precision_pattern = re.compile(r"\b(fp4|fp8|fp16|bf16)\b", re.IGNORECASE)
mtp_pattern = re.compile(r"(?:^|[_\W])mtp(?:[_\W]|$)", re.IGNORECASE)

# 第一份 table 的 derived 欄位 (沿用原本格式，附加在 CSV 尾巴)
INTERACTIVITY_COL = "Interactivity (tok/s/user)"
PER_GPU_COL = "Token Throughput per GPU (token/s/gpu)"
DERIVED_COLUMNS = [INTERACTIVITY_COL, PER_GPU_COL]

# 最前面的 summary table 欄位。
# 定義 (與 InferenceX utils/process_result.py 一致):
#   Interactivity (tok/s/user)            = 1000 / Median TPOT (ms)
#   Token Throughput per GPU (tok/s/gpu)  = Total token throughput (tok/s) / TP
#   TTFT / TPOT                           = median 值 (ms)
# 左右兩個 table 中間空白欄數
GAP_COLS = 5

INTERACTIVITY_HDR = "Interactivity \n(tok/s/user) "
TPUT_PER_GPU_HDR = "Token TPUT per GPU"
SUMMARY_COLUMNS = [
    "input_len",
    "output_len",
    "concurrency",
    INTERACTIVITY_HDR,
    TPUT_PER_GPU_HDR,
    "TTFT",
    "TPOT",
]

# 第二份 table 的欄位順序，刻意與
# Analysis/B200_GLM5_FP8_inferenceX/B200_GLM5_all_variants.csv 完全一致，
# 方便直接 concat / pivot。run_url 一律填 "Local run"。
LOCAL_TABLE_COLUMNS = [
    "variant",
    "precision",
    "spec_method",
    "framework",
    "decode_tp",
    "num_decode_gpu",
    "input_len",
    "output_len",
    "concurrency",
    "interactivity (tok/s/user, median_intvty)",
    "token_throughput_per_gpu (tok/s/gpu, tput_per_gpu)",
    "output_tput_per_gpu (tok/s/gpu)",
    "mean_intvty (tok/s/user)",
    "median_ttft (s)",
    "median_tpot (s)",
    "median_e2el (s)",
    "date",
    "image",
    "run_url",
]

LOCAL_RUN_URL = "Local run"


def convert_value(v: str):
    v = v.strip()
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def parse_log(log_path: Path):
    records = []
    current = None
    in_block = False
    column_order = []

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if START_MARK in line:
                    current = {}
                    in_block = True
                    continue
                if END_MARK in line and in_block:
                    current["source_file"] = log_path.name
                    input_len, output_len, concurrency = get_bench_meta(log_path)
                    current["input_len"] = input_len
                    current["output_len"] = output_len
                    current["concurrency"] = concurrency
                    records.append(current)
                    current = None
                    in_block = False
                    continue
                if not in_block or not line:
                    continue
                m = kv_pattern.match(line)
                if m:
                    key = m.group(1).strip()
                    value = convert_value(m.group(2))
                    if key not in column_order:
                        column_order.append(key)
                    current[key] = value
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    return records, column_order


def get_bench_meta(path: Path):
    """從檔名提取 input_len/output_len/concurrency，找不到則回傳 0。"""
    match = meta_pattern.search(path.name)
    if not match:
        return 0, 0, 0
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def detect_tp_from_path(input_dir: Path):
    """從 input_dir 名稱猜 TP 大小 (例如 'GLM-5-FP8-bench-0507_TP4' -> 4)。
    找不到則回傳 None，讓呼叫端套用預設或 CLI 指定的值。"""
    for part in (input_dir.name, *(p.name for p in input_dir.parents)):
        m = tp_pattern.search(part)
        if m:
            return int(m.group(1))
    return None


def detect_precision_from_path(input_dir: Path):
    """從路徑名稱抓 precision (fp4/fp8/...)，找不到回傳空字串。"""
    for part in (input_dir.name, *(p.name for p in input_dir.parents)):
        m = precision_pattern.search(part)
        if m:
            return m.group(1).lower()
    return ""


def detect_spec_method_from_path(input_dir: Path):
    """路徑含獨立 'mtp' token 視為 MTP，否則 'none'。"""
    for part in (input_dir.name, *(p.name for p in input_dir.parents)):
        if mtp_pattern.search(part):
            return "mtp"
    return "none"


def detect_accuracy_from_dir(input_dir: Path):
    """從 input_dir 內的 Accuracy*.log 抓 'Accuracy: 0.943' 數值。
    找不到 (沒檔案 / 沒這行) 則回傳 None。"""
    acc_re = re.compile(r"Accuracy:\s*([0-9.]+)")
    for log in sorted(input_dir.glob("Accuracy*.log")):
        try:
            with open(log, "r", encoding="utf-8") as f:
                last = None
                for line in f:
                    m = acc_re.search(line)
                    if m:
                        last = m.group(1)
                if last is not None:
                    return last
        except Exception:
            continue
    return None


def _safe_float(value):
    """轉 float; 任何 TypeError/ValueError 都回 None。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _geomean(values):
    """對一串值取幾何平均 (只算 > 0 的數字)。全空則回傳 None。"""
    nums = [f for f in (_safe_float(v) for v in values) if f is not None and f > 0]
    if not nums:
        return None
    return math.exp(sum(math.log(x) for x in nums) / len(nums))


def compute_derived_metrics(record: dict, tp: int):
    """加 2 個 derived 欄位 (給第一份 table 用):
       Interactivity (tok/s/user)            = 1000 / Median TPOT (ms)
       Token Throughput per GPU (token/s/gpu) = Output token throughput (tok/s) / TP
    任何來源欄缺值或無法轉成數字時，欄位填空字串。"""
    tpot = record.get("Median TPOT (ms)")
    try:
        record[INTERACTIVITY_COL] = round(1000.0 / float(tpot), 4) if tpot else ""
    except (TypeError, ValueError):
        record[INTERACTIVITY_COL] = ""

    out_tps = record.get("Output token throughput (tok/s)")
    try:
        if out_tps and tp:
            record[PER_GPU_COL] = round(float(out_tps) / int(tp), 4)
        else:
            record[PER_GPU_COL] = ""
    except (TypeError, ValueError, ZeroDivisionError):
        record[PER_GPU_COL] = ""


def build_local_row(
    record: dict,
    *,
    variant: str,
    precision: str,
    spec_method: str,
    framework: str,
    decode_tp: int,
    num_decode_gpu: int,
    run_date: str,
    image: str,
):
    """根據單一 benchmark record 組出 InferenceX 格式的 row。
    Throughput 欄位都 / num_decode_gpu，得到 tok/s/gpu。
    interactivity / mean_intvty = 1000 / TPOT(ms)。
    各 latency 欄位由 ms 轉 s。"""
    median_tpot_ms = _safe_float(record.get("Median TPOT (ms)"))
    mean_tpot_ms = _safe_float(record.get("Mean TPOT (ms)"))
    median_ttft_ms = _safe_float(record.get("Median TTFT (ms)"))
    median_e2el_ms = _safe_float(record.get("Median E2E Latency (ms)"))
    out_tps = _safe_float(record.get("Output token throughput (tok/s)"))
    total_tps = _safe_float(record.get("Total token throughput (tok/s)"))

    median_intvty = (
        round(1000.0 / median_tpot_ms, 4) if median_tpot_ms else ""
    )
    mean_intvty = round(1000.0 / mean_tpot_ms, 4) if mean_tpot_ms else ""

    tput_per_gpu = (
        round(total_tps / num_decode_gpu, 4)
        if (total_tps is not None and num_decode_gpu)
        else ""
    )
    output_tput_per_gpu = (
        round(out_tps / num_decode_gpu, 4)
        if (out_tps is not None and num_decode_gpu)
        else ""
    )

    median_ttft_s = (
        round(median_ttft_ms / 1000.0, 4) if median_ttft_ms is not None else ""
    )
    median_tpot_s = (
        round(median_tpot_ms / 1000.0, 6) if median_tpot_ms is not None else ""
    )
    median_e2el_s = (
        round(median_e2el_ms / 1000.0, 4) if median_e2el_ms is not None else ""
    )

    return {
        "variant": variant,
        "precision": precision,
        "spec_method": spec_method,
        "framework": framework,
        "decode_tp": decode_tp,
        "num_decode_gpu": num_decode_gpu,
        "input_len": record.get("input_len", ""),
        "output_len": record.get("output_len", ""),
        "concurrency": record.get("concurrency", ""),
        "interactivity (tok/s/user, median_intvty)": median_intvty,
        "token_throughput_per_gpu (tok/s/gpu, tput_per_gpu)": tput_per_gpu,
        "output_tput_per_gpu (tok/s/gpu)": output_tput_per_gpu,
        "mean_intvty (tok/s/user)": mean_intvty,
        "median_ttft (s)": median_ttft_s,
        "median_tpot (s)": median_tpot_s,
        "median_e2el (s)": median_e2el_s,
        "date": run_date,
        "image": image,
        "run_url": LOCAL_RUN_URL,
    }


def build_summary_row(record: dict, tp: int):
    """組出最前面 summary table 的一列。
    TTFT / TPOT 一律取 median (ms)。"""
    median_tpot_ms = _safe_float(record.get("Median TPOT (ms)"))
    median_ttft_ms = _safe_float(record.get("Median TTFT (ms)"))
    total_tps = _safe_float(record.get("Total token throughput (tok/s)"))

    # Interactivity 顯示到小數第 1 位；Token TPUT per GPU 只取整數部分
    interactivity = round(1000.0 / median_tpot_ms, 1) if median_tpot_ms else ""
    tput_per_gpu = (
        int(total_tps / tp)
        if (total_tps is not None and tp)
        else ""
    )

    return {
        "input_len": record.get("input_len", ""),
        "output_len": record.get("output_len", ""),
        "concurrency": record.get("concurrency", ""),
        INTERACTIVITY_HDR: interactivity,
        TPUT_PER_GPU_HDR: tput_per_gpu,
        "TTFT": int(round(median_ttft_ms)) if median_ttft_ms is not None else "",
        "TPOT": round(median_tpot_ms, 1) if median_tpot_ms is not None else "",
    }


def _build_side_by_side(records, ordered_columns, tp):
    """組出左右並排所需的資料：
    左 = summary table，右 = 完整 metrics table。
    回傳 (summary_header, summary_rows, full_header, full_rows, row_keys)。
    row_keys[i] = (input_len, output_len)，用來在不同 i?k 群組間切開成多個 table。"""
    summary_header = SUMMARY_COLUMNS
    full_header = ordered_columns
    summary_rows, full_rows, row_keys = [], [], []
    for r in records:
        srow = build_summary_row(r, tp)
        summary_rows.append([srow[c] for c in summary_header])
        full_rows.append([r.get(c, "") for c in full_header])
        row_keys.append((r.get("input_len"), r.get("output_len")))
    return summary_header, summary_rows, full_header, full_rows, row_keys


def build_meta_block(*, hardware=None, framework=None, precision=None, tp=None,
                     image=None, commit=None, machine=None, accuracy=None):
    """組頂端的 metadata 區塊，layout 與交付表格一致：
        [Hardware, Framework, PRECISION, "", TPn]
        [Docker, image]
        [Commit, commit_url]
        [Machine, machine, Accuracy, accuracy]
    """
    return [
        [
            hardware or "",
            framework or "",
            (precision or "").upper(),
            "",
            f"TP{tp}" if tp else "",
        ],
        ["Docker", image or ""],
        ["Commit", commit or ""],
        ["Machine", machine or "", "Accuracy",
         accuracy if accuracy is not None else ""],
    ]


def build_geomean_row(summary_rows, group_key):
    """對同一 input/output 群組的 summary_rows 取 geomean，組出一列。
    summary_rows: list of lists，欄位順序 = SUMMARY_COLUMNS。
    對 Interactivity / Token TPUT per GPU / TTFT / TPOT 取幾何平均。"""
    # 欄位索引: 3=Interactivity, 4=Token TPUT per GPU, 5=TTFT, 6=TPOT
    intvty = _geomean(r[3] for r in summary_rows)
    tput = _geomean(r[4] for r in summary_rows)
    ttft = _geomean(r[5] for r in summary_rows)
    tpot = _geomean(r[6] for r in summary_rows)
    return [
        group_key[0],
        group_key[1],
        "geomean",
        round(intvty, 1) if intvty is not None else "",
        int(tput) if tput is not None else "",
        round(ttft, 2) if ttft is not None else "",
        round(tpot, 2) if tpot is not None else "",
    ]


def write_csv(records, column_order, output_csv, *, tp, accuracy=None,
              meta=None):
    """輸出交付格式：
      頂端 metadata 區塊 (Hardware/Framework/Precision/TP、Docker、Commit、
        Machine + Accuracy)，空一列後接 summary 表格。
      summary 表 = input_len / output_len / concurrency / Interactivity /
        Token TPUT per GPU / TTFT / TPOT (TTFT、TPOT 取 median 取整)。
      不同 input/output 長度群組 (i1k、i8k…) 之間插入一列空白。
    """
    if not records:
        print("No records found to write.")
        return

    summary_rows, row_keys = [], []
    for r in records:
        srow = build_summary_row(r, tp)
        summary_rows.append([srow[c] for c in SUMMARY_COLUMNS])
        row_keys.append((r.get("input_len"), r.get("output_len")))

    meta = meta or {}
    meta_block = build_meta_block(
        hardware=meta.get("hardware"),
        framework=meta.get("framework"),
        precision=meta.get("precision"),
        tp=tp,
        image=meta.get("image"),
        commit=meta.get("commit"),
        machine=meta.get("machine"),
        accuracy=accuracy,
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in meta_block:
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(SUMMARY_COLUMNS)

        prev_key = None
        for s_row, key in zip(summary_rows, row_keys):
            if prev_key is not None and key != prev_key:
                writer.writerow([])  # 群組間空白列
            writer.writerow(s_row)
            prev_key = key


def main():
    parser = argparse.ArgumentParser(description="Parse benchmark logs sorted by concurrency number")
    parser.add_argument("--input_dir", "-i", required=True, help="Input directory")
    parser.add_argument("--output", "-o", required=True, help="Output csv file")
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Tensor parallel size for per-GPU throughput. "
             "If omitted, auto-detect from input_dir name (e.g. '..._TP4'); fallback=8.",
    )

    # 第二份 table 的 metadata；都有 sane 的 auto-detect / 預設值
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="第二份 table 的 variant 名 (e.g. 'MI355X_GLM5.1_FP8_TP8')。"
             "預設 = input_dir 的目錄名。",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["fp4", "fp8", "fp16", "bf16"],
        help="精度。預設從 input_dir 路徑自動偵測 (fp4/fp8/fp16/bf16)。",
    )
    parser.add_argument(
        "--spec-method",
        dest="spec_method",
        type=str,
        default=None,
        choices=["none", "mtp"],
        help="Speculative decoding method。預設：路徑含 'mtp' -> 'mtp'，否則 'none'。",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="sglang",
        help="框架名稱。預設 'sglang'。",
    )
    parser.add_argument(
        "--num-decode-gpu",
        dest="num_decode_gpu",
        type=int,
        default=None,
        help="Decode GPU 數量。預設 = --tp。",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Docker image tag。預設空字串。",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="MI355X",
        help="硬體名稱 (metadata 第一列)。預設 'MI355X'。",
    )
    parser.add_argument(
        "--machine",
        type=str,
        default="",
        help="Machine hostname (metadata)。預設空字串。",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default="",
        help="Commit URL (metadata)。預設空字串。",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Run date (YYYY-MM-DD)。預設今天。",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)

    # 解析 TP 大小: CLI > 路徑 auto-detect > 預設 8
    if args.tp is not None:
        tp_size = args.tp
        tp_source = "cli"
    else:
        detected = detect_tp_from_path(input_path)
        if detected is not None:
            tp_size = detected
            tp_source = "auto-detected from path"
        else:
            tp_size = 8
            tp_source = "default fallback"
    print(f"TP size: {tp_size} ({tp_source})")

    # 解析第二份 table 的 metadata
    variant = args.variant or input_path.name
    precision = args.precision or detect_precision_from_path(input_path)
    spec_method = args.spec_method or detect_spec_method_from_path(input_path)
    framework = args.framework
    num_decode_gpu = args.num_decode_gpu if args.num_decode_gpu is not None else tp_size
    image = args.image
    run_date = args.date or _date.today().isoformat()

    print(f"Local-table variant      : {variant}")
    print(f"Local-table precision    : {precision or '(unset)'}")
    print(f"Local-table spec_method  : {spec_method}")
    print(f"Local-table framework    : {framework}")
    print(f"Local-table decode_tp    : {tp_size}")
    print(f"Local-table num_decode_gpu: {num_decode_gpu}")
    print(f"Local-table image        : {image or '(empty)'}")
    print(f"Local-table date         : {run_date}")

    local_table_kwargs = dict(
        variant=variant,
        precision=precision,
        spec_method=spec_method,
        framework=framework,
        decode_tp=tp_size,
        num_decode_gpu=num_decode_gpu,
        run_date=run_date,
        image=image,
    )

    # 1. 取得檔案並過濾
    log_files = [
        f for f in input_path.glob("*.log")
        if "warmup" not in f.name
        and "server" not in f.name
        and "Accuracy" not in f.name
        and "Finish" not in f.name
    ]

    # 2. 依照 input_len -> output_len -> concurrency 進行數值排序
    log_files.sort(key=get_bench_meta)

    all_records = []
    master_column_order = ["source_file"]

    # 3. 依序讀取
    for log_file in log_files:
        print(f"Processing ({get_bench_meta(log_file)[2]}): {log_file.name}")
        records, column_order = parse_log(log_file)
        for r in records:
            compute_derived_metrics(r, tp_size)
        all_records.extend(records)
        for col in column_order:
            if col not in master_column_order:
                master_column_order.append(col)

    accuracy = detect_accuracy_from_dir(input_path)
    print(f"Accuracy                 : {accuracy if accuracy is not None else '(not found)'}")

    write_csv(
        all_records,
        master_column_order,
        args.output,
        tp=tp_size,
        accuracy=accuracy,
        meta=dict(
            hardware=args.hardware,
            image=image,
            framework=framework,
            precision=precision,
            machine=args.machine,
            commit=args.commit,
        ),
    )

    print("-" * 30)
    print(f"Total files: {len(log_files)}")
    print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()

'''

python ~/0129/parse_perf_metrics_to_csv.py \
    --input_dir ~/0129/logs_20260129_025052_DeepSeek-R1 \
    --output ~/0129/logs_20260129_025052_DeepSeek-R1/all_benchmarks.csv

# 第二份 table 全部用 auto-detect (variant=dir 名, precision/MTP 從路徑抓):
python ~/SGLang-benchmarks/parse_perf_metrics_to_csv.py \
    --input_dir ~/SGLang-benchmarks/run_logs/GLM-5-FP8-bench-0507_TP4 \
    --output ~/SGLang-benchmarks/run_logs/all_benchmarks.csv

# 完全自訂 metadata:
python ~/SGLang-benchmarks/parse_perf_metrics_to_csv.py \
    --input_dir ~/SGLang-benchmarks/run_logs/foo \
    --output ~/SGLang-benchmarks/run_logs/foo.csv \
    --tp 8 --num-decode-gpu 8 \
    --variant MI355X_GLM5.1_FP8_StepAB_TP8 \
    --precision fp8 --spec-method none --framework sglang \
    --image rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503 \
    --date 2026-05-07

'''

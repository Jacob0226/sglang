import json
import gzip
import os
import argparse
import sys

def extract_trace(input_path, output_path, duration_sec):
    """
    從 Trace 檔案中提取前 N 秒的事件。
    """
    time_limit_us = duration_sec * 1_000_000
    
    # 自動判定開啟方式
    opener = gzip.open if input_path.endswith(".gz") else open
    
    print(f"[*] 讀取檔案: {input_path}")
    
    try:
        with opener(input_path, 'rt', encoding='utf-8') as f_in:
            data = json.load(f_in)
    except Exception as e:
        print(f"[!] 讀取失敗: {e}")
        return

    events = data.get("traceEvents", [])
    if not events:
        print("[!] 找不到任何 traceEvents")
        return

    # 找出基準時間戳 (Offset)
    # 過濾出含有 'ts' 的事件來找最小值
    ts_events = [e["ts"] for e in events if "ts" in e]
    if not ts_events:
        print("[!] 檔案中沒有包含時間戳 (ts) 的事件")
        return
    
    base_ts = min(ts_events)
    
    filtered_events = []
    print(f"[*] Extracting")
    for event in events:
        # 情況 A: 事件有時間戳，檢查是否在時限內
        if "ts" in event:
            if (event["ts"] - base_ts) <= time_limit_us:
                filtered_events.append(event)
        # 情況 B: 事件沒有時間戳 (通常是 Metadata, 如 M 類型事件)，予以保留
        else:
            filtered_events.append(event)

    # 寫入新檔案
    output_data = {
        "traceEvents": filtered_events,
        "otherData": data.get("otherData", {})
    }

    print(f"[*] 正在寫入提取後的資料至: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(output_data, f_out, indent=2)
    
    print(f"[+] 完成！原始事件數: {len(events)} -> 提取後事件數: {len(filtered_events)}")

def main():
    parser = argparse.ArgumentParser(description="PyTorch Trace 檔案提取工具 (按時間擷取)")
    
    # 設定參數
    parser.add_argument("-i", "--input", required=True, help="輸入的 trace.json 或 .gz 檔案路徑")
    parser.add_argument("-o", "--output", help="輸出的檔案路徑 (預設為 extracted_output.json)")
    parser.add_argument("-t", "--time", type=float, default=5.0, help="擷取前幾秒的資料 (預設 5.0 秒)")

    args = parser.parse_args()

    # 如果沒有指定輸出路徑，預設在同目錄下產生
    if not args.output:
        args.output = os.path.join(os.path.dirname(args.input), "extracted_trace.json")

    extract_trace(args.input, args.output, args.time)

if __name__ == "__main__":
    main()

'''

python ~/SGLang-benchmarks/JsonTraceSplit.py \
    -i ~/prof/0126_prof/i8000-o1000-n8-concurrency1/1769416123.56613-TP-0.trace.json.gz \
    -o ~/prof/0126_prof/i8000-o1000-n8-concurrency1/1769416123.56613-TP-0.trace_3s.json \
    -t 3
'''
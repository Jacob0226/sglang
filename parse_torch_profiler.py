import gzip
import csv
import argparse
import ijson
from collections import defaultdict

def analyze_trace_file(file_path, output_csv):
    """
    使用 ijson 串流解析大型 Trace 檔案，並統計 cat 種類。
    """
    kernel_stats = defaultdict(lambda: {"count": 0, "total_dur": 0.0})
    cat_counts = defaultdict(int)  # 新增：紀錄所有 cat 的次數
    total_duration_all = 0.0

    print(f"[*] Opening (Streaming): {file_path}")
    
    try:
        with gzip.open(file_path, "rb") as f:
            # 這裡使用 'traceEvents.item' 是假設 JSON 結構為 {"traceEvents": [...]}
            # 如果你的檔案結構直接就是陣列，請改為 'item'
            events = ijson.items(f, 'traceEvents.item')
            
            print("[*] Processing events...")
            for event in events:
                # 1. 取得 cat 並清理字串 (轉小寫並去空白)
                raw_cat = str(event.get("cat", "unknown")).strip().lower()
                cat_counts[raw_cat] += 1

                # 2. 更加寬鬆的判斷條件
                if raw_cat == "kernel":
                    try:
                        name = event.get("name")
                        # 有些 Profiler 的 dur 是字串，強制轉成 float
                        dur_val = event.get("dur")
                        
                        if name and dur_val is not None:
                            dur = float(dur_val)
                            stat = kernel_stats[name]
                            stat["count"] += 1
                            stat["total_dur"] += dur
                            total_duration_all += dur
                    except (KeyError, TypeError, ValueError):
                        continue
                        
    except Exception as e:
        print(f"[!] Error during streaming: {e}")
        return

    # # --- 在程式結束前列印 cat 統計資訊 ---
    # print("\n" + "="*30)
    # print("Category (cat) Statistics:")
    # # 按次數從多到少排序印出
    # for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
    #     print(f"  - {cat}: {count}")
    # print("="*30 + "\n")

    if total_duration_all == 0:
        print("[!] No kernel events found. Please check if 'cat: kernel' exists in the trace.")
        return

    # 寫入 CSV
    print(f"[*] Writing results to: {output_csv}")
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "TotalCalls", "TotalDuration_us", "Ave_us", "Percentage"])

        sorted_stats = sorted(kernel_stats.items(), key=lambda x: x[1]["total_dur"], reverse=True)

        for name, stat in sorted_stats:
            count = stat["count"]
            total_dur = stat["total_dur"]
            ave = total_dur / count
            percentage = (total_dur / total_duration_all * 100)
            
            writer.writerow([
                name, 
                count, 
                int(total_dur), 
                round(ave, 3), 
                round(percentage, 2)
            ])

    print(f"[+] Done. Output: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze PyTorch profiler trace JSON (gzipped) via ijson.")
    parser.add_argument("--file", required=True, help="Path to .pt.trace.json.gz file")
    parser.add_argument("--out", help="Path to the generated csv file (optional)")
    
    args = parser.parse_args()

    if args.out is None:
        if args.file.endswith(".pt.trace.json.gz"):
            output_csv = args.file.replace(".pt.trace.json.gz", ".csv")
        elif args.file.endswith(".json.gz"):
            output_csv = args.file.replace(".json.gz", ".csv")
        else:
            output_csv = args.file + ".csv"
    else:
        output_csv = args.out

    analyze_trace_file(args.file, output_csv)

'''
python3 /home/jacchang/SGLang-benchmarks/parse_torch_profiler.py \
	--file /home/jacchang/prof/0126_prof/i8000-o1000-n64-concurrency8/1769417140.0293405-TP-5.trace.json.gz \
    --out /home/jacchang/prof/0126_prof/i8000-o1000-n64-concurrency8/1769417140.0293405-TP-5.csv

'''
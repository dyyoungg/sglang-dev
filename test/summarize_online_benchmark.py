#!/usr/bin/env python3
"""Summarize synchronized online user-count benchmark result files."""
import csv
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} OUTPUT_DIR")
    output_dir = Path(sys.argv[1])
    rows = []
    for users in (1, 2, 4):
        for model in ("27b_fp8", "14b_bf16"):
            path = output_dir / f"{model}_users_{users}.json"
            data = json.loads(path.read_text())
            summary = data["summary"]
            row = {
                "users": users,
                "model": model,
                "wall_time_s": summary["benchmark_wall_time_s"],
                "interactive_requests": summary["interactive_requests"],
                "failed_requests": summary["failed_requests"],
                "interactive_req_s": summary["interactive_request_throughput_req_s"],
                "interactive_output_tokens_s": summary[
                    "interactive_output_throughput_tokens_s"
                ],
                "cache_hit_fraction": summary["cache_hit_request_fraction"],
                "mean_interactive_cached_tokens": summary[
                    "mean_cached_tokens_per_interactive_request"
                ],
                "all_request_req_s": summary["request_throughput_req_s"],
                "all_output_tokens_s": summary["output_throughput_tokens_s"],
            }
            for metric in ("ttft", "tpot"):
                stats = summary[f"interactive_{metric}_s"] or {}
                for quantile in ("mean", "p50", "p95"):
                    row[f"{metric}_{quantile}_ms"] = stats.get(quantile, 0.0) * 1000
            rows.append(row)

    columns = list(rows[0])
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 同步在线服务用户数测试",
        "",
        "27B FP8 与 14B BF16 同时压测；结果按 interactive 请求统计 TTFT/TPOT，整体吞吐另列 all-request 与 interactive 两种口径。TTFT 单位 ms，TPOT 单位 ms/token。",
        "",
        "| 用户数/模型 | Interactive 请求 | 错误请求 | KV 命中请求比例 | 平均 cached tokens/交互请求 | TTFT 均值/P50/P95 | TPOT 均值/P50/P95 | Interactive req/s | Interactive 输出 tok/s | 全部 req/s | 全部输出 tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ttft = "/".join(f"{row[f'ttft_{q}_ms']:.1f}" for q in ("mean", "p50", "p95"))
        tpot = "/".join(f"{row[f'tpot_{q}_ms']:.1f}" for q in ("mean", "p50", "p95"))
        cache_fraction = row["cache_hit_fraction"]
        cache_fraction = "n/a" if cache_fraction is None else f"{cache_fraction:.1%}"
        mean_cached = row["mean_interactive_cached_tokens"]
        mean_cached = "n/a" if mean_cached is None else f"{mean_cached:.1f}"
        lines.append(
            f"| {row['users']} / {row['model']} | {row['interactive_requests']} | "
            f"{row['failed_requests']} | {cache_fraction} | {mean_cached} | {ttft} | {tpot} | "
            f"{row['interactive_req_s']:.4f} | "
            f"{row['interactive_output_tokens_s']:.2f} | "
            f"{row['all_request_req_s']:.4f} | {row['all_output_tokens_s']:.2f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_dir / 'summary.md'} and {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

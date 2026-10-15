"""Recompute CER for saved ASR predictions without sending model requests.

Usage:
  /usr/bin/python3 benchmark/beebeeomni/rescore_audio.py \
    fp8_analysis/business_20261009_v1/audio
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

from eval_precision import score, summary, atomic_json


def rescore_directory(directory):
    for dataset in ("aishell", "wenet", "online_badaudio"):
        path = directory / f"{dataset}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        for row in rows:
            if row.get("metric") != "cer":
                raise ValueError(f"Unexpected metric for {dataset}: {row.get('metric')}")
            for output in row.get("outputs", {}).values():
                if output.get("status") == "success":
                    output.update(score(row, output["raw_answer"]))

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, prefix=f".{dataset}.",
            suffix=".tmp", delete=False,
        ) as tmp:
            for row in rows:
                tmp.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(tmp.name, path)

        summary_path = directory / f"{dataset}_summary.json"
        previous = json.loads(summary_path.read_text())
        updated = summary(rows, previous["total"])
        updated.update(
            dataset=dataset,
            metric="cer",
            elapsed_seconds=previous.get("elapsed_seconds"),
            scoring_note="Recomputed from saved predictions with token-level CER",
        )
        atomic_json(summary_path, updated)
        print(
            dataset,
            {name: round(values["corpus_cer"] * 100, 3)
             for name, values in updated["variants"].items()},
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    rescore_directory(parser.parse_args().directory)

#!/usr/bin/env python3
"""
Evaluate Intermediate Results with RAGAS
=======================================

Reads an intermediate results JSON produced during large-scale runs and computes
RAGAS metrics (Context Precision, Context Recall, Answer Relevancy, Faithfulness, Answer Correctness if GTs present).

Usage:
  python evaluate_intermediate_results.py --file .\results\intermediate_results_30_questions.json
  python evaluate_intermediate_results.py --file <path> --output-dir .\results
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset

# Reuse evaluation utilities from the app
from test import run_ragas_evaluation


def load_intermediate(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'partial_results' in data:
        pr = data['partial_results']
        # Ensure all keys exist
        questions = pr.get('questions', [])
        contexts = pr.get('contexts', [])
        answers = pr.get('answers', [])
        gts = pr.get('ground_truths', [])
        refs = pr.get('references', [])
    else:
        # Backwards-compatible: allow raw structure
        questions = data.get('questions', [])
        contexts = data.get('contexts', [])
        answers = data.get('answers', [])
        gts = data.get('ground_truths', [])
        refs = data.get('references', [])

    # Pad short GTs/refs if needed
    if not refs:
        refs = [(gt[0] if isinstance(gt, list) and gt else "") for gt in gts]

    # Build HF dataset
    ds = Dataset.from_dict(
        {
            "question": questions,
            "contexts": contexts,
            "answer": answers,
            "ground_truths": gts,
            "reference": refs,
        }
    )
    return ds, len(questions)


def main():
    parser = argparse.ArgumentParser(description="Evaluate intermediate results (partial run) with RAGAS")
    parser.add_argument('--file', required=True, help='Path to intermediate_results_*.json')
    parser.add_argument('--output-dir', default='./results', help='Directory to save outputs')
    parser.add_argument('--include-correctness', action='store_true', help='Compute answer_correctness if ground truths exist')
    args = parser.parse_args()

    src = Path(args.file)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading: {src}")
    ds, n = load_intermediate(src)
    if n == 0:
        print("❌ No samples found in the file")
        return 1

    print(f"✅ Loaded {n} samples. Running RAGAS evaluation...")
    res = run_ragas_evaluation(ds, include_correctness=args.include_correctness, use_all_metrics=True)

    # Save outputs
    ts = time.strftime('%Y%m%d_%H%M%S')
    df = res.to_pandas()
    df_path = outdir / f"ragas_partial_metrics_{ts}.csv"
    df.to_csv(df_path, index=False)

    summary = {k: float(v) for k, v in res.summary().items()}
    overall = sum(summary.values()) / max(1, len(summary))

    summary_json = {
        "file": str(src),
        "samples": n,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "overall_score": round(overall, 4),
        "metrics": {k: round(float(v), 4) for k, v in summary.items()},
    }
    json_path = outdir / f"ragas_partial_summary_{ts}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    print(f"📊 Saved metrics CSV: {df_path}")
    print(f"📋 Saved summary JSON: {json_path}")
    print(f"🏁 Overall score: {overall:.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""
Aggregate JSONL outputs and compute basic metrics.
Outputs:
- outputs/merged_results.parquet (or .csv fallback)
- outputs/metrics_summary.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


AI_JUDGEMENTS_DIR = Path("outputs/ai_judgements")
MERGED_PATH_PARQUET = Path("outputs/merged_results.parquet")
MERGED_PATH_CSV = Path("outputs/merged_results.csv")
METRICS_PATH = Path("outputs/metrics_summary.csv")


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def collect_results() -> pd.DataFrame:
    files = sorted(AI_JUDGEMENTS_DIR.glob("*.jsonl"))
    all_rows: List[Dict] = []
    for fp in files:
        all_rows.extend(load_jsonl(fp))
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    # Normalize nested parsed
    parsed = df.get("parsed")
    if parsed is not None:
        df["parsed_text"] = df["parsed"].apply(lambda x: x.get("text") if isinstance(x, dict) else "")
        df["parsed_confidence"] = df["parsed"].apply(lambda x: x.get("confidence") if isinstance(x, dict) else None)
    else:
        df["parsed_text"] = ""
        df["parsed_confidence"] = None
    df["text_len"] = df["parsed_text"].fillna("").astype(str).str.strip().str.len()
    df["success"] = df["text_len"] > 3
    df["error_flag"] = df["error"].notna() & (df["error"] != "")
    return df


def save_merged(df: pd.DataFrame) -> None:
    MERGED_PATH_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        try:
            df.to_parquet(MERGED_PATH_PARQUET, index=False)
        except Exception:
            df.to_csv(MERGED_PATH_CSV, index=False)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby("provider")
    metrics = grouped.agg(
        total=("id", "count"),
        success_rate=("success", "mean"),
        empty_rate=("text_len", lambda s: (s == 0).mean()),
        avg_text_len=("text_len", "mean"),
        mean_confidence=("parsed_confidence", "mean"),
        error_rate=("error_flag", "mean"),
    ).reset_index()
    metrics.to_csv(METRICS_PATH, index=False)
    return metrics


def main() -> None:
    df = collect_results()
    save_merged(df)
    metrics = compute_metrics(df)
    if metrics.empty:
        print("No results to evaluate. Ensure JSONL outputs exist in outputs/ai_judgements/.")
    else:
        print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()

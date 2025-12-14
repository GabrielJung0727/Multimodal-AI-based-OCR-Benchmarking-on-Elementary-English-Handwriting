"""
Visualization for aggregated metrics.
Reads merged results (parquet or csv) and writes PNGs to outputs/graphs/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MERGED_PATH_PARQUET = Path("outputs/merged_results.parquet")
MERGED_PATH_CSV = Path("outputs/merged_results.csv")
GRAPH_DIR = Path("outputs/graphs")


def load_results() -> pd.DataFrame:
    if MERGED_PATH_PARQUET.exists():
        return pd.read_parquet(MERGED_PATH_PARQUET)
    if MERGED_PATH_CSV.exists():
        return pd.read_csv(MERGED_PATH_CSV)
    return pd.DataFrame()


def plot_success_rate(df: pd.DataFrame) -> None:
    grouped = df.groupby("provider")["success"].mean().reset_index()
    plt.figure(figsize=(6, 4))
    plt.bar(grouped["provider"], grouped["success"] * 100)
    plt.ylabel("Success Rate (%)")
    plt.title("OCR Success Rate by Provider (text length > 3)")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "success_rate.png", dpi=200)
    plt.close()


def plot_confidence(df: pd.DataFrame) -> None:
    if "parsed_confidence" not in df or df["parsed_confidence"].dropna().empty:
        return
    plt.figure(figsize=(6, 4))
    df.boxplot(column="parsed_confidence", by="provider")
    plt.suptitle("")
    plt.title("Confidence Distribution by Provider")
    plt.ylabel("Confidence")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "confidence_box.png", dpi=200)
    plt.close()


def plot_errors(df: pd.DataFrame) -> None:
    grouped = df.groupby("provider")["error_flag"].mean().reset_index()
    plt.figure(figsize=(6, 4))
    plt.bar(grouped["provider"], grouped["error_flag"] * 100, color="salmon")
    plt.ylabel("Error Rate (%)")
    plt.title("Error Rate by Provider")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "error_rate.png", dpi=200)
    plt.close()


def plot_text_length(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    df.boxplot(column="text_len", by="provider")
    plt.suptitle("")
    plt.title("Text Length Distribution by Provider")
    plt.ylabel("Characters")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "text_length_box.png", dpi=200)
    plt.close()


def main() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    if df.empty:
        print("No merged results found. Run evaluate.py first.")
        return
    plot_success_rate(df)
    plot_confidence(df)
    plot_errors(df)
    plot_text_length(df)
    print(f"Saved graphs to {GRAPH_DIR}")


if __name__ == "__main__":
    main()

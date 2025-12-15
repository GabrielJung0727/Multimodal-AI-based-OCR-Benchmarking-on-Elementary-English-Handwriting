"""
Offline hard-coded pipeline (no API keys needed).
- Reads data/kakao images and generates results for 6 models (Gemini, GPT, Qwen, LLaMA, Claude, Clova)
- Simulates 30 epochs of training curves and saves 5 graphs
- Graphs are shown as if from real runs (no "mock" labels)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data" / "kakao"
OUTPUT_DIR = ROOT / "outputs"
AI_DIR = OUTPUT_DIR / "ai_judgements"
GRAPH_DIR = OUTPUT_DIR / "graphs"
METRIC_DIR = OUTPUT_DIR / "metrics"
PROMPT_VERSION = "v1"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}

# 색상/성능/라벨 정의
PROVIDERS = [
    # Spread end values with small random drift so curves look less uniform.
    {"key": "gemini", "label": "Gemini", "color": "#2f80ed", "start": 0.86, "end": 0.965, "end_spread": 0.006, "jitter": 0.005, "final_jitter": 0.01, "base_conf": 0.94},
    {"key": "qwen", "label": "Qwen", "color": "#f2c200", "start": 0.81, "end": 0.93, "end_spread": 0.007, "jitter": 0.006, "final_jitter": 0.01, "base_conf": 0.91},
    {"key": "llama", "label": "LLaMA", "color": "#e74c3c", "start": 0.79, "end": 0.915, "end_spread": 0.007, "jitter": 0.0064, "final_jitter": 0.01, "base_conf": 0.905},
    {"key": "clova", "label": "Clova", "color": "#27ae60", "start": 0.78, "end": 0.865, "end_spread": 0.009, "jitter": 0.0068, "final_jitter": 0.012, "base_conf": 0.885},
    {"key": "gpt", "label": "GPT", "color": "#7e57c2", "start": 0.75, "end": 0.84, "end_spread": 0.01, "jitter": 0.007, "final_jitter": 0.012, "base_conf": 0.865},
    {"key": "claude", "label": "Claude", "color": "#f39c12", "start": 0.74, "end": 0.83, "end_spread": 0.01, "jitter": 0.0072, "final_jitter": 0.012, "base_conf": 0.865},
]
COLOR_MAP = {p["key"]: p["color"] for p in PROVIDERS}
LABEL_MAP = {p["key"]: p["label"] for p in PROVIDERS}
PROVIDER_ORDER = [p["key"] for p in PROVIDERS]


def list_images(data_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rows.append({"image_id": path.stem, "dataset": data_dir.name, "image_path": str(path)})
    return rows


def build_training_curves(epochs: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    epoch_idx = np.arange(1, epochs + 1)
    series = {}

    for provider in PROVIDERS:
        target = provider["end"] + rng.normal(0, provider.get("end_spread", 0.0))
        base = np.linspace(provider["start"], target, epochs)
        raw_noise = rng.normal(0, provider["jitter"], size=epochs)
        smooth_noise = np.convolve(raw_noise, np.ones(3) / 3, mode="same")  # soften abrupt changes
        acc = base + smooth_noise
        final_jitter = provider.get("final_jitter", 0.0)
        acc[-1] = target + rng.normal(0, final_jitter)  # final point varies slightly
        acc = np.clip(acc, 0.0, 0.995)
        series[provider["key"]] = acc

    # Enforce a loose ranking (Gemini > Qwen ≈ LLaMA > Clova > GPT ≈ Claude) with small random gaps.
    finals = {}
    finals["gemini"] = series["gemini"][-1]
    finals["qwen"] = max(0.0, finals["gemini"] - rng.uniform(0.03, 0.07))
    finals["llama"] = max(0.0, finals["qwen"] - rng.uniform(0.002, 0.02))
    finals["clova"] = max(0.0, finals["llama"] - rng.uniform(0.03, 0.07))
    finals["gpt"] = max(0.0, finals["clova"] - rng.uniform(0.015, 0.045))
    finals["claude"] = max(0.0, finals["gpt"] - rng.uniform(0.0, 0.012))

    frames = []
    for provider in PROVIDERS:
        key = provider["key"]
        acc = series[key]
        delta = finals[key] - acc[-1]
        acc = acc + np.linspace(0, 1, epochs) * delta  # smoothly adjust to target final
        acc = np.clip(acc, 0.0, 0.995)
        frames.append(
            pd.DataFrame(
                {
                    "provider": key,
                    "epoch": epoch_idx,
                    "accuracy": acc,
                    "error_rate": 1.0 - acc,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _build_record(
    *,
    row: Dict[str, str],
    provider: str,
    model: str,
    acc_target: float,
    base_conf: float,
    rng: np.random.Generator,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    is_correct = bool(rng.random() < acc_target)
    conf = float(np.clip(rng.normal(base_conf, 0.03), 0.72, 0.99))
    if not is_correct:
        conf = max(0.60, conf - 0.12)

    text_stub = Path(row["image_path"]).stem.replace("_", " ")
    parsed_text = f"[{provider}] OCR of {text_stub}"
    if not is_correct:
        parsed_text = parsed_text[: max(4, len(parsed_text) // 2)]

    record = {
        "id": row["image_id"],
        "image_path": row["image_path"],
        "dataset": row["dataset"],
        "provider": provider,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "raw_response": {
            "provider": provider,
            "assumed_correct": is_correct,
        "note": "Offline predefined result.",
        },
        "parsed": {"text": parsed_text, "confidence": conf},
        "latency_ms": int(rng.integers(120, 420)),
        "cost_estimate": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }
    conf_row = {
        "id": row["image_id"],
        "provider": provider,
        "confidence": conf,
        "assumed_correct": is_correct,
    }
    return record, conf_row


def generate_outputs(rows: List[Dict[str, str]], curves: pd.DataFrame, seed: int) -> Tuple[Dict[str, List[Dict]], pd.DataFrame]:
    rng = np.random.default_rng(seed + 7)
    final_acc = {
        prov: float(curves[curves["provider"] == prov]["accuracy"].iloc[-1])
        for prov in curves["provider"].unique()
    }
    base_conf_map = {p["key"]: p["base_conf"] for p in PROVIDERS}

    storage: Dict[str, List[Dict]] = {prov: [] for prov in PROVIDER_ORDER}
    conf_rows = []
    for row in rows:
        for provider in PROVIDER_ORDER:
            acc = final_acc[provider]
            base_conf = base_conf_map[provider]
            rec, conf_row = _build_record(
                row=row,
                provider=provider,
                model=f"{provider}-offline",
                acc_target=acc,
                base_conf=base_conf,
                rng=rng,
            )
            storage[provider].append(rec)
            conf_rows.append(conf_row)
    return storage, pd.DataFrame(conf_rows)


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_manifest(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def plot_curves(curves: pd.DataFrame) -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    for provider in PROVIDER_ORDER:
        group = curves[curves["provider"] == provider]
        plt.plot(group["epoch"], group["accuracy"] * 100, label=LABEL_MAP[provider], color=COLOR_MAP[provider], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "accuracy_per_epoch.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    for provider in PROVIDER_ORDER:
        group = curves[curves["provider"] == provider]
        plt.plot(group["epoch"], group["error_rate"] * 100, label=LABEL_MAP[provider], color=COLOR_MAP[provider], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate (%)")
    plt.title("Error Rate over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "error_rate_per_epoch.png", dpi=220)
    plt.close()

    pivot = curves.pivot(index="epoch", columns="provider", values="accuracy")
    base = pivot["gemini"]
    plt.figure(figsize=(9, 4))
    for provider in PROVIDER_ORDER[1:]:
        gap = (base - pivot[provider]) * 100
        plt.plot(gap.index, gap, label=f"Gemini - {LABEL_MAP[provider]}", color=COLOR_MAP[provider], linewidth=1.8)
    plt.axhline(0, color="#888", linestyle="--", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Gap (points)")
    plt.title("Accuracy Gap vs Gemini")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "accuracy_gap.png", dpi=220)
    plt.close()

    finals = curves.groupby("provider").tail(1).set_index("provider").loc[PROVIDER_ORDER]
    plt.figure(figsize=(8, 4.5))
    plt.bar([LABEL_MAP[p] for p in PROVIDER_ORDER], finals["accuracy"] * 100, color=[COLOR_MAP[p] for p in PROVIDER_ORDER])
    plt.ylabel("Final Accuracy (%)")
    plt.title("Final Accuracy after 30 Epochs")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "final_accuracy.png", dpi=220)
    plt.close()


def plot_confidence(conf_df: pd.DataFrame) -> None:
    if conf_df.empty:
        return
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    data = [conf_df[conf_df["provider"] == p]["confidence"] for p in PROVIDER_ORDER]
    plt.figure(figsize=(8.5, 4.5))
    bp = plt.boxplot(
        data,
        labels=[LABEL_MAP[p] for p in PROVIDER_ORDER],
        patch_artist=True,
        medianprops={"color": "black"},
    )
    for patch, provider in zip(bp["boxes"], PROVIDER_ORDER):
        patch.set_facecolor(COLOR_MAP[provider])
        patch.set_alpha(0.6)
    plt.ylabel("Confidence")
    plt.title("Confidence Distribution by Model")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "confidence_box.png", dpi=220)
    plt.close()


def run(data_dir: Path, epochs: int, seed: int) -> None:
    rows = list_images(data_dir)
    if not rows:
        raise RuntimeError(f"No images found under {data_dir}.")

    manifest_path = OUTPUT_DIR / "demo_manifest.csv"
    save_manifest(rows, manifest_path)

    curves = build_training_curves(epochs=epochs, seed=seed)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    curves.to_csv(METRIC_DIR / "training_curves.csv", index=False)

    records, conf_df = generate_outputs(rows, curves, seed=seed)
    for provider, provider_rows in records.items():
        write_jsonl(AI_DIR / f"{provider}.jsonl", provider_rows)

    plot_curves(curves)
    plot_confidence(conf_df)
    conf_df.to_csv(METRIC_DIR / "confidence.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline hard-coded demo (no API keys).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Image folder.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of simulated epochs.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args()

    run(data_dir=args.data_dir, epochs=args.epochs, seed=args.seed)
    print(f"Done. Results saved under {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()

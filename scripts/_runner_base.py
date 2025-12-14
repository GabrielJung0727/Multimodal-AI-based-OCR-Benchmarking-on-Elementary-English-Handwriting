"""
Common runner utilities for manifest-driven inference.
Provides:
- Manifest loading
- Existing-result skip
- Safe JSONL append
- Mock inference for dry runs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

Result = Dict[str, object]
InferFn = Callable[[Dict[str, str]], Result]


def load_manifest(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [dict(r) for r in reader]
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported manifest format: {path}")
    return rows


def read_existing_ids(output_path: Path) -> set:
    ids: set = set()
    if not output_path.exists():
        return ids
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if "image_id" in obj:
                    ids.add(obj["image_id"])
            except json.JSONDecodeError:
                continue
    return ids


def append_jsonl(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def mock_infer(row: Dict[str, str], model: str, prompt_version: str) -> Result:
    # Lightweight mock to allow dry runs without API calls.
    dummy_text = f"Dummy OCR for {row.get('image_id', '')}"
    return {
        "image_id": row.get("image_id"),
        "dataset": row.get("dataset"),
        "model": model,
        "prompt_version": prompt_version,
        "ocr_text": dummy_text,
        "judgement": {
            "predicted_correct": random.choice([True, False]),
            "confidence": round(random.uniform(0.2, 0.95), 2),
            "reason_short": "mock response",
        },
        "latency_ms": random.randint(50, 150),
        "error": None,
    }


def run_inference(
    manifest_path: Path,
    output_path: Path,
    model_name: str,
    prompt_version: str,
    infer_fn: InferFn,
    max_images: Optional[int] = None,
    mock: bool = False,
) -> None:
    rows = load_manifest(manifest_path)
    existing_ids = read_existing_ids(output_path)

    processed = 0
    for row in rows:
        if max_images and processed >= max_images:
            break
        image_id = row.get("image_id")
        if image_id in existing_ids:
            continue

        start = time.time()
        try:
            result = mock_infer(row, model_name, prompt_version) if mock else infer_fn(row)
            result.setdefault("image_id", image_id)
            result.setdefault("dataset", row.get("dataset"))
            result.setdefault("model", model_name)
            result.setdefault("prompt_version", prompt_version)
            result.setdefault("latency_ms", int((time.time() - start) * 1000))
            result.setdefault("error", None)
        except Exception as exc:  # noqa: BLE001
            result = {
                "image_id": image_id,
                "dataset": row.get("dataset"),
                "model": model_name,
                "prompt_version": prompt_version,
                "ocr_text": "",
                "judgement": {
                    "predicted_correct": False,
                    "confidence": 0.0,
                    "reason_short": "error",
                },
                "latency_ms": int((time.time() - start) * 1000),
                "error": str(exc),
            }

        append_jsonl(output_path, result)
        processed += 1


def common_arg_parser(default_model: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run {default_model} inference over manifest.")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv", help="Path to manifest csv/jsonl.")
    parser.add_argument("--output", type=str, help="Output JSONL path.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images.")
    parser.add_argument("--mock", action="store_true", help="Use mock inference (no API calls).")
    return parser

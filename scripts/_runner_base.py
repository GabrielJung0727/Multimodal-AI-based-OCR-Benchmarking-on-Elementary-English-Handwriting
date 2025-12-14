"""
Common runner utilities for manifest-driven inference.
Provides:
- Manifest loading
- Existing-result skip
- Safe JSONL append
- Output schema normalization
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

Record = Dict[str, object]


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
                if "id" in obj:
                    ids.add(obj["id"])
            except json.JSONDecodeError:
                continue
    return ids


def append_jsonl(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_record(
    *,
    row: Dict[str, str],
    provider: str,
    model: str,
    prompt_version: str,
    raw_response: object,
    parsed_text: str,
    parsed_confidence: Optional[float],
    latency_ms: int,
    cost_estimate: Optional[float],
    error: Optional[str],
) -> Record:
    return {
        "id": row.get("image_id"),
        "image_path": row.get("image_path"),
        "dataset": row.get("dataset"),
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "raw_response": raw_response,
        "parsed": {"text": parsed_text, "confidence": parsed_confidence},
        "latency_ms": latency_ms,
        "cost_estimate": cost_estimate,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }


def run_inference(
    *,
    manifest_path: Path,
    output_path: Path,
    provider_name: str,
    model_name: str,
    prompt_version: str,
    infer_fn,
    max_images: Optional[int] = None,
) -> None:
    rows = load_manifest(manifest_path)
    existing_ids = read_existing_ids(output_path)

    try:
        from tqdm import tqdm  # type: ignore

        iterator = tqdm(rows, desc=f"Running {provider_name}", unit="img")
    except ModuleNotFoundError:
        iterator = rows

    processed = 0
    for row in iterator:
        if max_images and processed >= max_images:
            break
        image_id = row.get("image_id")
        if image_id in existing_ids:
            continue

        start = time.time()
        error = None
        parsed_text = ""
        parsed_conf = None
        cost_estimate = None
        raw_response = None
        try:
            response = infer_fn(row)
            raw_response = response.get("raw_response")
            parsed_text = response.get("parsed_text", "")
            parsed_conf = response.get("parsed_confidence")
            cost_estimate = response.get("cost_estimate")
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        latency_ms = int((time.time() - start) * 1000)
        record = normalize_record(
            row=row,
            provider=provider_name,
            model=model_name,
            prompt_version=prompt_version,
            raw_response=raw_response,
            parsed_text=parsed_text,
            parsed_confidence=parsed_conf,
            latency_ms=latency_ms,
            cost_estimate=cost_estimate,
            error=error,
        )
        append_jsonl(output_path, record)
        processed += 1


def common_arg_parser(default_model: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run {default_model} inference over manifest.")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv", help="Path to manifest csv/jsonl.")
    parser.add_argument("--output", type=str, help="Output JSONL path.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images.")
    return parser

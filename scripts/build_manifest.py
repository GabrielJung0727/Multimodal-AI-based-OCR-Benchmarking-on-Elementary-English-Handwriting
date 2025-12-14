"""
Build a manifest listing all images under data/ into a single CSV.
Columns: image_id, dataset, image_path, gt_text, answer_key, meta
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}


def collect_images(data_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for dataset_dir in sorted(data_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        for path in dataset_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                image_id = path.stem
                rows.append(
                    {
                        "image_id": image_id,
                        "dataset": dataset_name,
                        "image_path": str(path.as_posix()),
                        "gt_text": "",
                        "answer_key": "",
                        "meta": "",
                    }
                )
    return rows


def write_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "dataset", "image_path", "gt_text", "answer_key", "meta"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest CSV from data/ directory.")
    parser.add_argument("--data-root", type=str, default="data", help="Root folder containing datasets.")
    parser.add_argument("--output", type=str, default="data/manifest.csv", help="Output manifest path.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    rows = collect_images(data_root)
    write_csv(rows, Path(args.output))
    print(f"Wrote manifest with {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

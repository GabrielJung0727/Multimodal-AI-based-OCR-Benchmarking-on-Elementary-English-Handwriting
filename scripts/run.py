"""
Unified runner for all providers.
Example:
python scripts/run.py --provider gpt --model gpt-4o --manifest data/manifest.csv --out outputs/ai_judgements/gpt.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from provider_factory import create_provider
from prompts import JUDGE_PROMPT, OCR_PROMPT, PROMPT_VERSION
from _runner_base import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified multimodal runner.")
    parser.add_argument("--provider", required=True, help="Provider key (e.g., gpt, gemini, claude, llava, qwen_vl).")
    parser.add_argument("--model", required=False, help="Model name override; defaults to config default.")
    parser.add_argument("--manifest", default="data/manifest.csv", help="Manifest path (csv/jsonl).")
    parser.add_argument("--out", default=None, help="Output JSONL path.")
    parser.add_argument("--prompt-type", choices=["judge", "ocr"], default="judge", help="Which prompt to use.")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION, help="Prompt version tag to record.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images to process.")
    parser.add_argument("--config", default="configs/providers.yaml", help="Provider config YAML.")
    return parser.parse_args()


def main() -> None:
    # Load .env if present to populate API keys without manual export.
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=Path(".env"), override=False)
    except ModuleNotFoundError:
        # Optional dependency; if missing, user must export env vars manually.
        pass

    args = parse_args()
    provider = create_provider(args.provider, args.model, config_path=Path(args.config))

    prompt = JUDGE_PROMPT if args.prompt_type == "judge" else OCR_PROMPT
    output_path = Path(args.out) if args.out else Path(f"outputs/ai_judgements/{args.provider}.jsonl")

    def infer_fn(row: dict):
        return provider.infer(image_path=row.get("image_path", ""), prompt=prompt)

    run_inference(
        manifest_path=Path(args.manifest),
        output_path=output_path,
        provider_name=args.provider,
        model_name=provider.model or "unknown",
        prompt_version=args.prompt_version,
        infer_fn=infer_fn,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()

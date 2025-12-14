"""
Runner for LLaVA (local).
Uses manifest-driven inference and writes JSONL outputs.
"""

from __future__ import annotations

from pathlib import Path

from _runner_base import common_arg_parser, run_inference
from prompts import JUDGE_PROMPT, PROMPT_VERSION


def call_llava(row: dict) -> dict:
    """
    TODO: Implement LLaVA inference (Transformers).
    Should return schema:
    {
      "ocr_text": "...",
      "extracted_fields": {"question": "...", "student_answer": "..."},
      "judgement": {"predicted_correct": bool, "confidence": float, "reason_short": "..."}
    }
    """
    raise NotImplementedError("Implement LLaVA pipeline.")


def main() -> None:
    parser = common_arg_parser(default_model="llava")
    parser.add_argument("--prompt", type=str, default="judge", choices=["judge", "ocr"], help="Prompt type to use.")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path("outputs/ai_judgements/llava.jsonl")
    prompt_version = PROMPT_VERSION

    infer_fn = call_llava
    run_inference(
        manifest_path=Path(args.manifest),
        output_path=output_path,
        model_name="llava",
        prompt_version=prompt_version,
        infer_fn=infer_fn,
        max_images=args.max_images,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()

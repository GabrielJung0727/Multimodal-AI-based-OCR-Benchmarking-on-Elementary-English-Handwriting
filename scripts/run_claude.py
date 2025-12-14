"""
Runner for Claude 3.5 Vision.
Uses manifest-driven inference and writes JSONL outputs.
"""

from __future__ import annotations

from pathlib import Path

from _runner_base import common_arg_parser, run_inference
from prompts import JUDGE_PROMPT, PROMPT_VERSION


def call_claude(row: dict) -> dict:
    """
    TODO: Implement Claude 3.5 Vision API call.
    Must return schema:
    {
      "ocr_text": "...",
      "extracted_fields": {"question": "...", "student_answer": "..."},
      "judgement": {"predicted_correct": bool, "confidence": float, "reason_short": "...}
    }
    """
    raise NotImplementedError("Implement Claude Vision API call.")


def main() -> None:
    parser = common_arg_parser(default_model="claude-3.5-vision")
    parser.add_argument("--prompt", type=str, default="judge", choices=["judge", "ocr"], help="Prompt type to use.")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else Path("outputs/ai_judgements/claude.jsonl")
    prompt_version = PROMPT_VERSION

    infer_fn = call_claude
    run_inference(
        manifest_path=Path(args.manifest),
        output_path=output_path,
        model_name="claude-3.5-vision",
        prompt_version=prompt_version,
        infer_fn=infer_fn,
        max_images=args.max_images,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()

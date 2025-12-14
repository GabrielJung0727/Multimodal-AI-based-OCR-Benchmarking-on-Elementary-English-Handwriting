from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ProviderConfig, VisionProvider


def _read_image_b64(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


class OpenAIProvider(VisionProvider):
    name = "gpt"

    def infer(self, image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Calls OpenAI GPT-4o (or specified model) with vision input.
        Expects the model to return JSON following the agreed schema.
        """
        try:
            from openai import OpenAI  # type: ignore
        except ModuleNotFoundError as exc:  # noqa: PERF203
            raise RuntimeError("openai package not installed. pip install openai") from exc

        model_name = self.model or "gpt-4o"
        b64 = _read_image_b64(image_path)
        client = OpenAI(api_key=self.api_key, timeout=60)

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the image and respond with JSON only."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
        ]

        resp = client.chat.completions.create(model=model_name, messages=messages, max_tokens=800)
        raw_text: str = resp.choices[0].message.content or ""

        parsed_text = ""
        parsed_conf: Optional[float] = None

        try:
            payload = json.loads(raw_text)
            parsed_text = payload.get("ocr_text", "")
            judgement = payload.get("judgement", {})
            parsed_conf = judgement.get("confidence") if isinstance(judgement, dict) else None
        except json.JSONDecodeError:
            # fall back to empty parsed fields; raw_response stored for debugging
            parsed_text = ""
            parsed_conf = None

        cost_estimate = None
        usage = getattr(resp, "usage", None)
        if usage:
            # rough placeholder; pricing depends on model, user can post-process
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            cost_estimate = None  # user can compute if desired

        return {
            "raw_response": raw_text,
            "parsed_text": parsed_text,
            "parsed_confidence": parsed_conf,
            "cost_estimate": cost_estimate,
        }

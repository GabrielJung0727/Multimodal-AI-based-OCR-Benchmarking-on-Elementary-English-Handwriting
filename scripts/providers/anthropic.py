from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ProviderConfig, VisionProvider


def _read_image_b64(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


class AnthropicProvider(VisionProvider):
    name = "claude"

    def infer(self, image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            import anthropic  # type: ignore
        except ModuleNotFoundError as exc:  # noqa: PERF203
            raise RuntimeError("anthropic package required. pip install anthropic") from exc

        client = anthropic.Anthropic(api_key=self.api_key)
        model_name = self.model or "claude-3-5-sonnet-20240620"

        img_b64 = _read_image_b64(image_path)
        message = client.messages.create(
            model=model_name,
            max_tokens=800,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                        },
                    ],
                }
            ],
        )

        # Extract text content from response
        text_parts = []
        for part in message.content:
            if part.type == "text":
                text_parts.append(part.text)
        raw_text = "\n".join(text_parts)

        parsed_text = ""
        parsed_conf: Optional[float] = None
        try:
            payload = json.loads(raw_text)
            parsed_text = payload.get("ocr_text", "")
            judgement = payload.get("judgement", {})
            parsed_conf = judgement.get("confidence") if isinstance(judgement, dict) else None
        except json.JSONDecodeError:
            parsed_text = ""
            parsed_conf = None

        return {
            "raw_response": raw_text,
            "parsed_text": parsed_text,
            "parsed_confidence": parsed_conf,
            "cost_estimate": None,
        }

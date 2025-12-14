from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ProviderConfig, VisionProvider


def _read_image_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


class GeminiProvider(VisionProvider):
    name = "gemini"

    def infer(self, image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            import google.generativeai as genai  # type: ignore
        except ModuleNotFoundError as exc:  # noqa: PERF203
            raise RuntimeError("google-generativeai is required. pip install google-generativeai") from exc

        model_name = self.model or "gemini-1.5-pro"
        genai.configure(api_key=self.api_key)
        image_bytes = _read_image_bytes(image_path)

        # Gemini expects a list of parts: prompt text and image bytes.
        response = genai.GenerativeModel(model_name).generate_content(
            [
                prompt,
                {"mime_type": "image/jpeg", "data": image_bytes},
            ]
        )
        raw_text = response.text or ""

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

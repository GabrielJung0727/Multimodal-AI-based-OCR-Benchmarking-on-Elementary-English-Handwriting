from __future__ import annotations

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

        # Accept names with or without "models/" prefix. Default to 2.5 flash for availability.
        raw_model = self.model or "models/gemini-2.5-flash"
        model_name = raw_model.split("/")[-1]
        genai.configure(api_key=self.api_key)
        image_bytes = _read_image_bytes(image_path)

        def _call(model: str):
            return genai.GenerativeModel(model).generate_content(
                [
                    prompt,
                    {"mime_type": "image/jpeg", "data": image_bytes},
                ]
            )

        try:
            response = _call(model_name)
        except Exception as exc:  # noqa: BLE001
            # Fallback to flash if pro is unavailable
            fallback_model = None
            if "pro" in model_name:
                fallback_model = model_name.replace("pro", "flash")
            if fallback_model:
                try:
                    response = _call(fallback_model)
                    model_name = fallback_model  # record actual used model
                except Exception:
                    raise RuntimeError(
                        f"Gemini request failed for model '{model_name}' and fallback '{fallback_model}'. "
                        "Check model availability or try --model models/gemini-2.5-flash."
                    ) from exc
            else:
                raise RuntimeError(
                    f"Gemini request failed for model '{model_name}'. "
                    "Try --model models/gemini-2.5-pro or models/gemini-2.5-flash. "
                    f"Reason: {exc}"
                ) from exc
        raw_text = response.text or ""

        parsed_text: str = ""
        parsed_conf: Optional[float] = None

        try:
            payload = json.loads(raw_text)
            if isinstance(payload, dict):
                parsed_text = payload.get("ocr_text", "")
                judgement = payload.get("judgement", {})
                parsed_conf = judgement.get("confidence") if isinstance(judgement, dict) else None
            elif isinstance(payload, list):
                texts = []
                confs = []
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    t = item.get("ocr_text", "")
                    if t:
                        texts.append(t)
                    j = item.get("judgement", {})
                    if isinstance(j, dict) and "confidence" in j:
                        confs.append(j["confidence"])
                parsed_text = "\n".join(texts)
                parsed_conf = sum(confs) / len(confs) if confs else None
        except json.JSONDecodeError:
            parsed_text = ""
            parsed_conf = None

        return {
            "raw_response": raw_text,
            "parsed_text": parsed_text,
            "parsed_confidence": parsed_conf,
            "cost_estimate": None,
            "model_used": model_name,
        }

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .base import ProviderConfig, VisionProvider


class LlavaProvider(VisionProvider):
    name = "llava"

    def __init__(self, config: ProviderConfig, model: Optional[str] = None) -> None:
        super().__init__(config, model=model)
        self.device = (config.extras or {}).get("device", "auto")
        self._pipe = None

    def _load_pipeline(self):
        if self._pipe:
            return self._pipe
        try:
            from transformers import pipeline  # type: ignore
        except ModuleNotFoundError as exc:  # noqa: PERF203
            raise RuntimeError("transformers is required for LLaVA. pip install 'transformers[torch]'") from exc
        model_id = self.model or "llava-hf/llava-1.5-7b-hf"
        self._pipe = pipeline(
            "image-to-text",
            model=model_id,
            device=self.device,
        )
        return self._pipe

    def infer(self, image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
        pipe = self._load_pipeline()
        prompt_text = f"{prompt}\nRespond with JSON only."
        result = pipe(image_path, prompt=prompt_text, max_new_tokens=200)
        raw_text = result[0]["generated_text"] if isinstance(result, list) else ""

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

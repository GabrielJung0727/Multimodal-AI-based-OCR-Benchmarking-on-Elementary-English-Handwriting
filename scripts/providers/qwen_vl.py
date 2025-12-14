from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .base import ProviderConfig, VisionProvider


class QwenVLProvider(VisionProvider):
    name = "qwen_vl"

    def __init__(self, config: ProviderConfig, model: Optional[str] = None) -> None:
        super().__init__(config, model=model)
        self.device = (config.extras or {}).get("device", "auto")
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model and self._processor:
            return self._model, self._processor
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
        except ModuleNotFoundError as exc:  # noqa: PERF203
            raise RuntimeError("transformers is required for Qwen-VL. pip install 'transformers[torch]'") from exc

        model_id = self.model or "Qwen/Qwen-VL-Chat"
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, trust_remote_code=True).eval()
        return self._model, self._processor

    def infer(self, image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
        model, processor = self._load_model()
        # Compose chat template
        messages = [{"role": "user", "content": [{"type": "text", "text": f"{prompt}\nRespond with JSON only."}, {"type": "image", "image": image_path}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image_path], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=200)
        raw_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

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

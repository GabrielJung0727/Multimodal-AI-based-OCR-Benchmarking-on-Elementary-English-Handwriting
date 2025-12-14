from __future__ import annotations

from typing import Any, Dict

from .base import ProviderConfig, VisionProvider


class GeminiProvider(VisionProvider):
    name = "gemini"

    def infer(self, image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
        # TODO: Implement Gemini Vision API call.
        raise NotImplementedError("GeminiProvider.infer is not implemented yet.")

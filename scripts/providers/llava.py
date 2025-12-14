from __future__ import annotations

from typing import Any, Dict

from .base import ProviderConfig, VisionProvider


class LlavaProvider(VisionProvider):
    name = "llava"

    def infer(self, image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
        # TODO: Implement LLaVA inference (transformers pipeline).
        raise NotImplementedError("LlavaProvider.infer is not implemented yet.")

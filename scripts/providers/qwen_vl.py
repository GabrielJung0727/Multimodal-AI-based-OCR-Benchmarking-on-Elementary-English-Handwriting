from __future__ import annotations

from typing import Any, Dict

from .base import ProviderConfig, VisionProvider


class QwenVLProvider(VisionProvider):
    name = "qwen_vl"

    def infer(self, image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
        # TODO: Implement Qwen-VL inference.
        raise NotImplementedError("QwenVLProvider.infer is not implemented yet.")

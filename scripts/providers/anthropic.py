from __future__ import annotations

from typing import Any, Dict

from .base import ProviderConfig, VisionProvider


class AnthropicProvider(VisionProvider):
    name = "claude"

    def infer(self, image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
        # TODO: Implement Claude Vision API call.
        raise NotImplementedError("AnthropicProvider.infer is not implemented yet.")

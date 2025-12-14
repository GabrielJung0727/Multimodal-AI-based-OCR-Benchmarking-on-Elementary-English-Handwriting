from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import ProviderConfig, VisionProvider


class OpenAIProvider(VisionProvider):
    name = "gpt"

    def infer(self, image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
        # TODO: Implement OpenAI Vision call using official SDK.
        # Expected to set raw_response and parsed_text/confidence.
        raise NotImplementedError("OpenAIProvider.infer is not implemented yet.")

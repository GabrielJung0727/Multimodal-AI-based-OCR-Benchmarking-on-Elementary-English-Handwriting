from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ProviderConfig:
    name: str
    env_key: Optional[str] = None
    endpoint: Optional[str] = None
    default_model: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None


class VisionProvider:
    name: str

    def __init__(self, config: ProviderConfig, model: Optional[str] = None) -> None:
        self.config = config
        self.model = model or config.default_model
        self.api_key = os.getenv(config.env_key) if config.env_key else None
        if config.env_key and not self.api_key:
            raise RuntimeError(f"Missing API key for provider '{config.name}'. Set env {config.env_key}.")

    def infer(self, image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Run inference for a single image. Must return:
        {
          "raw_response": ...,
          "parsed_text": "...",
          "parsed_confidence": float|None,
          "cost_estimate": float|None
        }
        """
        raise NotImplementedError

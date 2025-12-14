from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, Type

from providers import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
    ProviderConfig,
    QwenVLProvider,
    VisionProvider,
    LlavaProvider,
)


PROVIDER_CLASSES: Dict[str, Type[VisionProvider]] = {
    "OpenAIProvider": OpenAIProvider,
    "GeminiProvider": GeminiProvider,
    "AnthropicProvider": AnthropicProvider,
    "LlavaProvider": LlavaProvider,
    "QwenVLProvider": QwenVLProvider,
}


def load_yaml(path: Path) -> Dict:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # noqa: PERF203
        raise RuntimeError("PyYAML is required. Install with `pip install pyyaml`.") from exc

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_config(provider_name: str, data: Dict) -> ProviderConfig:
    if provider_name not in data:
        raise ValueError(f"Provider '{provider_name}' not found in config.")
    entry = data[provider_name]
    return ProviderConfig(
        name=provider_name,
        env_key=entry.get("env_key") if entry.get("env_key") not in ("", None) else None,
        endpoint=entry.get("endpoint"),
        default_model=entry.get("default_model"),
        extras=entry.get("extras") or {},
    )


def create_provider(provider_name: str, model_override: str | None, config_path: Path = Path("configs/providers.yaml")) -> VisionProvider:
    config_data = load_yaml(config_path)
    provider_config = build_config(provider_name, config_data)

    class_name = config_data[provider_name].get("class")
    if class_name not in PROVIDER_CLASSES:
        raise ValueError(f"Provider class '{class_name}' is not registered.")

    provider_cls = PROVIDER_CLASSES[class_name]
    return provider_cls(provider_config, model=model_override)

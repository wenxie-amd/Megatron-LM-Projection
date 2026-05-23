"""Load YAML files bundled with the package or from arbitrary paths."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from projection.configs import GPUSpec, ModelConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_packaged_yaml(subpackage: str, filename: str) -> dict[str, Any]:
    """Load a YAML file bundled inside the ``projection`` package."""
    pkg = f"projection.{subpackage}"
    with resources.files(pkg).joinpath(filename).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_packaged_yamls(subpackage: str) -> list[str]:
    pkg = f"projection.{subpackage}"
    return sorted(p.name for p in resources.files(pkg).iterdir() if p.is_file() and p.name.endswith(".yaml"))


def load_model_config(name_or_path: str | Path) -> ModelConfig:
    """Load a model config either from a built-in name (e.g. ``llama3.1_8B``) or a YAML path.

    Built-in name resolution: strip ``.yaml``, lowercase, replace ``.`` and ``-`` with ``_``.
    """
    if isinstance(name_or_path, Path) or "/" in str(name_or_path) or str(name_or_path).endswith(".yaml"):
        data = load_yaml(name_or_path)
    else:
        filename = _builtin_model_filename(name_or_path)
        data = load_packaged_yaml("model_configs", filename)
    return ModelConfig.model_validate(data)


def _builtin_model_filename(name: str) -> str:
    canonical = name.lower().replace(".", "_").replace("-", "_")
    return f"{canonical}.yaml"


def load_gpu_spec(name_or_path: str | Path) -> GPUSpec:
    if isinstance(name_or_path, Path) or "/" in str(name_or_path) or str(name_or_path).endswith(".yaml"):
        data = load_yaml(name_or_path)
    else:
        filename = f"{str(name_or_path).lower()}.yaml"
        data = load_packaged_yaml("gpu_specs", filename)
    return GPUSpec.model_validate(data)


def list_builtin_models() -> list[str]:
    return [p.removesuffix(".yaml") for p in list_packaged_yamls("model_configs")]


def list_builtin_gpus() -> list[str]:
    return [p.removesuffix(".yaml") for p in list_packaged_yamls("gpu_specs")]

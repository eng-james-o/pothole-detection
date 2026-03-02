import threading
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from deployment.config import DEFAULT_MODEL_NAME

class ModelStore:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self._config = self._read_config(config_path)
        self._models: dict[str, YOLO] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _read_config(config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Model config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}

        models = config.get("models", [])
        if not isinstance(models, list) or not models:
            raise ValueError("model_config.yaml must define a non-empty `models` list")

        for item in models:
            if "name" not in item or "path" not in item:
                raise ValueError("Each model entry requires `name` and `path`")

        return config

    def reload_config(self) -> None:
        with self._lock:
            self._config = self._read_config(self.config_path)
            self._models = {}

    @property
    def model_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for item in self._config.get("models", []):
            mapping[item["name"]] = item["path"]
        return mapping

    @property
    def default_model(self) -> str:
        if DEFAULT_MODEL_NAME and DEFAULT_MODEL_NAME in self.model_map:
            return DEFAULT_MODEL_NAME

        configured = self._config.get("default_model")
        if configured and configured in self.model_map:
            return configured

        return next(iter(self.model_map.keys()))

    def available_models(self) -> list[dict[str, Any]]:
        out = []
        for name, path in self.model_map.items():
            full_path = Path(path)
            if not full_path.is_absolute():
                full_path = Path.cwd() / full_path
            out.append(
                {
                    "name": name,
                    "path": str(full_path),
                    "exists": full_path.exists(),
                    "loaded": name in self._models,
                }
            )
        return out

    def get_model(self, model_name: str) -> YOLO:
        if model_name not in self.model_map:
            raise KeyError(f"Unknown model `{model_name}`")

        with self._lock:
            if model_name in self._models:
                return self._models[model_name]

            model_path = Path(self.model_map[model_name])
            if not model_path.is_absolute():
                model_path = Path.cwd() / model_path

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            loaded = YOLO(str(model_path))
            self._models[model_name] = loaded
            return loaded

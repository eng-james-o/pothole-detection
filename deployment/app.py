import os
import time
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml 
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO

APP_NAME = "Pothole Detection API"
APP_VERSION = "1.0.0"

MODEL_CONFIG_PATH = Path(os.getenv("MODEL_CONFIG_PATH", "deployment/model_config.yaml"))
INFERENCE_CONF = float(os.getenv("INFERENCE_CONF", "0.25"))
INFERENCE_IOU = float(os.getenv("INFERENCE_IOU", "0.45"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", "8000000"))
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "").strip()


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
        # Highest precedence: env override for deployment-time control.
        if DEFAULT_MODEL_NAME and DEFAULT_MODEL_NAME in self.model_map:
            return DEFAULT_MODEL_NAME

        # Next: configuration file default.
        configured = self._config.get("default_model")
        if configured and configured in self.model_map:
            return configured

        # Final fallback: first configured model.
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


store = ModelStore(MODEL_CONFIG_PATH)
app = FastAPI(title=APP_NAME, version=APP_VERSION)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image bytes")
    return image


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
        "default_model": store.default_model,
        "models": store.available_models(),
    }


@app.get("/models")
def models() -> dict[str, Any]:
    return {
        "default_model": store.default_model,
        "models": store.available_models(),
    }


@app.post("/reload")
def reload_config() -> dict[str, Any]:
    try:
        store.reload_config()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}") from exc

    return {
        "status": "reloaded",
        "default_model": store.default_model,
        "models": store.available_models(),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    conf: float | None = Form(None),
    iou: float | None = Form(None),
) -> JSONResponse:
    model_name = model or store.default_model
    default_model_used = model is None
    conf_threshold = INFERENCE_CONF if conf is None else float(conf)
    iou_threshold = INFERENCE_IOU if iou is None else float(iou)

    if not (0.0 <= conf_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="`conf` must be in [0.0, 1.0]")
    if not (0.0 <= iou_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="`iou` must be in [0.0, 1.0]")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({len(image_bytes)} bytes). Max allowed: {MAX_IMAGE_BYTES}",
        )

    try:
        image = _decode_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        yolo_model = store.get_model(model_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    t0 = time.perf_counter()
    results = yolo_model.predict(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    first = results[0]
    names = getattr(first, "names", {})

    detections = []
    if first.boxes is not None and len(first.boxes) > 0:
        xyxy = first.boxes.xyxy.detach().cpu().numpy()
        confs = first.boxes.conf.detach().cpu().numpy()
        clss = first.boxes.cls.detach().cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            class_id = int(clss[i])
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                    "confidence": float(confs[i]),
                    "bbox_xyxy": [float(v) for v in xyxy[i].tolist()],
                }
            )

    response = {
        "model": model_name,
        "default_model_used": default_model_used,
        "image": {
            "filename": file.filename,
            "content_type": file.content_type,
            "shape_hwc": list(image.shape),
            "bytes": len(image_bytes),
        },
        "inference": {
            "conf": conf_threshold,
            "iou": iou_threshold,
            "latency_ms": round(latency_ms, 3),
        },
        "summary": {
            "num_detections": len(detections),
        },
        "detections": detections,
    }
    return JSONResponse(content=response)

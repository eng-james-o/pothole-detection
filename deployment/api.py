import time
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from deployment.config import (
    APP_NAME,
    APP_VERSION,
    INFERENCE_CONF,
    INFERENCE_IOU,
    MAX_IMAGE_BYTES,
    MODEL_CONFIG_PATH,
)
from deployment.model_store import ModelStore
from deployment.utils import decode_image, extract_detections

store = ModelStore(MODEL_CONFIG_PATH)

def create_app() -> FastAPI:
    app = FastAPI(title=APP_NAME, version=APP_VERSION)

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
            image = decode_image(image_bytes)
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
        detections = extract_detections(first)

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

    return app

app = create_app()

# Pothole Detection

This project detects potholes in road images. Training and analysis are done in [experiments/pothole-detection-notebook.ipynb](experiments/pothole-detection-notebook.ipynb).

## Production API Deployment

A production-oriented inference API is provided in `deployment/`.

### Deployment files

- `deployment/app.py`: FastAPI app that loads `.pt` models and serves inference.
- `deployment/model_config.yaml`: model registry used by the API.
- `deployment/requirements.txt`: runtime dependencies for serving.
- `deployment/render.yaml`: Render service definition.
- `deployment/models/`: place `.pt` files here (or point paths elsewhere).

## Configure models

Edit `deployment/model_config.yaml` and map API model names to `.pt` files.

```yaml
models:
  - name: yolov8n
    path: deployment/models/yolov8n.pt
  - name: yolov9c
    path: deployment/models/yolov9c.pt
  - name: yolov11
    path: deployment/models/yolov11.pt
  - name: yolov12n
    path: deployment/models/yolov12n.pt
  - name: yolov8_fpn
    path: deployment/models/yolov8-fpn.pt

default_model: yolov9c
```

Notes:

- Paths can be absolute or relative to the repo root.
- API will lazy-load models on first request.
- `default_model` is returned by `GET /models` and used by `POST /predict` when `model` is omitted.
- Optional runtime override: set `DEFAULT_MODEL_NAME` env var to any configured model name.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r deployment/requirements.txt
```

3. Start the API:

```bash
uvicorn deployment.app:app --host 0.0.0.0 --port 8000
```

4. Open docs:

- Swagger UI: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## API endpoints

- `GET /health`: service status and model availability.
- `GET /models`: configured/default model plus file existence.
- `POST /reload`: reload `deployment/model_config.yaml` without restart.
- `POST /predict`: run inference on an uploaded image.

### Predict request example

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/test-image.jpg" \
  -F "model=yolov8n" \
  -F "conf=0.25" \
  -F "iou=0.45"
```

Response contains:

- model used
- input image metadata
- inference settings and latency
- detection list with `class_id`, `class_name`, `confidence`, and `bbox_xyxy`

## Deploy to Render (Free)

### Option A: Blueprint (`render.yaml`)

1. Push this repo to GitHub.
2. In Render, create a **Blueprint** and select the repo.
3. Render reads `deployment/render.yaml` and provisions the web service.

### Option B: Manual Web Service

Use these settings:

- Environment: `Python`
- Build command: `pip install -r deployment/requirements.txt`
- Start command: `uvicorn deployment.app:app --host 0.0.0.0 --port $PORT`

Set environment variables:

- `MODEL_CONFIG_PATH=deployment/model_config.yaml`
- `INFERENCE_CONF=0.25`
- `INFERENCE_IOU=0.45`
- `MAX_IMAGE_BYTES=8000000`
- `DEFAULT_MODEL_NAME=yolov8n`

## Important production notes

- Free tiers are CPU-only; expect lower throughput than GPU benchmarks.
- Keep model files reasonably small for cold-start and memory limits.
- Ensure `.pt` files are available in the deployed filesystem (committed, downloaded at build, or mounted).
- If you change model mappings, call `POST /reload` or redeploy.

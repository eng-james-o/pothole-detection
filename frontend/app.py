import io
import os
from typing import Any

import requests
import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Pothole Detection", page_icon="🚧", layout="wide")

st.title("Pothole Detection API")
st.markdown(f"Connected to: `{API_URL}`")


def load_models() -> dict[str, Any]:
    """Load available models from the API."""
    try:
        response = requests.get(f"{API_URL}/models", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to load models: {e}")
        return {}


def predict_image(
    image_bytes: bytes, filename: str, model: str | None = None, conf: float | None = None, iou: float | None = None
) -> dict[str, Any]:
    """Send image to API for prediction."""
    files = {"file": (filename, image_bytes, "image/jpeg")}
    data = {}
    if model:
        data["model"] = model
    if conf is not None:
        data["conf"] = str(conf)
    if iou is not None:
        data["iou"] = str(iou)

    try:
        response = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Prediction failed: {e}")
        return {}


def main() -> None:
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")

        # Load models
        with st.spinner("Loading models..."):
            models_info = load_models()

        if models_info:
            default_model = models_info.get("default_model", "")
            available_models = [m["name"] for m in models_info.get("models", [])]

            st.subheader("Model Selection")
            selected_model = st.selectbox(
                "Select Model",
                options=available_models,
                index=available_models.index(default_model) if default_model in available_models else 0,
            )

            # Show model status
            for m in models_info.get("models", []):
                status = "✅ Loaded" if m.get("loaded") else "⏳ Not loaded"
                exists = "✅" if m.get("exists") else "❌"
                st.text(f"{m['name']}: {status} {exists}")

        st.subheader("Inference Parameters")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

        st.divider()
        st.subheader("API Status")
        try:
            health = requests.get(f"{API_URL}/health", timeout=5).json()
            st.success(f"✅ API Status: {health.get('status', 'unknown')}")
            st.text(f"Default Model: {health.get('default_model', 'N/A')}")
        except requests.RequestException:
            st.error("❌ API unreachable")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

            if st.button("Run Detection", type="primary"):
                with st.spinner("Running detection..."):
                    image_bytes = uploaded_file.read()
                    result = predict_image(
                        image_bytes,
                        uploaded_file.name,
                        model=selected_model,
                        conf=conf_threshold,
                        iou=iou_threshold,
                    )

                if result:
                    st.session_state["result"] = result
                    st.session_state["image_bytes"] = image_bytes

    with col2:
        st.header("Results")

        if "result" in st.session_state:
            result = st.session_state["result"]

            # Summary metrics
            st.subheader("Detection Summary")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Detections", result["summary"]["num_detections"])
            col_b.metric("Model Used", result["model"])
            col_c.metric("Latency", f"{result['inference']['latency_ms']:.1f}ms")

            # Inference parameters
            st.subheader("Parameters Used")
            st.json(
                {
                    "confidence": result["inference"]["conf"],
                    "iou": result["inference"]["iou"],
                    "default_model_used": result["default_model_used"],
                }
            )

            # Detections
            if result["detections"]:
                st.subheader("Detections")
                for i, det in enumerate(result["detections"], 1):
                    with st.expander(f"Detection {i}"):
                        st.json(det)
            else:
                st.info("No potholes detected")

            # Image info
            st.subheader("Image Info")
            st.json(
                {
                    "filename": result["image"]["filename"],
                    "shape": result["image"]["shape_hwc"],
                    "size_bytes": result["image"]["bytes"],
                }
            )
        else:
            st.info("Upload an image and run detection to see results")


if __name__ == "__main__":
    main()

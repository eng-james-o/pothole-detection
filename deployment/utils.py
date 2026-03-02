import cv2
import numpy as np

def decode_image(image_bytes: bytes) -> np.ndarray:
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image bytes")
    return image

def extract_detections(result) -> list[dict]:
    names = getattr(result, "names", {})
    detections = []

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy().astype(int)

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

    return detections

"""
Image preprocessing + inference.

NOTE: the exact preprocessing (resize, normalization mean/std) must match
whatever transform was used at training time in the resnet18 /
mobilenet_v3_small pipelines. Adjust IMAGENET_MEAN/STD or IMAGE_SIZE below
if the training code used something different (check preprocessing/augment.py
in the training repo).
"""
import base64
import io
import logging

import numpy as np
from PIL import Image

from app.config import settings
from app.model_loader import get_model
from app.schemas import PredictionResult

logger = logging.getLogger("serving_app.inference")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# TODO: replace with your actual class list, ideally loaded from a
# labels.json artifact logged alongside the model in MLflow.
CLASS_NAMES = [f"class_{i}" for i in range(101)]  # placeholder for e.g. Caltech-101


def decode_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img


def decode_base64_image(image_base64: str) -> Image.Image:
    # Strip a data URI prefix if present, e.g. "data:image/png;base64,...."
    if "," in image_base64[:50]:
        image_base64 = image_base64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_base64)
    return decode_image(image_bytes)


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> CHW, add batch dim
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    return arr


def predict(img: Image.Image, top_k: int = 3) -> PredictionResult:
    model = get_model()
    input_array = preprocess(img)

    # mlflow.pyfunc models expose .predict(); the exact output shape
    # depends on how the model was logged. This assumes it returns
    # logits/probabilities of shape (1, num_classes) — adjust if your
    # model wraps this differently (e.g. returns a dict).
    output = model.predict(input_array)
    output = np.asarray(output).reshape(-1)

    # Softmax if the model returns raw logits (skip if already probabilities)
    exp = np.exp(output - np.max(output))
    probs = exp / exp.sum()

    top_indices = probs.argsort()[::-1][:top_k]
    top_k_results = [
        {"class": CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i), "confidence": float(probs[i])}
        for i in top_indices
    ]

    best_idx = int(top_indices[0])
    return PredictionResult(
        predicted_class=CLASS_NAMES[best_idx] if best_idx < len(CLASS_NAMES) else str(best_idx),
        confidence=float(probs[best_idx]),
        top_k=top_k_results,
    )

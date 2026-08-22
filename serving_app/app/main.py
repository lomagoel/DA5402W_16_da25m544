import logging
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.schemas import PredictBase64Request, PredictionResult, HealthResponse
from app.model_loader import load_model, get_model_version
from app.inference import decode_image, decode_base64_image, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serving_app.main")

APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title=settings.APP_TITLE)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def ui_home(request: Request):
    """Simple browser UI for calling /predict — for manual testing/demoing."""
    return templates.TemplateResponse(request, "index.html", {})


@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        # Don't crash the app on startup if MLflow/registry isn't reachable yet —
        # log it loudly so it's visible, and let /health report "model_not_loaded".
        logger.error(f"Failed to load model at startup: {e}")


# --------------------------------------------------------------------------
# Basic request latency logging — a starting point for the monitoring task
# (prediction logging / API latency requirement). Extend with Prometheus
# client + /metrics endpoint if you want scrape-based monitoring instead.
# --------------------------------------------------------------------------
@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} completed_in={duration_ms:.1f}ms status={response.status_code}")
    return response


@app.get("/health", response_model=HealthResponse)
async def health():
    version = get_model_version()
    return HealthResponse(
        status="ok" if version is not None else "model_not_loaded",
        model_name=settings.MLFLOW_MODEL_NAME,
        model_stage=settings.MLFLOW_MODEL_STAGE,
        model_version=version,
    )


@app.post("/reload")
async def reload_model():
    """
    Force-reload the model from the registry — call this after promoting
    a new model version to the configured stage, so the API picks it up
    without a redeploy.
    """
    try:
        load_model(force=True)
        return {"status": "reloaded", "model_version": get_model_version()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


@app.post("/predict", response_model=PredictionResult)
async def predict_upload(file: UploadFile = File(...)):
    """Predict from a multipart file upload."""
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail=f"File too large ({size_mb:.1f} MB)")

    try:
        img = decode_image(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file")

    try:
        return predict(img)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/base64", response_model=PredictionResult)
async def predict_base64(body: PredictBase64Request):
    """Predict from a base64-encoded image in a JSON body."""
    try:
        img = decode_base64_image(body.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode base64 image")

    try:
        return predict(img)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

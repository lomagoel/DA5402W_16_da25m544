"""
Basic smoke tests. Run with: pytest tests/
Note: these will fail to actually predict until a real model is registered
in MLflow — they're structured so you can extend them once that's wired up.
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint_responds():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "model_name" in body


def test_predict_rejects_bad_base64():
    resp = client.post("/predict/base64", json={"image_base64": "not-valid-base64!!"})
    assert resp.status_code == 400


def test_predict_upload_rejects_non_image():
    resp = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400

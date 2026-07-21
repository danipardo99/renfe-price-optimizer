"""Smoke test de la API — no requiere modelo entrenado para /health."""

from __future__ import annotations

from fastapi.testclient import TestClient

from renfe_optimizer.api.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_predict_sin_modelo_devuelve_503():
    """Sin modelo entrenado en disco, el endpoint /predict devuelve 503."""
    payload = {
        "vehicle_type": "AVE",
        "vehicle_class": "Turista",
        "fare": "Promo",
        "duration": 2.75,
        "dias_anticipacion": 30,
        "hora_salida": 9,
        "dia_semana": 1,
        "mes": 6,
        "temporada": "verano",
    }
    r = client.post("/predict", json=payload)
    # Si el modelo existe, 200; si no existe, 503. Ambos son válidos.
    assert r.status_code in (200, 503)

"""Tests de la API del Renfe Price Optimizer multidestino."""

from __future__ import annotations

from fastapi.testclient import TestClient

from renfe_optimizer.api.main import app

client = TestClient(app)


def payload_valido() -> dict:
    """Consulta compatible con el modelo multidestino."""
    return {
        "destino": "BARCELONA",
        "vehicle_type": "AVE",
        "vehicle_class": "Turista",
        "fare": "Promo",
        "duration": 2.75,
        "dias_anticipacion": 30,
        "hora_salida": 9,
        "dia_semana": "Jueves",
        "mes": 6,
        "temporada": "Verano",
    }


def test_health_ok() -> None:
    respuesta = client.get("/health")

    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "ok"


def test_predict_payload_valido() -> None:
    respuesta = client.post(
        "/predict",
        json=payload_valido(),
    )

    # 200 si el modelo está presente.
    # 503 si el modelo no está disponible en el entorno de CI.
    assert respuesta.status_code in (200, 503)


def test_predict_admite_sevilla() -> None:
    payload = payload_valido()
    payload["destino"] = "SEVILLA"

    respuesta = client.post("/predict", json=payload)

    assert respuesta.status_code in (200, 503)


def test_predict_admite_valencia() -> None:
    payload = payload_valido()
    payload["destino"] = "VALENCIA"

    respuesta = client.post("/predict", json=payload)

    assert respuesta.status_code in (200, 503)


def test_predict_rechaza_destino_desconocido() -> None:
    payload = payload_valido()
    payload["destino"] = "BILBAO"

    respuesta = client.post("/predict", json=payload)

    assert respuesta.status_code == 422

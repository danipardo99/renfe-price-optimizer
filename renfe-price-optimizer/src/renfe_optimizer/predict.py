"""Servicio de predicción y recomendación de compra.

Se separa del entrenamiento para que la API cargue el modelo UNA sola vez
al arrancar y no en cada request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from renfe_optimizer.config import MODEL_PATH
from renfe_optimizer.features import normaliza_recomendacion


@lru_cache(maxsize=1)
def load_model(model_path: str = MODEL_PATH):
    """Carga el modelo una única vez (cacheada)."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No se encuentra el modelo en {model_path}. "
            "Ejecuta antes: python -m renfe_optimizer.train"
        )
    return joblib.load(model_path)


def predict_price(features: dict) -> float:
    """Predice el precio de un billete dado un dict de features."""
    model = load_model()
    df = pd.DataFrame([features])
    return float(model.predict(df)[0])


def recommend(features: dict, ventanas: list[int] | None = None) -> dict:
    """Devuelve la recomendación completa (predicción + acción).

    Para cada ventana de antelación candidata, estima el precio; el mínimo
    marca el momento óptimo de compra. Si el óptimo cae en la antelación
    actual, se recomienda comprar hoy.
    """
    ventanas = ventanas or [90, 60, 30, 21, 14, 7, 3, 1]
    model = load_model()

    dias_hoy = int(features["dias_anticipacion"])
    # Simulamos la curva precio-antelación para el mismo billete
    filas = []
    for d in ventanas:
        f = dict(features)
        f["dias_anticipacion"] = d
        filas.append(f)
    df = pd.DataFrame(filas)
    precios = model.predict(df)

    idx_min = int(np.argmin(precios))
    dias_optimo = int(ventanas[idx_min])
    precio_optimo = float(precios[idx_min])
    precio_hoy = float(model.predict(pd.DataFrame([features]))[0])

    dias_hasta_optimo = dias_hoy - dias_optimo   # >0 → esperar; ≤0 → comprar
    return {
        "precio_estimado_hoy": round(precio_hoy, 2),
        "precio_minimo_esperado": round(precio_optimo, 2),
        "antelacion_optima_dias": dias_optimo,
        "ahorro_estimado_eur": round(precio_hoy - precio_optimo, 2),
        "ahorro_estimado_pct": round((precio_hoy - precio_optimo) / precio_hoy * 100, 2),
        "recomendacion": normaliza_recomendacion(dias_hasta_optimo),
        "curva": [
            {"dias_anticipacion": int(d), "precio_estimado": round(float(p), 2)}
            for d, p in zip(ventanas, precios)
        ],
    }

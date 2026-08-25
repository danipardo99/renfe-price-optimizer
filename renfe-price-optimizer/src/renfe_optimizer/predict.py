"""Servicio de predicción y recomendación de compra.

Se separa del entrenamiento para que la API cargue el modelo UNA sola vez
al arrancar y no en cada request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
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


def recommend(
    features: dict,
    ventanas: list[int] | None = None,
    margen_pct: float = 1.0,
) -> dict:
    """Devuelve la recomendación completa (predicción + acción).

    Estrategia de optimal stopping: partiendo de la antelación actual, se
    estima el precio en cada momento futuro alcanzable (antelaciones MENORES
    que la actual, porque esperar solo reduce la antelación) y se compara con
    el precio de hoy.

    Solo se recomienda ESPERAR si existe un momento futuro más barato que hoy
    por encima de `margen_pct` (para no recomendar esperas con ahorro
    despreciable o negativo). En caso contrario, se recomienda COMPRAR HOY;
    y si además el precio solo sube, se avisa de ello.
    """
    ventanas = ventanas or [90, 60, 30, 21, 14, 7, 3, 1]
    model = load_model()

    dias_hoy = int(features["dias_anticipacion"])

    # Evaluamos la curva en las ventanas candidatas + SIEMPRE el momento actual,
    # para que la comparación sea coherente (mismo billete, distinta antelación).
    puntos = sorted(set(ventanas) | {dias_hoy}, reverse=True)
    filas = []
    for d in puntos:
        f = dict(features)
        f["dias_anticipacion"] = d
        filas.append(f)
    precios = model.predict(pd.DataFrame(filas))
    precio_por_dia = {int(d): float(p) for d, p in zip(puntos, precios)}

    # Precio de comprar HOY (en la antelación actual)
    precio_hoy = precio_por_dia[dias_hoy]

    # Momentos alcanzables esperando: antelaciones ESTRICTAMENTE menores que hoy
    futuros = {d: p for d, p in precio_por_dia.items() if d < dias_hoy}

    if futuros:
        dias_fut_opt = min(futuros, key=futuros.get)
        precio_fut_opt = futuros[dias_fut_opt]
    else:
        # Ya estamos en el día del viaje: no hay margen para esperar
        dias_fut_opt, precio_fut_opt = dias_hoy, precio_hoy

    # ¿El precio solo sube desde hoy? (ningún futuro alcanzable es más barato)
    precio_sube = all(p >= precio_hoy for p in futuros.values()) if futuros else True

    # Solo esperamos si el mejor futuro es más barato que hoy por encima del margen
    umbral = precio_hoy * (1 - margen_pct / 100.0)
    if precio_fut_opt < umbral:
        dias_optimo, precio_optimo = dias_fut_opt, precio_fut_opt
        recomendacion = normaliza_recomendacion(dias_hoy - dias_optimo)
    else:
        dias_optimo, precio_optimo = dias_hoy, precio_hoy
        # Compramos hoy: si además el precio solo sube, avisamos
        recomendacion = "COMPRA HOY: el precio va a subir" if precio_sube else "COMPRA HOY"

    ahorro_eur = precio_hoy - precio_optimo     # nunca negativo con esta lógica

    return {
        "precio_estimado_hoy": round(precio_hoy, 2),
        "precio_minimo_esperado": round(precio_optimo, 2),
        "antelacion_optima_dias": int(dias_optimo),
        "ahorro_estimado_eur": round(ahorro_eur, 2),
        "ahorro_estimado_pct": round(ahorro_eur / precio_hoy * 100, 2) if precio_hoy else 0.0,
        "recomendacion": recomendacion,
        "curva": [
            {"dias_anticipacion": int(d), "precio_estimado": round(precio_por_dia[d], 2)}
            for d in sorted(precio_por_dia, reverse=True)
        ],
    }

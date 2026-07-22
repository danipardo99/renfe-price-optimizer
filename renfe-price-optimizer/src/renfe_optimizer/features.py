"""Ingeniería de variables para el dataset Renfe.

Reglas del bloque 8: la lógica reutilizable NO vive en notebooks. Vive aquí,
en funciones testables. El notebook explora; features.py produce el dataset
que consumirá train.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from renfe_optimizer.config import ROOT_DIR, load_params


# ---------------------------------------------------------------------------
# Funciones puras (fácilmente testables)
# ---------------------------------------------------------------------------

def calcular_dias_anticipacion(fecha_viaje: pd.Series, fecha_captura: pd.Series) -> pd.Series:
    """Días entre la captura del precio y la fecha del viaje.

    Es la variable clave del TFM: modeliza la curva precio-antelación.
    """
    delta = (pd.to_datetime(fecha_viaje) - pd.to_datetime(fecha_captura)).dt.total_seconds()
    return (delta / 86400).clip(lower=0).round().astype("Int64")


def clasifica_temporada(mes: int) -> str:
    """Devuelve una etiqueta de temporada a partir del mes del viaje."""
    if mes in (6, 7, 8):
        return "verano"
    if mes in (12, 1, 2):
        return "invierno"
    if mes in (3, 4, 5):
        return "primavera"
    return "otoño"


def normaliza_recomendacion(dias_hasta_optimo: int) -> str:
    """Transforma el output del modelo en una recomendación accionable.

    Regla académica utilizada para el problema de optimal stopping:
      - Si el precio mínimo esperado se sitúa hoy → COMPRA HOY
      - Si aún queda margen → ESPERA {n} días
      - Si el precio está subiendo → ALERTA
    """
    if dias_hasta_optimo <= 0:
        return "COMPRA HOY"
    if dias_hasta_optimo > 30:
        return "ALERTA: el precio va a subir"
    return f"ESPERA {dias_hasta_optimo} días"


# ---------------------------------------------------------------------------
# Pipeline de features (usa las funciones anteriores)
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enriquecimiento del dataframe Renfe:

    - filtra a Madrid-Barcelona
    - descarta filas con precio nulo (venta cerrada / sin oferta)
    - deriva días de anticipación, hora, día de semana, mes, temporada
    """
    df = df.copy()

    # Filtrado del corredor de interés
    df = df[(df["origin"] == "MADRID") & (df["destination"] == "BARCELONA")]

    # Precio nulo → no hay oferta, se descarta
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 5]      # descarta billetes de 0€ o sospechosamente baratos
    df = df[df["price"] < 500]    # descarta outliers absurdos

    # Fechas
    df["departure"] = pd.to_datetime(df["departure"], format="mixed", errors="coerce")
    df["insert_date"] = pd.to_datetime(df["insert_date"], format="mixed", errors="coerce")

    df["dias_anticipacion"] = calcular_dias_anticipacion(df["departure"], df["insert_date"])
    df["hora_salida"] = df["departure"].dt.hour
    df["dia_semana"] = df["departure"].dt.dayofweek       # 0 lunes … 6 domingo
    df["mes"] = df["departure"].dt.month
    df["temporada"] = df["mes"].apply(clasifica_temporada)

    # Duración: viene en horas decimales, la dejamos numérica
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df = df.dropna(subset=["duration", "dias_anticipacion"])

    # Selección final de columnas del modelo
    cols = [
        "vehicle_type", "vehicle_class", "fare",
        "duration", "dias_anticipacion", "hora_salida", "dia_semana", "mes", "temporada",
        "price",
    ]
    return df[cols].reset_index(drop=True)


def load_and_build(raw_path: str | Path) -> pd.DataFrame:
    """Lee el CSV en crudo, aplica build_features y devuelve el dataset listo."""
    df_raw = pd.read_csv(raw_path)
    return build_features(df_raw)


def main() -> None:
    """Entry point invocado desde `dvc.yaml` o CLI."""
    params = load_params()
    raw_path = ROOT_DIR / params["data"]["raw_path"]
    processed_path = ROOT_DIR / params["data"]["processed_path"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_build(raw_path)
    df.to_parquet(processed_path, index=False)
    print(f"[features] Guardado dataset procesado: {processed_path} ({len(df)} filas)")


if __name__ == "__main__":
    main()

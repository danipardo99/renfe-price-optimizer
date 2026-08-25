"""Ingeniería de variables para el dataset Renfe (multidestino desde Madrid).

El dataset de entrada ya viene limpio y con features calculadas (proceso de EDA
del equipo). Este módulo mapea las columnas al formato que espera el modelo y
selecciona las columnas finales. Cubre los destinos Barcelona, Sevilla y Valencia.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from renfe_optimizer.config import ROOT_DIR, load_params

# ---------------------------------------------------------------------------
# Funciones puras (se conservan para tests y para predict.py)
# ---------------------------------------------------------------------------

def calcular_dias_anticipacion(fecha_viaje: pd.Series, fecha_captura: pd.Series) -> pd.Series:
    """Días entre la captura del precio y la fecha del viaje."""
    delta = (pd.to_datetime(fecha_viaje) - pd.to_datetime(fecha_captura)).dt.total_seconds()
    return (delta / 86400).clip(lower=0).round().astype("Int64")


def clasifica_temporada(mes: int) -> str:
    """Devuelve una etiqueta de temporada a partir del mes del viaje."""
    if mes in (6, 7, 8):
        return "Verano"
    if mes in (12, 1, 2):
        return "Invierno"
    if mes in (3, 4, 5):
        return "Primavera"
    return "Otoño"


def normaliza_recomendacion(dias_hasta_optimo: int) -> str:
    """Transforma el output del modelo en una recomendación accionable."""
    if dias_hasta_optimo <= 0:
        return "COMPRA HOY"
    return f"ESPERA {dias_hasta_optimo} días"


# ---------------------------------------------------------------------------
# Mapeo de columnas del CSV limpio → nombres que usa el modelo
# ---------------------------------------------------------------------------

MAPEO_COLUMNAS = {
    "ciudad_destino": "destino",
    "tipo_tren": "vehicle_type",
    "clase": "vehicle_class",
    "tarifa": "fare",
    "duracion": "duration",
    "mes_salida": "mes",
    "precio": "price",
    # dias_anticipacion, hora_salida, dia_semana, temporada se conservan igual
}


# ---------------------------------------------------------------------------
# Pipeline de features
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adapta el dataset limpio del equipo al formato que consume el modelo.

    El CSV de entrada ya trae features calculadas. Aquí mapeamos nombres,
    incorporamos el destino como variable y descartamos precios no válidos.
    Todos los trayectos parten de Madrid hacia Barcelona, Sevilla o Valencia.
    """
    df = df.copy()

    # Renombrar columnas al formato del modelo
    df = df.rename(columns=MAPEO_COLUMNAS)

    # Descartar precios no válidos
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 5]
    df = df[df["price"] < 500]

    # Asegurar tipos numéricos
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["dias_anticipacion"] = pd.to_numeric(df["dias_anticipacion"], errors="coerce")
    df["hora_salida"] = pd.to_numeric(df["hora_salida"], errors="coerce")
    df["mes"] = pd.to_numeric(df["mes"], errors="coerce")
    df = df.dropna(subset=["duration", "dias_anticipacion"])

    # dia_semana y temporada vienen como texto → se mantienen como categóricas

    # Selección final de columnas del modelo (incluye destino)
    cols = [
        "destino",
        "vehicle_type", "vehicle_class", "fare",
        "duration", "dias_anticipacion", "hora_salida", "dia_semana", "mes", "temporada",
        "price",
    ]
    return df[cols].reset_index(drop=True)


def load_and_build(raw_path: str | Path) -> pd.DataFrame:
    """Lee el CSV limpio, aplica build_features y devuelve el dataset listo."""
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

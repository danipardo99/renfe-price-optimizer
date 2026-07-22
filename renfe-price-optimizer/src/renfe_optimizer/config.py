"""Configuración centralizada: rutas y variables de entorno.

Todo lo configurable vive aquí o en params.yaml. El código NUNCA hardcodea
rutas: así funciona igual en local, en Docker y en GitHub Actions.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT_DIR / "params.yaml"


def load_params(path: str | Path = PARAMS_PATH) -> dict:
    """Carga el fichero params.yaml y devuelve un dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Variables de entorno con valores por defecto sensatos
MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT_DIR / "models" / "renfe_model.pkl"))
DATA_PATH = os.getenv("DATA_PATH", str(ROOT_DIR / "data" / "processed" / "train.parquet"))
_mlflow_default = (ROOT_DIR / "mlruns").as_uri()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", _mlflow_default)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "renfe-price-optimizer")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

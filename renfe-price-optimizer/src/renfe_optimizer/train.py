"""Entrenamiento del modelo con tracking en MLflow.

Ejecutable:
    python -m renfe_optimizer.train --config params.yaml
    python -m renfe_optimizer.train --sample --max-rows 500   # modo CI

Registra: parámetros, métricas (MAE, RMSE, MAPE, R²), el propio modelo y
un artefacto con la métrica principal en JSON.
"""

from __future__ import annotations

import argparse
import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from renfe_optimizer.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    ROOT_DIR,
    load_params,
)
from renfe_optimizer.features import load_and_build

# ---------------------------------------------------------------------------
# Factory de modelos
# ---------------------------------------------------------------------------

def make_estimator(model_cfg: dict):
    """Devuelve el estimador segun `model.type` en params.yaml."""
    mtype = model_cfg["type"]
    params = model_cfg.get("params", {})

    if mtype == "baseline":
        return LinearRegression()
    if mtype == "random_forest":
        return RandomForestRegressor(**params)
    if mtype == "xgboost":
        # Import perezoso para que un entorno sin xgboost pueda entrenar baseline
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:squarederror", **params)
    raise ValueError(f"Modelo desconocido: {mtype}")


def build_pipeline(model_cfg: dict, feat_cfg: dict) -> Pipeline:
    """Preprocesado + estimador en un único Pipeline sklearn."""
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feat_cfg["numerical"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                feat_cfg["categorical"],
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("model", make_estimator(model_cfg))])


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Métricas del TFM: MAE, RMSE, MAPE (%), R²."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train(config_path: str = "params.yaml", sample: bool = False, max_rows: int = 500) -> dict:
    params = load_params(ROOT_DIR / config_path)

    # ----- Datos ---------------------------------------------------------
    raw_path = ROOT_DIR / params["data"]["raw_path"]
    df = load_and_build(raw_path)
    if sample:
        df = df.head(max_rows)
        print(f"[train] Modo sample activo: usando {len(df)} filas")

    target = params["training"]["target"]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["training"]["test_size"],
        random_state=params["training"]["random_state"],
    )

    # ----- MLflow --------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{params['model']['type']}-baseline"):
        pipe = build_pipeline(params["model"], params["features"])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = evaluate(y_test.to_numpy(), y_pred)

        # Registro
        mlflow.log_params(params["model"]["params"])
        mlflow.log_param("model_type", params["model"]["type"])
        mlflow.log_metrics(metrics)

        # Guardado local + artefacto MLflow
        model_out = ROOT_DIR / params["paths"]["model_out"]
        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_out)
        mlflow.log_artifact(str(model_out), artifact_path="model")

        metrics_out = ROOT_DIR / params["paths"]["metrics_out"]
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(metrics, indent=2))

        print(f"[train] Modelo guardado: {model_out}")
        print(f"[train] Métricas: {metrics}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el modelo Renfe con tracking MLflow")
    parser.add_argument("--config", default="params.yaml", help="Ruta a params.yaml")
    parser.add_argument("--sample", action="store_true", help="Usar una muestra para CI/CD")
    parser.add_argument("--max-rows", type=int, default=500, help="Filas del sample")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(config_path=args.config, sample=args.sample, max_rows=args.max_rows)

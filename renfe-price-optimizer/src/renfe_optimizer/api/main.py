"""Aplicación FastAPI del renfe-price-optimizer."""

from __future__ import annotations

from fastapi import FastAPI

from renfe_optimizer import __version__
from renfe_optimizer.api.routers import predict as predict_router

app = FastAPI(
    title="Renfe Price Optimizer API",
    description=(
        "Predice el precio del billete Madrid-Barcelona y recomienda el "
        "momento óptimo de compra. TFM MLOps 2026."
    ),
    version=__version__,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


app.include_router(predict_router.router, tags=["predict"])

"""Router del endpoint /predict."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from renfe_optimizer.api.schemas import ConsultaIn, RecomendacionOut
from renfe_optimizer.predict import recommend

router = APIRouter()


@router.post("/predict", response_model=RecomendacionOut)
def predict(consulta: ConsultaIn) -> RecomendacionOut:
    try:
        resultado = recommend(consulta.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RecomendacionOut(**resultado)

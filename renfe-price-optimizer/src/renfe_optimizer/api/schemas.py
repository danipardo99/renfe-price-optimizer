"""Modelos Pydantic de entrada y salida de la API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ConsultaIn(BaseModel):
    """Payload de la consulta del usuario. Trayectos desde Madrid."""

    destino: Literal["BARCELONA", "SEVILLA", "VALENCIA"] = "BARCELONA"
    vehicle_type: Literal[
        "AVE", "R. EXPRES", "AV City", "INTERCITY", "ALVIA", "MD", "REGIONAL"
    ] = "AVE"
    vehicle_class: Literal["Turista", "Turista Plus", "Preferente"] = "Turista"
    fare: Literal["Promo", "Flexible", "Adulto ida"] = "Promo"

    duration: float = Field(2.75, ge=0.5, le=12.0, description="Duración en horas")
    dias_anticipacion: int = Field(30, ge=0, le=180)
    hora_salida: int = Field(9, ge=0, le=23)
    dia_semana: Literal[
        "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"
    ] = "Jueves"
    mes: int = Field(6, ge=1, le=12)
    temporada: Literal["Primavera", "Verano", "Otoño", "Invierno"] = "Verano"


class PuntoCurva(BaseModel):
    dias_anticipacion: int
    precio_estimado: float


class RecomendacionOut(BaseModel):
    precio_estimado_hoy: float
    precio_minimo_esperado: float
    antelacion_optima_dias: int
    ahorro_estimado_eur: float
    ahorro_estimado_pct: float
    recomendacion: str
    curva: list[PuntoCurva]

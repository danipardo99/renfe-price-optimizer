"""Tests unitarios de la lógica de features."""

from __future__ import annotations

import pandas as pd

from renfe_optimizer.features import (
    build_features,
    calcular_dias_anticipacion,
    clasifica_temporada,
    normaliza_recomendacion,
)


def test_dias_anticipacion_basico():
    fecha_viaje = pd.Series(pd.to_datetime(["2019-05-01"]))
    fecha_captura = pd.Series(pd.to_datetime(["2019-04-11"]))
    resultado = calcular_dias_anticipacion(fecha_viaje, fecha_captura)
    assert int(resultado.iloc[0]) == 20


def test_dias_anticipacion_no_negativa():
    """Si la captura es posterior al viaje, debe devolverse 0."""
    fecha_viaje = pd.Series(pd.to_datetime(["2019-04-11"]))
    fecha_captura = pd.Series(pd.to_datetime(["2019-05-01"]))
    resultado = calcular_dias_anticipacion(fecha_viaje, fecha_captura)
    assert int(resultado.iloc[0]) == 0


def test_temporada():
    assert clasifica_temporada(7) == "verano"
    assert clasifica_temporada(1) == "invierno"
    assert clasifica_temporada(4) == "primavera"
    assert clasifica_temporada(10) == "otoño"


def test_recomendacion_compra_hoy():
    assert normaliza_recomendacion(0) == "COMPRA HOY"
    assert normaliza_recomendacion(-5) == "COMPRA HOY"


def test_recomendacion_espera():
    assert normaliza_recomendacion(10) == "ESPERA 10 días"


def test_recomendacion_alerta():
    assert normaliza_recomendacion(45).startswith("ALERTA")


def test_build_features_filtra_madrid_barcelona():
    df = pd.DataFrame({
        "origin": ["MADRID", "SEVILLA"],
        "destination": ["BARCELONA", "MADRID"],
        "departure": ["2019-06-01 09:00:00", "2019-06-01 09:00:00"],
        "insert_date": ["2019-05-01 10:00:00", "2019-05-01 10:00:00"],
        "duration": [2.75, 2.5],
        "price": [65.0, 40.0],
        "vehicle_type": ["AVE", "AVE"],
        "vehicle_class": ["Turista", "Turista"],
        "fare": ["Promo", "Promo"],
    })
    out = build_features(df)
    assert len(out) == 1
    assert set(["dias_anticipacion", "temporada", "price"]).issubset(out.columns)

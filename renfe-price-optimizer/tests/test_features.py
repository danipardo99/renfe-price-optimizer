"""Tests unitarios de la lógica de features multidestino."""

from __future__ import annotations

import pandas as pd

from renfe_optimizer.features import (
    build_features,
    calcular_dias_anticipacion,
    clasifica_temporada,
    normaliza_recomendacion,
)


def crear_dataframe_limpio() -> pd.DataFrame:
    """Crea una muestra con el mismo esquema que el CSV limpio oficial."""
    return pd.DataFrame(
        {
            "ciudad_origen": [
                "MÁLAGA",
                "MADRID",
                "SEVILLA",
            ],
            "ciudad_destino": [
                "BARCELONA",
                "SEVILLA",
                "VALENCIA",
            ],
            "fecha_salida": [
                "2019-06-01 09:00:00",
                "2019-07-05 10:30:00",
                "2019-05-20 08:15:00",
            ],
            "fecha_llegada": [
                "2019-06-01 11:45:00",
                "2019-07-05 13:10:00",
                "2019-05-20 10:00:00",
            ],
            "duracion": [2.75, 2.67, 1.75],
            "tipo_tren": ["AVE", "AVE", "INTERCITY"],
            "clase": ["Turista", "Preferente", "Turista Plus"],
            "precio": [65.0, 82.5, 49.9],
            "tarifa": ["Promo", "Flexible", "Adulto ida"],
            "fecha_captura": [
                "2019-05-01 10:00:00",
                "2019-06-05 10:00:00",
                "2019-05-01 10:00:00",
            ],
            "mes_salida": [6, 7, 5],
            "dia_semana": ["Sábado", "Viernes", "Lunes"],
            "hora_salida": [9, 10, 8],
            "hora_captura": [10, 10, 10],
            "dias_anticipacion": [31, 30, 19],
            "temporada": ["Verano", "Verano", "Primavera"],
            "duracion_minutos": [165, 160, 105],
        }
    )


def test_dias_anticipacion_basico() -> None:
    fecha_viaje = pd.Series(pd.to_datetime(["2019-05-01"]))
    fecha_captura = pd.Series(pd.to_datetime(["2019-04-11"]))

    resultado = calcular_dias_anticipacion(
        fecha_viaje,
        fecha_captura,
    )

    assert int(resultado.iloc[0]) == 20


def test_dias_anticipacion_no_negativa() -> None:
    fecha_viaje = pd.Series(pd.to_datetime(["2019-04-11"]))
    fecha_captura = pd.Series(pd.to_datetime(["2019-05-01"]))

    resultado = calcular_dias_anticipacion(
        fecha_viaje,
        fecha_captura,
    )

    assert int(resultado.iloc[0]) == 0


def test_temporada() -> None:
    assert clasifica_temporada(7) == "Verano"
    assert clasifica_temporada(1) == "Invierno"
    assert clasifica_temporada(4) == "Primavera"
    assert clasifica_temporada(10) == "Otoño"


def test_recomendacion_compra_hoy() -> None:
    assert normaliza_recomendacion(0) == "COMPRA HOY"
    assert normaliza_recomendacion(-5) == "COMPRA HOY"


def test_recomendacion_espera() -> None:
    assert normaliza_recomendacion(10) == "ESPERA 10 días"


def test_build_features_admite_tres_destinos() -> None:
    resultado = build_features(crear_dataframe_limpio())

    assert len(resultado) == 3
    assert set(resultado["destino"]) == {
        "BARCELONA",
        "SEVILLA",
        "VALENCIA",
    }


def test_build_features_genera_esquema_del_modelo() -> None:
    resultado = build_features(crear_dataframe_limpio())

    columnas_esperadas = [
        "origen",
        "destino",
        "vehicle_type",
        "vehicle_class",
        "fare",
        "duration",
        "dias_anticipacion",
        "hora_salida",
        "dia_semana",
        "mes",
        "temporada",
        "price",
    ]

    assert list(resultado.columns) == columnas_esperadas


def test_build_features_mapea_columnas() -> None:
    resultado = build_features(crear_dataframe_limpio())

    assert resultado.loc[0, "vehicle_type"] == "AVE"
    assert resultado.loc[0, "vehicle_class"] == "Turista"
    assert resultado.loc[0, "fare"] == "Promo"
    assert resultado.loc[0, "price"] == 65.0
    assert resultado.loc[0, "mes"] == 6


def test_build_features_elimina_precios_invalidos() -> None:
    dataframe = crear_dataframe_limpio()

    dataframe.loc[0, "precio"] = 0
    dataframe.loc[1, "precio"] = 600

    resultado = build_features(dataframe)

    assert len(resultado) == 1
    assert resultado.iloc[0]["destino"] == "VALENCIA"


def test_build_features_incluye_origen() -> None:
    resultado = build_features(crear_dataframe_limpio())

    assert "origen" in resultado.columns
    assert set(resultado["origen"]) == {"MÁLAGA", "MADRID", "SEVILLA"}

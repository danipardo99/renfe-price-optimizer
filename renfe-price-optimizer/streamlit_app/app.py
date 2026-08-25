"""Interfaz Streamlit del Renfe Price Optimizer multidestino."""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

DESTINOS = ["BARCELONA", "SEVILLA", "VALENCIA"]

TIPOS_TREN = [
    "AVE",
    "R. EXPRES",
    "AV City",
    "INTERCITY",
    "ALVIA",
    "MD",
    "REGIONAL",
]

CLASES = ["Turista", "Turista Plus", "Preferente"]

TARIFAS = ["Promo", "Flexible", "Adulto ida"]

DIAS_SEMANA = [
    "Lunes",
    "Martes",
    "Miércoles",
    "Jueves",
    "Viernes",
    "Sábado",
    "Domingo",
]


def obtener_temporada(mes: int) -> str:
    """Obtiene la temporada con el formato usado al entrenar el modelo."""
    if mes in (6, 7, 8):
        return "Verano"
    if mes in (12, 1, 2):
        return "Invierno"
    if mes in (3, 4, 5):
        return "Primavera"
    return "Otoño"


st.set_page_config(
    page_title="Renfe Price Optimizer",
    page_icon="🚄",
    layout="wide",
)

st.title("🚄 Renfe Price Optimizer")

st.caption(
    "Predicción de precios y recomendación del momento óptimo de compra "
    "para trayectos desde Madrid."
)

st.info(
    "Modelo XGBoost entrenado con más de 13 millones de registros. "
    "Destinos disponibles: Barcelona, Sevilla y Valencia."
)

with st.sidebar:
    st.header("Estado del servicio")

    try:
        respuesta_health = requests.get(f"{API_URL}/health", timeout=3)
        respuesta_health.raise_for_status()
        health = respuesta_health.json()

        st.success(
            f"API disponible · versión {health.get('version', 'desconocida')}"
        )
    except requests.RequestException:
        st.error(
            "La API no está disponible. Arranca Uvicorn en otra terminal."
        )

st.subheader("Datos del trayecto")

columna_1, columna_2 = st.columns(2)

with columna_1:
    destino = st.selectbox(
        "Destino",
        options=DESTINOS,
        index=0,
    )

    vehicle_type = st.selectbox(
        "Tipo de tren",
        options=TIPOS_TREN,
        index=0,
    )

    vehicle_class = st.selectbox(
        "Clase",
        options=CLASES,
        index=0,
    )

    fare = st.selectbox(
        "Tarifa",
        options=TARIFAS,
        index=0,
    )

    duration = st.number_input(
        "Duración estimada del trayecto en horas",
        min_value=0.5,
        max_value=12.0,
        value=2.75,
        step=0.05,
    )

with columna_2:
    dias_anticipacion = st.slider(
        "Días de antelación",
        min_value=0,
        max_value=180,
        value=30,
    )

    hora_salida = st.slider(
        "Hora de salida",
        min_value=0,
        max_value=23,
        value=9,
    )

    dia_semana = st.selectbox(
        "Día de la semana",
        options=DIAS_SEMANA,
        index=3,
    )

    mes = st.slider(
        "Mes del viaje",
        min_value=1,
        max_value=12,
        value=6,
    )

temporada = obtener_temporada(mes)

st.caption(f"Temporada calculada automáticamente: **{temporada}**")

st.divider()

if st.button(
    "🔍 ¿Compro o espero?",
    type="primary",
    use_container_width=True,
):
    payload = {
        "destino": destino,
        "vehicle_type": vehicle_type,
        "vehicle_class": vehicle_class,
        "fare": fare,
        "duration": float(duration),
        "dias_anticipacion": int(dias_anticipacion),
        "hora_salida": int(hora_salida),
        "dia_semana": dia_semana,
        "mes": int(mes),
        "temporada": temporada,
    }

    try:
        with st.spinner("Calculando la recomendación..."):
            respuesta = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

        respuesta.raise_for_status()
        resultado = respuesta.json()

    except requests.ConnectionError:
        st.error(
            "No se ha podido conectar con la API. "
            "Comprueba que Uvicorn está ejecutándose en el puerto 8000."
        )

    except requests.Timeout:
        st.error("La API ha tardado demasiado en responder.")

    except requests.HTTPError:
        st.error(
            f"La API devolvió un error {respuesta.status_code}: "
            f"{respuesta.text}"
        )

    except requests.RequestException as exc:
        st.error(f"Error al consultar la API: {exc}")

    else:
        recomendacion = resultado["recomendacion"]

        if recomendacion.startswith("ESPERA"):
            st.warning(f"### ⏳ {recomendacion}")
        elif "subir" in recomendacion:
            st.error(f"### 📈 {recomendacion}")
        else:
            st.success(f"### ✅ {recomendacion}")

        st.write(f"Ruta seleccionada: **Madrid → {destino.title()}**")

        metrica_1, metrica_2, metrica_3, metrica_4 = st.columns(4)

        metrica_1.metric(
            "Precio estimado hoy",
            f"{resultado['precio_estimado_hoy']:.2f} €",
        )

        metrica_2.metric(
            "Precio mínimo esperado",
            f"{resultado['precio_minimo_esperado']:.2f} €",
        )

        metrica_3.metric(
            "Ahorro estimado",
            f"{resultado['ahorro_estimado_eur']:.2f} €",
            f"{resultado['ahorro_estimado_pct']:.2f} %",
        )

        metrica_4.metric(
            "Antelación óptima",
            f"{resultado['antelacion_optima_dias']} días",
        )

        st.subheader("Curva estimada de precio y antelación")

        curva = pd.DataFrame(resultado["curva"])

        if not curva.empty:
            curva = curva.sort_values("dias_anticipacion")

            st.line_chart(
                curva,
                x="dias_anticipacion",
                y="precio_estimado",
            )

            with st.expander("Ver datos de la curva"):
                st.dataframe(
                    curva.rename(
                        columns={
                            "dias_anticipacion": "Días de antelación",
                            "precio_estimado": "Precio estimado (€)",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning("La API no ha devuelto puntos para la curva.")

        with st.expander("Ver consulta enviada a la API"):
            st.json(payload)

        with st.expander("Ver respuesta completa de la API"):
            st.json(resultado)

st.divider()

st.caption(
    "TFM · Renfe Price Optimizer · Modelo supervisado de regresión XGBoost"
)

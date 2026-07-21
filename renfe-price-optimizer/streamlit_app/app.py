"""Interfaz Streamlit — Renfe Price Optimizer.

Consume la API FastAPI y muestra la recomendación al usuario final.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Renfe Price Optimizer", page_icon="🚄", layout="wide")

st.title("🚄 Renfe Price Optimizer")
st.caption("Madrid ↔ Barcelona · TFM MLOps 2026 · Montenegro · Nieto · Pardo")

col1, col2 = st.columns(2)

with col1:
    vehicle_type = st.selectbox("Servicio", ["AVE", "AVE-TGV", "AVLO", "ALVIA"])
    vehicle_class = st.selectbox("Clase", ["Turista", "Turista Plus", "Preferente"])
    fare = st.selectbox("Tarifa", ["Promo", "Promo +", "Flexible", "Adulto ida"])
    duration = st.number_input("Duración (horas)", 1.0, 6.0, 2.75, 0.25)

with col2:
    dias_anticipacion = st.slider("Días de antelación desde hoy", 0, 180, 30)
    hora_salida = st.slider("Hora de salida", 0, 23, 9)
    dia_semana = st.selectbox(
        "Día de la semana",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda d: ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"][d],
    )
    mes = st.slider("Mes del viaje", 1, 12, 6)

temporada = {12: "invierno", 1: "invierno", 2: "invierno",
             3: "primavera", 4: "primavera", 5: "primavera",
             6: "verano", 7: "verano", 8: "verano",
             9: "otoño", 10: "otoño", 11: "otoño"}[mes]

if st.button("🔍 ¿Compro o espero?", type="primary"):
    payload = {
        "vehicle_type": vehicle_type,
        "vehicle_class": vehicle_class,
        "fare": fare,
        "duration": duration,
        "dias_anticipacion": dias_anticipacion,
        "hora_salida": hora_salida,
        "dia_semana": dia_semana,
        "mes": mes,
        "temporada": temporada,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        st.error(f"No se pudo contactar con la API: {exc}")
        st.stop()

    st.success(f"### 💡 {data['recomendacion']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio estimado hoy", f"{data['precio_estimado_hoy']} €")
    m2.metric("Precio mínimo esperado", f"{data['precio_minimo_esperado']} €")
    m3.metric("Ahorro estimado", f"{data['ahorro_estimado_eur']} €",
              f"{data['ahorro_estimado_pct']}%")
    m4.metric("Antelación óptima", f"{data['antelacion_optima_dias']} días")

    st.subheader("Curva precio-antelación estimada")
    curva = pd.DataFrame(data["curva"]).sort_values("dias_anticipacion")
    st.line_chart(curva, x="dias_anticipacion", y="precio_estimado")

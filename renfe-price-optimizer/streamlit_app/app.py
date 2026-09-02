"""Interfaz moderna del Renfe Price Optimizer (corredores bidireccionales)."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
APP_DIR = Path(__file__).resolve().parent

# Las 5 ciudades disponibles como origen y destino. Solo MÁLAGA lleva tilde.
CIUDADES = ["MADRID", "BARCELONA", "SEVILLA", "VALENCIA", "MÁLAGA"]
TIPOS_TREN = ["AVE", "R. EXPRES", "AV City", "INTERCITY", "ALVIA", "MD", "REGIONAL"]
CLASES = ["Turista", "Turista Plus", "Preferente"]
TARIFAS = ["Promo", "Flexible", "Adulto ida"]
DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


def temporada_de(mes: int) -> str:
    if mes in (6, 7, 8):
        return "Verano"
    if mes in (12, 1, 2):
        return "Invierno"
    if mes in (3, 4, 5):
        return "Primavera"
    return "Otoño"


def cargar_css() -> None:
    css_path = APP_DIR / "assets" / "style.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def api_disponible() -> tuple[bool, str]:
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        response.raise_for_status()
        version = response.json().get("version", "activa")
        return True, str(version)
    except requests.RequestException:
        return False, "sin conexión"


def tarjeta_metricas(resultado: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio hoy", f"{resultado['precio_estimado_hoy']:.2f} €")
    c2.metric("Mínimo esperado", f"{resultado['precio_minimo_esperado']:.2f} €")
    c3.metric(
        "Ahorro potencial",
        f"{resultado['ahorro_estimado_eur']:.2f} €",
        f"{resultado['ahorro_estimado_pct']:.2f} %",
    )
    c4.metric("Antelación óptima", f"{resultado['antelacion_optima_dias']} días")


st.set_page_config(
    page_title="Renfe Price Optimizer",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded",
)
cargar_css()

# Hero
st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">TFM · XGBoost · MLOps</div>
      <h1>Compra tu billete en el mejor momento</h1>
      <p>Simula la evolución del precio y recibe una recomendación clara para tus trayectos entre Madrid, Barcelona, Sevilla, Valencia y Málaga.</p>
      <div class="hero-badges">
        <span>13 M+ registros</span><span>MAPE 7,67 %</span><span>R² 0,94</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

online, version = api_disponible()
with st.sidebar:
    st.markdown("## Renfe Optimizer")
    st.caption("Asistente de decisión de compra")
    if online:
        st.success(f"API conectada · {version}")
    else:
        st.error("API no disponible")
        st.caption("Ejecuta Uvicorn en el puerto 8000.")

    st.divider()
    st.markdown("### Configuración")
    st.caption("Modelo XGBoost · corredores bidireccionales")
    st.caption("Madrid · Barcelona · Sevilla · Valencia · Málaga")
    st.divider()
    st.caption("Las estimaciones son orientativas y se basan en datos históricos de 2019.")

# Formulario en tarjeta
st.markdown('<div class="section-label">PLANIFICA TU VIAJE</div>', unsafe_allow_html=True)
with st.container(border=True):
    top1, top2, top3 = st.columns([1.15, 1.15, 0.8])
    with top1:
        origen = st.selectbox("Origen", CIUDADES, index=0)
    with top2:
        # El destino excluye la ciudad elegida como origen para evitar origen == destino.
        destinos_disponibles = [c for c in CIUDADES if c != origen]
        destino = st.selectbox("Destino", destinos_disponibles)
    with top3:
        mes = st.selectbox("Mes", list(range(1, 13)), index=5)

    st.divider()
    a, b, c = st.columns(3)
    with a:
        vehicle_type = st.selectbox("Tipo de tren", TIPOS_TREN)
        vehicle_class = st.selectbox("Clase", CLASES)
        fare = st.selectbox("Tarifa", TARIFAS)
    with b:
        duration = st.number_input(
            "Duración estimada (horas)", min_value=0.5, max_value=12.0, value=2.75, step=0.05
        )
        hora_salida = st.slider("Hora de salida", 0, 23, 9)
        dia_semana = st.selectbox("Día de la semana", DIAS_SEMANA, index=3)
    with c:
        dias_anticipacion = st.slider("Días de antelación", 0, 180, 30)
        temporada = temporada_de(mes)
        st.text_input("Temporada", value=temporada, disabled=True)
        st.markdown(
            f"<div class='route-card'><span>Ruta</span><strong>{origen.title()} → {destino.title()}</strong>"
            f"<small>{vehicle_type} · {vehicle_class} · {fare}</small></div>",
            unsafe_allow_html=True,
        )

    analizar = st.button("Analizar mejor momento de compra", type="primary", use_container_width=True)

if analizar:
    if origen == destino:
        st.warning("El origen y el destino no pueden ser la misma ciudad. Elige un destino diferente.")
        st.stop()

    payload = {
        "origen": origen,
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
        with st.spinner("Analizando la curva de precio..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            response.raise_for_status()
            resultado = response.json()
    except requests.ConnectionError:
        st.error("No se ha podido conectar con la API. Comprueba que Uvicorn está activo.")
    except requests.Timeout:
        st.error("La API ha tardado demasiado en responder.")
    except requests.HTTPError:
        st.error(f"La API devolvió un error {response.status_code}: {response.text}")
    except requests.RequestException as exc:
        st.error(f"Error al consultar la API: {exc}")
    else:
        recomendacion = resultado["recomendacion"]
        estilo = "wait" if recomendacion.startswith("ESPERA") else "buy"
        icono = "◷" if estilo == "wait" else "✓"
        st.markdown(
            f"""
            <section class="decision {estilo}">
              <div class="decision-icon">{icono}</div>
              <div><span>RECOMENDACIÓN</span><h2>{recomendacion}</h2>
              <p>{origen.title()} → {destino.title()} · {vehicle_type} · {vehicle_class}</p></div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        tarjeta_metricas(resultado)

        tab1, tab2, tab3 = st.tabs(["Evolución del precio", "Resumen del viaje", "Detalle técnico"])
        with tab1:
            curva = pd.DataFrame(resultado["curva"])
            if not curva.empty:
                curva = curva.sort_values("dias_anticipacion")
                st.line_chart(curva, x="dias_anticipacion", y="precio_estimado", height=380)
                st.caption("A la izquierda, menor antelación. A la derecha, compra más anticipada.")
        with tab2:
            r1, r2 = st.columns(2)
            r1.markdown(f"**Origen**  \n{origen.title()}")
            r1.markdown(f"**Destino**  \n{destino.title()}")
            r1.markdown(f"**Servicio**  \n{vehicle_type}")
            r2.markdown(f"**Clase y tarifa**  \n{vehicle_class} · {fare}")
            r2.markdown(f"**Salida**  \n{dia_semana}, {hora_salida}:00")
            r2.markdown(f"**Temporada**  \n{temporada}")
        with tab3:
            st.json({"consulta": payload, "respuesta": resultado})

st.markdown(
    """
    <footer class="footer">Renfe Price Optimizer · XGBoost · FastAPI · Streamlit · MLflow</footer>
    """,
    unsafe_allow_html=True,
)

# 🚄 Renfe Price Optimizer

[![CI](https://github.com/danipardo99/renfe-price-optimizer/actions/workflows/ci.yml/badge.svg)](https:e-optimizer/actions)

TFM MLOps para la predicción de precios de billetes ferroviarios y la recomendación del momento óptimo de compra en tres corredores con origen Madrid:

- Madrid → Barcelona
- Madrid → Sevilla
- Madrid → Valencia

**Autores:** Alejandro Montenegro · Laura Nieto · Daniel Pardo  
**Modelo final:** XGBoost · regresión tabular supervisada  
**Datos:** histórico de precios Renfe de 2019

## 🎯 Qué hace este proyecto

El sistema predice el precio de un billete en función del destino, el tipo de tren, la clase, la tarifa, la duración, los días de antelación, la hora de salida, el día de la semana, el mes y la temporada.

Además de estimar el precio, evalúa la curva precio-antelación y devuelve una recomendación accionable:

- **COMPRA HOY**
- **COMPRA HOY: el precio va a subir**
- **ESPERA X días**

El proyecto implementa un flujo MLOps end-to-end con:

- preparación y validación del dato;
- entrenamiento reproducible con XGBoost;
- tracking de experimentos y artefactos con MLflow;
- API de predicción con FastAPI;
- aplicación multidestino con Streamlit;
- empaquetado con Docker;
- validación automática con GitHub Actions;
- versionado de código con Git y de datos pesados mediante DVC o almacenamiento externo.

## 📊 Resultados del modelo final

El modelo final se entrenó sobre el dataset limpio multidestino generado por el proceso de EDA.

| Métrica | Resultado | Objetivo | Estado |
|---|---:|---:|:---:|
| Registros procesados | 13.025.112 | — | ✅ |
| MAE | 4,42 € | — | ✅ |
| RMSE | 6,73 € | — | ✅ |
| MAPE | 8,16 % | < 10 % | ✅ |
| R² | 0,93 | ≥ 0,80 | ✅ |

El modelo supera los objetivos técnicos definidos para el TFM. Los indicadores de comportamiento de usuario, ahorro real o conversión continúan siendo objetivos de negocio futuros, ya que requieren uso real de la plataforma.

## 🧠 Modelado

El problema se formula como **regresión tabular supervisada**.

Se compararon los siguientes enfoques:

1. Regresión lineal como baseline.
2. Random Forest como alternativa no lineal.
3. XGBoost como modelo final seleccionado.

XGBoost se utiliza en la versión servida por FastAPI y Streamlit debido a sus mejores resultados sobre el dataset multidestino.

---

## 🏗️ Arquitectura

```
CSV Renfe → data/raw (DVC) → features.py → train.py (MLflow) 
        → models/model.pkl → FastAPI /predict → Streamlit UI
                                     ↑
                        GitHub Actions (CI: ruff + pytest)
                                     ↑
                              Docker + docker-compose
```

## 📂 Estructura

```
renfe-price-optimizer/
├── README.md
├── MEMORIA_MLOPS.md              # 📘 Memoria completa para entregar (rúbrica)
├── GITHUB_SETUP.md               # 🚀 Guía paso a paso para crear el repo
├── pyproject.toml / requirements.txt / .gitignore / .env.example
├── params.yaml / dvc.yaml
├── Dockerfile / docker-compose.yml
├── .github/workflows/ci.yml
├── data/raw/renfe_clean_sample.csv     # dataset (versionado con DVC en real)
├── notebooks/01_eda.ipynb
├── src/renfe_optimizer/
│   ├── config.py · features.py · train.py · predict.py
│   └── api/ (main.py, schemas.py, routers/predict.py)
├── streamlit_app/app.py
├── tests/ (test_features.py, test_api.py)
└── docs/architecture.md
```

---

## ⚡ Arranque rápido (VS Code + WSL2)

```bash
# 1. Clonar
git clone https://github.com/<tu-usuario>/renfe-price-optimizer.git
cd renfe-price-optimizer

# 2. Entorno
uv sync                       # o: pip install -r requirements.txt

# 3. Tests + estilo
uv run ruff check .
uv run pytest

# 4. Entrenamiento con tracking en MLflow
uv run python -m renfe_optimizer.train --config params.yaml

# 5. Revisar experimentos
uv run mlflow ui              # http://localhost:5000

# 6. Levantar API + Streamlit
uv run uvicorn renfe_optimizer.api.main:app --reload
uv run streamlit run streamlit_app/app.py
```

## 🐳 O con Docker

```bash
docker compose up --build
# FastAPI:   http://localhost:8000/docs
# Streamlit: http://localhost:8501
```

---

## 📊 KPIs

Distinguimos entre **KPIs técnicos medibles en el TFM** (evaluables con el dataset de 2019) y **KPIs de negocio** (objetivos a futuro que requieren usuarios reales y, por tanto, NO son medibles en este trabajo).

### KPIs técnicos (medibles en el TFM)

| KPI | Meta |
|---|---|
| MAPE del modelo | < 10 % |
| R² del modelo | ≥ 0,80 |
| MAE / RMSE | Reportados en cada run de MLflow |
| Latencia de recomendación | < 3 s |

### KPIs de negocio (objetivo a futuro, NO medibles con datos de 2019)

| KPI | Meta |
|---|---|
| Ahorro medio por billete | ≥ 15 % |
| % compras en ventana óptima | ≥ 60 % |
| Tasa de conversión | +10 p.p. |
| Plataformas piloto integradas | 2 |

---

## 📄 Licencia

Trabajo académico — TFM Máster en Data Science 2026.
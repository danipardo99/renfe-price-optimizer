# renfe-price-optimizer

**TFM MLOps — Predicción de precios dinámicos Renfe (Madrid–Barcelona) y recomendación del momento óptimo de compra.**

Autores: Alejandro Montenegro · Laura Nieto · Daniel Pardo · *Julio 2026*

---

## 🎯 Qué hace este proyecto

A partir de un histórico real de precios de Renfe (AVE, AVLO, ALVIA) en el corredor Madrid–Barcelona (abril–agosto 2019), entrena modelos supervisados que **predicen el precio del billete** en función de la antelación de compra y el resto de variables explicativas, y devuelve al usuario una **recomendación accionable**: *«compra hoy»*, *«espera X días»* o *«alerta: el precio va a subir»*.

El objetivo del repositorio **no es solo el modelo**, es demostrar un flujo **MLOps end-to-end**: estructura profesional, control de versiones (Git + DVC), experimentación reproducible (MLflow), servicio (FastAPI + Streamlit), empaquetado (Docker) y validación automática (GitHub Actions).

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
├── data/raw/renfe_sample.csv     # dataset (versionado con DVC en real)
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

## 📊 KPIs objetivo

| KPI | Meta |
|---|---|
| Ahorro medio por billete | ≥ 15 % |
| % compras en ventana óptima | ≥ 60 % |
| MAPE del modelo | < 10 % |
| R² del modelo | ≥ 0,80 |
| Latencia de recomendación | < 3 s |

---

## 📄 Licencia

Trabajo académico — TFM Máster en Data Science 2026.

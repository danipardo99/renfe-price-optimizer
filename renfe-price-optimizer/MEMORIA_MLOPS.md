# Trabajo de MLOps — Renfe Price Optimizer

**Trabajo Fin de Máster · Bloque MLOps · Julio de 2026**

**Autores:** Alejandro Montenegro · Laura Nieto · Daniel Pardo

**Caso conductor:** predicción de precios dinámicos Renfe en el corredor Madrid–Barcelona y recomendación del momento óptimo de compra.

---

## Resumen ejecutivo

Este documento presenta el planteamiento MLOps aplicado al TFM *Predicción de precios dinámicos en el sector ferroviario español de alta velocidad*. Sobre un histórico real de precios de la web oficial de Renfe (abril–agosto 2019), montamos una arquitectura MLOps end-to-end que va del CSV en crudo hasta una plataforma web accionable, pasando por versionado de datos (DVC), experimentación reproducible (MLflow), servicio (FastAPI + Streamlit), empaquetado (Docker) y validación automática (GitHub Actions).

El trabajo cubre los seis criterios de la rúbrica de evaluación (10 puntos totales) y se acompaña de un repositorio ejecutable con el código, los tests, los workflows y la memoria.

### 🎯 Trazabilidad rúbrica ↔ documento

| Criterio de la rúbrica | Puntos | Sección(es) de este documento | Evidencia concreta |
|---|---|---|---|
| Comprensión de MLOps | 2 | §1 | Definición, DevOps vs MLOps, ciclo de vida |
| Ciclo de vida y pipelines | 2 | §2, §5.6 | Fases + separación DataOps / MLOps + pipeline Renfe |
| Uso de herramientas y plataformas | 1,5 | §3 | MLflow, DVC, FastAPI + comparativa Azure ML / SageMaker / Kubeflow |
| Automatización y gobernanza | 1,5 | §4 | Reproducibilidad, trazabilidad, versionado, monitorización, secrets |
| Calidad de la exposición | 1 | Todo el doc | Estructura, diagramas ASCII, tablas, código |
| Propuesta aplicada / caso práctico | 2 | §5 completo | GitHub + VS Code + DVC + MLflow + FastAPI + Docker sobre Renfe |

---

## 1. ¿Qué es MLOps y por qué existe? ▶ Criterio 1

**MLOps** (Machine Learning Operations) es la disciplina que aplica los principios de DevOps al ciclo de vida completo de los modelos de Machine Learning. No es solo "poner un modelo en producción": es garantizar que el modelo se puede **entrenar de forma repetible, desplegar sin fricción, monitorizar en el tiempo y actualizar sin romper nada**.

El problema que resuelve MLOps es concreto: entre el 60 % y el 80 % de los modelos que entrena un equipo de Data Science **nunca llegan a producción**. Y de los que llegan, la mayoría se degradan silenciosamente porque nadie observa la deriva. MLOps existe para cerrar la brecha entre el notebook del data scientist y el sistema real que consume otro equipo, otro sistema o el usuario final.

### 1.1. DevOps · DataOps · MLOps: qué aporta cada uno

```
                    ┌─────────────┐
                    │   DevOps    │  Código de software
                    │  (código)   │  · Git, CI/CD, tests, deploy
                    └─────────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      ┌─────────────┐           ┌─────────────┐
      │   DataOps   │           │    MLOps    │
      │   (datos)   │           │  (modelos)  │
      └─────────────┘           └─────────────┘
      Pipelines de datos         Pipelines de ML
      · ingestas, ETL            · entrenamiento
      · calidad, catálogo        · experimentación
      · linaje de datos          · registro de modelos
      · Airflow, dbt,            · monitorización de deriva
        Apache Hop               · MLflow, Kubeflow
```

| Disciplina | Foco | Artefacto principal | Herramientas típicas |
|---|---|---|---|
| **DevOps** | Software estable en producción | Código + binarios | Git, GitHub Actions, Docker |
| **DataOps** | Datos confiables, actualizados y trazables | Datasets + pipelines | Airflow, dbt, Apache Hop |
| **MLOps** | Modelos reproducibles y monitorizables | Modelos + experimentos | MLflow, DVC, SageMaker, Kubeflow |

### 1.2. Rol de MLOps en un proyecto de Data Science

En un TFM tradicional, el data scientist termina con un notebook que entrena un modelo y muestra unas métricas. **Ese notebook no es un producto**: nadie más puede ejecutarlo, no hay pruebas, no hay versionado del dato, no hay forma de saber qué versión del modelo devolvió una predicción concreta. MLOps convierte ese notebook en un **sistema pequeño, reproducible y observable**.

Aplicado a nuestro TFM Renfe, el rol de MLOps es:

- Garantizar que **cualquier miembro del equipo** (o el tribunal) puede clonar el repositorio y ejecutar el pipeline completo.
- Que **cada experimento** (probando XGBoost, Random Forest, Prophet…) queda registrado con sus parámetros y métricas.
- Que el **modelo servido** por la plataforma Streamlit es exactamente el que se entrenó y validó, sin ambigüedades.
- Que si mañana **cambian los datos** (post-liberalización, con Ouigo e Iryo), el pipeline se re-ejecuta con un solo comando.

---

## 2. Ciclo de vida y pipelines ▶ Criterio 2

El ciclo de vida de un proyecto MLOps se estructura en cinco fases encadenadas por *feedback loops*:

```
   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │  Datos   │→→│ Training │→→│ Validación│→→│  Deploy  │→→│Monitoring│
   │(DataOps) │  │          │  │          │  │          │  │          │
   └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
        ▲                                                        │
        └────────────────────── retraining ◄─────────────────────┘
```

### 2.1. Fases del ciclo

| Fase | Objetivo | Actividades | En el TFM Renfe |
|---|---|---|---|
| **1. Datos** | Ingesta, limpieza y versionado | Extracción, validación, feature engineering | CSV Kaggle → capa raw → ETL Apache Hop → DWH MySQL |
| **2. Training** | Entrenar modelos reproducibles | Split, tuning, tracking de experimentos | XGBoost / RF / SARIMAX con MLflow |
| **3. Validación** | Comprobar métricas y sesgos | MAE, RMSE, MAPE, R², SHAP, walk-forward | Meta: MAPE < 10 %, R² ≥ 0,80 |
| **4. Deploy** | Servir el modelo | Empaquetado, API, contenedor | FastAPI + Streamlit + Docker |
| **5. Monitoring** | Vigilar rendimiento y drift | Métricas de servicio, drift de datos | Logs FastAPI + comparativa de métricas MLflow entre runs |

### 2.2. CI, CD y CT — los tres pilares de la automatización

- **CI (Continuous Integration).** Cada Pull Request desencadena GitHub Actions: instalar dependencias, `ruff check`, `pytest`, smoke test del entrenamiento con una muestra pequeña. Si algo falla, el merge se bloquea.
- **CD (Continuous Delivery / Deployment).** Al mergear a `main`, se construye la imagen Docker y se publica un artefacto (en la versión productiva, se subiría a un registry como Azure Container Registry o Docker Hub).
- **CT (Continuous Training).** Cuando cambian los datos o se detecta deriva, el pipeline DVC re-ejecuta `prepare → train → evaluate` y registra un nuevo run en MLflow. Si supera al modelo actual, se promociona a "Production" en el Model Registry.

### 2.3. Pipeline de datos (DataOps) vs pipeline de ML (MLOps)

Es una distinción que la rúbrica pide explícitamente:

| Aspecto | Pipeline de datos (DataOps) | Pipeline de ML (MLOps) |
|---|---|---|
| Entrada | Fuentes en crudo (CSV, APIs, DB) | Dataset ya procesado |
| Salida | Tabla limpia en el DWH | Modelo entrenado + métricas |
| Frecuencia | Batch diario/horario o streaming | Bajo demanda o programado |
| Éxito | Datos frescos, correctos, trazables | Modelo válido, sin deriva, reproducible |
| Herramienta en el TFM | Apache Hop → MySQL | MLflow + DVC + XGBoost |

Ambos pipelines conviven en el mismo proyecto pero **tienen dueños distintos**: el de datos alimenta al de ML, no al revés. En nuestro TFM Renfe, Apache Hop se encarga de dejar la tabla limpia en MySQL; a partir de ahí, `renfe_optimizer.features` y `renfe_optimizer.train` orquestan el pipeline de ML.

---

## 3. Herramientas y plataformas ▶ Criterio 3

### 3.1. Stack elegido para el TFM

| Capa | Herramienta | Por qué |
|---|---|---|
| Control de versiones de código | **Git + GitHub** | Estándar de facto; integración nativa con Actions |
| Versionado de datos y modelos | **DVC** | Git no maneja datasets pesados; DVC guarda punteros y sincroniza con Google Drive |
| Experimentación | **MLflow** | Registra params, metrics, artifacts y ofrece un Model Registry local; open source |
| Servicio API | **FastAPI** | Alto rendimiento, validación automática con Pydantic, `/docs` interactivo |
| Interfaz de usuario | **Streamlit** | Prototipo web funcional en 50 líneas; ideal para demo al tribunal |
| Empaquetado | **Docker + Compose** | Reproduce el entorno exacto en cualquier máquina |
| CI/CD | **GitHub Actions** | Integrado con el repo; ejecuta ruff, pytest y smoke test en cada PR |
| Entorno Python | **uv + pyproject.toml** | Rápido, lockfile determinista, sustituye pip+venv |
| IDE | **VS Code + WSL2** | Trabajar sobre Linux desde Windows sin cambiar de máquina |

### 3.2. Comparativa con las plataformas MLOps de mercado

Aunque hemos elegido un stack open source ligero para el TFM, conviene saber qué ofrecen las plataformas gestionadas de nube:

| Plataforma | Ventajas | Cuándo elegirla |
|---|---|---|
| **Azure Machine Learning** | Integración total con el ecosistema Microsoft; AutoML, endpoints gestionados, compute clusters bajo demanda; RBAC empresarial | Empresa ya "azureñada" (Endesa/Enel, banca) |
| **AWS SageMaker** | El más maduro; SageMaker Pipelines, Feature Store, Model Registry, Endpoints multimodelo | Startups y empresas AWS-first |
| **Google Vertex AI** | Fuerte en AutoML de tabular/visión, integración con BigQuery, generativa nativa (Gemini) | Datos ya en BigQuery, casos de generativa |
| **Kubeflow** | Open source, kubernetes-native, portable entre nubes | Equipos con maestría en Kubernetes |
| **Databricks + MLflow** | Notebooks + Spark + MLflow gestionado en una sola plataforma | Datasets masivos, lakehouse |
| **MLflow open source** *(TFM)* | Ligero, sin coste, se integra con cualquier framework, portable | TFM, prototipos, equipos pequeños |

**Ventajas comunes de una plataforma MLOps gestionada**:

1. **Trazabilidad extremo a extremo** (código, dato, modelo, métrica, endpoint).
2. **Model Registry con etapas** (Staging / Production / Archived) que evita ambigüedad de "¿qué modelo hay ahora?".
3. **Compute escalable bajo demanda** (entrenamientos GPU sin gestionar máquinas).
4. **Endpoints gestionados** con autoescalado, autenticación, logs y monitorización de latencia.
5. **Gobernanza y auditoría** (quién ejecutó qué, con qué versión del dato, cuándo).

Para el TFM la elección de MLflow + DVC + FastAPI cubre todos estos puntos de forma didáctica sin coste, y demuestra que el alumno entiende qué habría que exigirle a una plataforma cuando la empresa escalara el proyecto.

---

## 4. Automatización y gobernanza ▶ Criterio 4

MLOps solo funciona si cuatro cualidades están garantizadas:

### 4.1. Reproducibilidad

Cualquier persona debe poder **reconstruir el mismo resultado** a partir del repositorio.

- Entorno declarado en `pyproject.toml` + `uv.lock` (versiones fijas).
- Parámetros del experimento en `params.yaml` (no en el código).
- Datos versionados con DVC (`data/raw/*.dvc` en Git; el dato pesado en Drive).
- Pipeline explícito en `dvc.yaml`.
- Comandos de arranque documentados en el README.

**Test de reproducibilidad**: un compañero clona, ejecuta 5 comandos y obtiene las mismas métricas.

### 4.2. Trazabilidad

Ante una predicción de producción, debe poder responderse: *¿qué modelo la generó, con qué datos y con qué código?*

- **Git commit** identifica la versión del código.
- **`.dvc`** identifica la versión del dataset.
- **MLflow run_id** identifica la versión del modelo, sus parámetros y sus métricas.
- La API expone `/health` con la versión desplegada.

### 4.3. Versionado de modelos y datos

| Elemento | Versionado con | Motivo |
|---|---|---|
| Código | Git | Pequeño, texto, colaborativo |
| Notebooks limpios | Git | Documentan decisiones |
| Datos brutos | **DVC** | Grandes, binarios, cambian |
| Datos procesados | DVC (opcional) | Solo si son costosos de regenerar |
| Modelos entrenados | **DVC + MLflow Model Registry** | Trazables y promocionables |
| Secretos / credenciales | **NUNCA en Git.** `.env` local + GitHub Secrets | Compromiso de seguridad |

### 4.4. Monitorización

En un TFM académico, la monitorización se reduce a tres capas:

1. **Métricas del modelo** (evaluación offline): quedan registradas en MLflow. Al reentrenar, comparamos MAPE, R², RMSE contra el modelo actual antes de promocionar.
2. **Métricas del servicio** (online): logs de FastAPI con latencia por request; `/health` para health-check.
3. **Deriva de datos**: comparación periódica de la distribución de features nuevas contra la de entrenamiento (KS-test, PSI). Se planifica como línea futura del TFM.

En producción se integrarían herramientas como **Prometheus + Grafana**, **Evidently AI** o los servicios nativos de la nube.

### 4.5. Gobernanza de secretos

Regla de oro del bloque 4 y 6:

- `.env.example` documenta las variables necesarias, se sube a Git.
- `.env` contiene los valores reales, **nunca se sube** (incluido en `.gitignore`).
- Para CI/CD, los secretos se cargan como **GitHub Secrets** y se inyectan como variables de entorno.
- El código Python nunca "hardcodea" contraseñas; usa `os.getenv()`.

---

## 5. Caso práctico aplicado — Renfe Price Optimizer ▶ Criterio 6

### 5.1. Arquitectura del proyecto

```
                    ┌──────────────────────────────────────────┐
                    │      RENFE PRICE OPTIMIZER · MLOps       │
                    └──────────────────────────────────────────┘

  CSV Kaggle Renfe                                          Usuario final
    (2019, MAD-BCN)                                    (viajero / broker)
         │                                                     ▲
         ▼                                                     │
   ┌─────────────┐                                     ┌───────────────┐
   │ data/raw/   │◄──── DVC ────► Google Drive         │ Streamlit UI  │
   │  (.dvc)     │        (remote)                     │  (Recomienda) │
   └─────────────┘                                     └───────────────┘
         │                                                     ▲
         ▼                                                     │
   ┌─────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────┐
   │features.py  │──►│  train.py    │──►│ model.pkl  │──►│ FastAPI  │
   │ (ETL Python)│   │  + MLflow    │   │  + DVC     │   │ /predict │
   └─────────────┘   └──────────────┘   └────────────┘   └──────────┘
         ▲                   │                                 ▲
         │                   ▼                                 │
   ┌─────────────┐   ┌──────────────┐                   ┌──────────┐
   │ params.yaml │   │ MLflow UI    │                   │  Docker  │
   │  (config)   │   │ mlruns/      │                   │ Compose  │
   └─────────────┘   └──────────────┘                   └──────────┘
                             │
                             ▼
                     ┌────────────────┐
                     │ GitHub Actions │
                     │  CI: ruff +    │
                     │  pytest + smoke│
                     └────────────────┘
```

### 5.2. Montaje de GitHub paso a paso (VS Code + WSL2)

> Guía completa y ejecutable en [`GITHUB_SETUP.md`](./GITHUB_SETUP.md). Aquí resumimos los pasos clave.

```bash
# 1. Crear el repositorio en GitHub (web): "renfe-price-optimizer"
#    - Público, con .gitignore Python, licencia MIT.

# 2. Clonar en WSL desde VS Code
git clone https://github.com/<usuario>/renfe-price-optimizer.git
cd renfe-price-optimizer
code .

# 3. Copiar la estructura del proyecto
#    (ya generada por este TFM en el ZIP entregado)

# 4. Primer commit
git add .
git commit -m "chore: bootstrap del proyecto Renfe Price Optimizer"
git push origin main

# 5. Proteger main (Settings → Branches → main → Require PR)
```

Reglas del flujo (bloque 2):

- Nadie hace push directo a `main`.
- Cada tarea → rama `feature/xxx`.
- Al terminar, PR con descripción y check verde de GitHub Actions.
- Merge solo si CI pasa y hay revisión.

### 5.3. Entorno de desarrollo en VS Code

**Stack local sugerido** (Windows + WSL2 según bloque 3):

```bash
# Dentro de Ubuntu WSL
sudo apt update && sudo apt install git curl -y
curl -LsSf https://astral.sh/uv/install.sh | sh

cd renfe-price-optimizer
uv sync                    # instala todas las dependencias
uv run pytest              # tests iniciales verdes
```

**Extensiones VS Code imprescindibles**:

| Extensión | Uso |
|---|---|
| Python + Pylance | Autocompletado y errores |
| Jupyter | Notebooks embebidos |
| GitLens | Historial de cada línea |
| Docker | Gestionar imágenes y contenedores |
| Ruff | Linter en tiempo real |
| Thunder Client | Probar la API sin salir del IDE |
| GitHub Pull Requests | Revisar PRs desde VS Code |

### 5.4. Estructura de carpetas final

```
renfe-price-optimizer/
├── README.md                    # arranque rápido
├── MEMORIA_MLOPS.md             # este documento
├── GITHUB_SETUP.md              # guía paso a paso para crear el repo
├── pyproject.toml               # dependencias y metadatos
├── requirements.txt             # alternativa pip
├── .gitignore                   # archivos que NO van a Git
├── .env.example                 # variables necesarias (sin valores)
├── params.yaml                  # parámetros del pipeline
├── dvc.yaml                     # pipeline reproducible con DVC
├── Dockerfile                   # empaquetado de la API
├── docker-compose.yml           # api + streamlit + mlflow
├── .github/workflows/ci.yml     # GitHub Actions
├── data/
│   ├── raw/renfe_sample.csv     # muestra versionable
│   └── processed/               # generado por features.py
├── models/                      # modelo entrenado (DVC)
├── reports/metrics.json         # métricas del último run
├── notebooks/
│   └── 01_eda.ipynb             # exploración
├── src/renfe_optimizer/
│   ├── __init__.py
│   ├── config.py                # rutas y .env
│   ├── features.py              # ingeniería de variables
│   ├── train.py                 # entrenamiento + MLflow
│   ├── predict.py               # servicio de recomendación
│   └── api/
│       ├── main.py              # FastAPI app
│       ├── schemas.py           # Pydantic
│       └── routers/predict.py   # endpoint /predict
├── streamlit_app/app.py         # UI
├── tests/
│   ├── test_features.py         # unit tests
│   └── test_api.py              # smoke tests
└── docs/architecture.md         # ADRs y arquitectura
```

### 5.5. Control de versiones — código Y documentación

Aplicamos versionado con Git también a la documentación:

- **README.md** vive en Git y evoluciona con el proyecto.
- **MEMORIA_MLOPS.md** (este documento) va en Git.
- **docs/architecture.md** recoge decisiones arquitectónicas (ADRs).
- **CHANGELOG.md** (opcional) documenta cambios entre versiones.

Cada cambio significativo en la documentación → commit con mensaje descriptivo → PR → revisión. La memoria del TFM es un artefacto **vivo y trazable**, no un Word suelto.

### 5.6. Pipeline MLflow — instalación y uso

**Instalación** (ya incluida en `requirements.txt`):

```bash
uv add mlflow scikit-learn xgboost pandas
```

**Lanzar la interfaz local**:

```bash
uv run mlflow ui --backend-store-uri ./mlruns
# http://localhost:5000
```

**Registrar un entrenamiento** (extracto de `src/renfe_optimizer/train.py`):

```python
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("renfe-price-optimizer")

with mlflow.start_run(run_name="xgboost-baseline"):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = evaluate(y_test, y_pred)   # MAE, RMSE, MAPE, R²

    mlflow.log_params(params["model"]["params"])
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_metrics(metrics)

    joblib.dump(pipe, model_out)
    mlflow.log_artifact(str(model_out), artifact_path="model")
```

**Qué queda registrado en cada run**:

| Elemento | Ejemplo |
|---|---|
| Parámetros | `n_estimators=400`, `max_depth=6`, `learning_rate=0.08` |
| Métricas | `MAE=8.3`, `RMSE=12.5`, `MAPE=9.4 %`, `R²=0.83` |
| Artefactos | `renfe_model.pkl`, gráfico de importancia SHAP |
| Modelo | Objeto sklearn/xgboost listo para cargar |
| Metadatos | `run_id`, `git_commit`, timestamp, usuario |

**Model Registry** (siguiente paso):

```python
# Registrar y promocionar el mejor run
mlflow.register_model("runs:/<run_id>/model", "renfe-price-optimizer")
# Luego, desde la UI: promocionar a "Staging" o "Production"
```

### 5.7. CI con GitHub Actions

El workflow `.github/workflows/ci.yml` se ejecuta en cada PR:

```yaml
name: renfe-optimizer-ci
on:
  push:    { branches: [main] }
  pull_request: { branches: [main] }

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - name: Ruff (estilo)
        run: ruff check .
      - name: Pytest (unit + smoke)
        run: pytest -v
      - name: Smoke train
        run: python -m renfe_optimizer.train --config params.yaml --sample --max-rows 500
```

**Qué valida cada paso**:

- `ruff check .` → estilo y errores comunes de Python.
- `pytest -v` → tests unitarios de features y smoke test de la API.
- `python -m renfe_optimizer.train --sample` → el pipeline arranca de punta a punta con una muestra.

Si algo se rompe en verde, la PR queda bloqueada y GitHub muestra el log.

### 5.8. Reproducibilidad end-to-end

**El test definitivo**: un compañero clona el repo desde cero.

```bash
git clone https://github.com/<usuario>/renfe-price-optimizer.git
cd renfe-price-optimizer
uv sync
dvc pull                                       # opcional: trae los datos
uv run pytest                                  # tests en verde
uv run python -m renfe_optimizer.features      # genera dataset
uv run python -m renfe_optimizer.train         # entrena y registra en MLflow
uv run mlflow ui                               # revisa experimentos
uv run uvicorn renfe_optimizer.api.main:app    # levanta la API
uv run streamlit run streamlit_app/app.py      # levanta la UI
```

O directamente con Docker:

```bash
docker compose up --build
# API en :8000, Streamlit en :8501, MLflow en :5000
```

---

## 6. Despliegue con FastAPI + Docker (anexo del caso práctico)

### 6.1. FastAPI — modelo servido como API

El endpoint `/predict` recibe una consulta del usuario y devuelve la recomendación. La documentación interactiva vive en `/docs` gracias a FastAPI + Pydantic.

**Ejemplo de request**:

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "vehicle_type": "AVE",
       "vehicle_class": "Turista",
       "fare": "Promo",
       "duration": 2.75,
       "dias_anticipacion": 45,
       "hora_salida": 9,
       "dia_semana": 4,
       "mes": 6,
       "temporada": "verano"
     }'
```

**Ejemplo de respuesta**:

```json
{
  "precio_estimado_hoy": 72.4,
  "precio_minimo_esperado": 58.9,
  "antelacion_optima_dias": 60,
  "ahorro_estimado_eur": 13.5,
  "ahorro_estimado_pct": 18.6,
  "recomendacion": "ESPERA 15 días",
  "curva": [ ... ]
}
```

### 6.2. Docker — empaquetado y despliegue

`Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt . && RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/ && COPY models/ models/
EXPOSE 8000
CMD ["uvicorn", "renfe_optimizer.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml` levanta tres servicios: **api** (FastAPI), **streamlit** (UI) y **mlflow** (tracking UI). Un solo `docker compose up --build` reproduce todo el sistema.

---

## 7. Monitorización y observabilidad

En un despliegue real del Renfe Price Optimizer, la monitorización se organizaría en tres capas:

### 7.1. Monitorización del servicio (infra)

- **Latencia** por request (objetivo TFM: < 3 s).
- **Throughput** (peticiones/segundo) y **error rate** (% de 5xx).
- **Salud del contenedor** (`/health` de FastAPI).
- Stack sugerido: **Prometheus + Grafana** en Kubernetes; **Application Insights** en Azure.

### 7.2. Monitorización del modelo (offline vs online)

- **Comparativa entre runs de MLflow**: al reentrenar, ¿mejora MAPE contra el modelo actual?
- **Métricas de negocio**: ¿los usuarios que siguen la recomendación ahorran realmente ese 15 %?

### 7.3. Deriva de datos (data drift)

El modelo entrenado sobre datos 2019 se degrada porque el mercado post-liberalización (Ouigo, Iryo desde 2021-2022) cambia la política tarifaria de Renfe. Detección propuesta:

- **PSI** (Population Stability Index) por variable.
- **KS-test** sobre la distribución de `price` y `dias_anticipacion`.
- Si drift > umbral → alarma + trigger de reentrenamiento.

---

## 8. Conclusiones y roadmap

Este trabajo demuestra cómo **transformar un notebook académico en un sistema MLOps real**. Sobre el caso conductor del TFM Renfe Madrid-Barcelona hemos montado:

- Estructura profesional de repositorio (bloque 1).
- Flujo colaborativo Git + GitHub con ramas, PRs y protección de main (bloque 2).
- Entorno VS Code + WSL2 preparado para trabajar como un ingeniero (bloque 3).
- CI con GitHub Actions que valida cada cambio (bloque 4).
- Servicio con FastAPI + Streamlit consumible por el usuario final (bloque 5).
- Empaquetado con Docker + Compose reproducible en cualquier máquina (bloque 6).
- Versionado con DVC y experimentación con MLflow (bloque 7).
- Caso práctico integral con checklist de entrega profesional (bloque 8).

**Roadmap de evolución**:

1. **Model Registry en producción**: promoción automática a Staging/Production tras métrica objetivo.
2. **Data drift automático**: job programado que compara distribuciones y dispara reentrenamiento.
3. **A/B testing** entre modelos (Random Forest vs XGBoost vs Prophet) sobre tráfico real.
4. **Ampliar el dataset** a datos post-liberalización con Ouigo e Iryo.
5. **Migrar el tracking** a Azure ML o Databricks cuando el volumen lo justifique.
6. **Evolucionar la interfaz** hacia un agente conversacional (Copilot Studio + LLM).

---

## Anexos

### Anexo A — Comandos MLflow más usados

```bash
mlflow ui                                              # UI local
mlflow experiments list
mlflow runs list --experiment-id 1
mlflow models serve -m runs:/<run_id>/model -p 5001
mlflow model register -m runs:/<run_id>/model \
                     -n renfe-price-optimizer
```

### Anexo B — Comandos DVC más usados

```bash
dvc init
dvc add data/raw/renfe_2019_full.csv                   # empieza a versionar
dvc remote add --default drive gdrive://<folder-id>    # remoto Google Drive
dvc push                                                # sube el dato
dvc pull                                                # descarga el dato
dvc repro                                               # re-ejecuta el pipeline
```

### Anexo C — Buenas prácticas resumidas

| Regla | Consecuencia si se incumple |
|---|---|
| No push directo a `main` | Bugs sin revisar en producción |
| No subir `.env` a Git | Filtración de credenciales |
| No entrenar dentro de un notebook | Imposible reproducir en CI/CD |
| No usar rutas absolutas | Falla en máquinas de otros |
| Un commit = una idea | Historial ilegible |
| Cada PR debe pasar CI | Rotura silenciosa de `main` |
| Todo experimento va a MLflow | Se pierde qué funcionó |
| Datos pesados con DVC, nunca en Git | Repo inclonable |

---

*Fin del documento. Autores: Alejandro Montenegro · Laura Nieto · Daniel Pardo — Julio de 2026.*

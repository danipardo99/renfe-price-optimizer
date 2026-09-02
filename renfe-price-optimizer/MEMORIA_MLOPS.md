# Trabajo de MLOps — Renfe Price Optimizer

**Trabajo Fin de Máster · Bloque MLOps · Julio de 2026**

**Autores:** Alejandro Montenegro · Laura Nieto · Daniel Pardo

**Caso conductor:** predicción de precios dinámicos de Renfe en corredores bidireccionales entre Madrid, Barcelona, Sevilla, Valencia y Málaga —cada ciudad puede ser origen y destino—, y recomendación del momento óptimo de compra.

---

## Resumen ejecutivo

Este documento presenta el planteamiento MLOps aplicado al TFM _Predicción de precios dinámicos en el sector ferroviario español de alta velocidad_. Sobre un histórico real de precios de Renfe de 2019, el equipo construye una arquitectura MLOps end-to-end para **corredores bidireccionales** entre cinco ciudades (Madrid, Barcelona, Sevilla, Valencia y Málaga), en los que cada ciudad puede actuar como origen y como destino. El sistema recorre el ciclo completo desde el dato en crudo hasta una plataforma web accionable, pasando por limpieza y EDA, entrenamiento de un modelo XGBoost de regresión tabular, tracking con MLflow, servicio mediante FastAPI y Streamlit, empaquetado con Docker y validación automática con GitHub Actions.

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
- Que **cada experimento** (probando XGBoost, Random Forest, regresión lineal…) queda registrado con sus parámetros y métricas.
- Que el **modelo servido** por la plataforma Streamlit es exactamente el que se entrenó y validó, sin ambigüedades.
- Que si mañana **cambian los datos** (post-liberalización, con Ouigo e Iryo) o **se amplía el ámbito de corredores**, el pipeline se re-ejecuta con un solo comando.

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
| **2. Training** | Entrenar modelos reproducibles | Split, tuning, tracking de experimentos | XGBoost / RF / regresión lineal con MLflow |
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
    (2019, corredores bidireccionales                            (viajero / broker)
     MAD · BCN · SVQ · VLC · AGP)                                    ▲
         │                                                     │
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
│   ├── raw/renfe_clean_sample.csv     # muestra versionable
│   └── processed/               # generado por features.py
├── models/                      # modelo entrenado (DVC)
├── reports/metrics.json         # métricas del último run
├── notebooks/
├── 01_eda.ipynb
└── 02_eda_completo.ipynb             # exploración
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

with mlflow.start_run(run_name="xgboost"):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = evaluate(y_test, y_pred)   # MAE, RMSE, MAPE, R²

    mlflow.log_params(params["model"].get("params", {}))
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_metrics(metrics)

    joblib.dump(pipe, model_out)
    mlflow.log_artifact(str(model_out), artifact_path="model")
```

**Qué queda registrado en cada run**:

| Elemento | Ejemplo |
|---|---|
| Parámetros | `n_estimators=50`, `max_depth=8`, `random_state=42` |
| Métricas | `MAE=4.07`, `RMSE=6.17`, `MAPE=7.67 %`, `R²=0.94` |
| Artefactos | `renfe_model.pkl`, gráfico de importancia SHAP |
| Modelo | Objeto sklearn/xgboost listo para cargar |
| Metadatos | `run_id`, `git_commit`, timestamp, usuario |

**Model Registry** (siguiente paso):

```python
# Registrar y promocionar el mejor run
mlflow.register_model("runs:/<run_id>/model", "renfe-price-optimizer")
# Luego, desde la UI: promocionar a "Staging" o "Production"
```

#### 5.6.1. Modelo final de corredores bidireccionales y resultados

La versión final del pipeline utiliza un modelo **XGBoost de regresión tabular supervisada**. El dataset limpio incorpora **corredores bidireccionales** entre cinco ciudades, en los que cada una puede actuar como origen y como destino:

- Madrid;
- Barcelona;
- Sevilla;
- Valencia;
- Málaga.

Las variables `origen` y `destino` se incorporan como features categóricas junto con el tipo de tren, la clase, la tarifa, el día de la semana y la temporada. Las variables numéricas incluyen duración, días de antelación, hora de salida y mes.

El proceso de feature engineering generó **13.025.112 registros válidos** para el entrenamiento.

| Métrica | Resultado | Objetivo técnico | Cumplimiento |
|---|---:|---:|:---:|
| MAE | 4,07 € | — | ✅ |
| RMSE | 6,17 € | — | ✅ |
| MAPE | 7,67 % | < 10 % | ✅ |
| R² | 0,94 | ≥ 0,80 | ✅ |

El modelo explica aproximadamente el 94 % de la variabilidad observada en el precio y mantiene un error porcentual absoluto medio inferior al 8 %. La incorporación de la variable `origen` —motivada al confirmarse la naturaleza bidireccional de los corredores— mejoró todas las métricas respecto a la versión anterior, que asumía Madrid como origen único (MAE 4,42 € → 4,07 €; RMSE 6,73 € → 6,17 €; MAPE 8,16 % → 7,67 %; R² 0,931 → 0,942).

Los indicadores de ahorro real, adopción, conversión o porcentaje de compras realizadas en la ventana óptima se consideran **KPIs de negocio futuros**. No se presentan como resultados medidos porque el TFM utiliza datos históricos y no dispone todavía de usuarios reales en producción.

#### 5.6.2. Interpretabilidad del modelo mediante SHAP

El análisis SHAP proporciona una explicación consistente del modelo XGBoost.

La tarifa es la variable con mayor influencia global, seguida del tipo de tren, la
duración y, de forma muy próxima entre sí, el destino y el origen. Estas variables
representan las características comerciales, operativas y de corredor más utilizadas
por el modelo para distinguir niveles de precio.

La incorporación del origen como variable —motivada por la naturaleza bidireccional
de los corredores— resultó decisiva: alcanza una importancia prácticamente
equivalente a la del destino y su inclusión mejoró todas las métricas del modelo.
Ello demuestra que el precio depende del par origen-destino completo y no
únicamente de la ciudad de llegada.

La tarifa Flexible, el servicio AVE y la clase Preferente presentan generalmente
contribuciones asociadas a predicciones superiores. La clase Turista se asocia
principalmente a contribuciones negativas.

La duración presenta un comportamiento no lineal y debe interpretarse junto con
la ruta y el tipo de servicio. Las variables temporales tienen una importancia
global inferior, aunque continúan interviniendo en predicciones concretas.

Los días de antelación presentan una importancia global moderada, pero son
esenciales para construir la curva de precios y generar la recomendación de
comprar o esperar.

Finalmente, las explicaciones globales y locales mejoran la trazabilidad del
modelo, permiten justificar sus predicciones y reducen la opacidad del sistema.
Los resultados deben interpretarse como contribuciones predictivas, no como
relaciones causales. El detalle completo del análisis se recoge en la sección 13
de la memoria y en `reports/shap/`.

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

> **Nota sobre el sample de CI:** la muestra `renfe_clean_sample.csv` se regeneró para incluir explícitamente trayectos con Málaga (origen y destino), de forma que el smoke test valide también el corredor incorporado y no solo los originales.

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

**Ejemplo de request:**

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "origen": "MÁLAGA",
       "destino": "SEVILLA",
       "vehicle_type": "AVE",
       "vehicle_class": "Turista",
       "fare": "Promo",
       "duration": 2.70,
       "dias_anticipacion": 30,
       "hora_salida": 9,
       "dia_semana": "Viernes",
       "mes": 7,
       "temporada": "Verano"
     }'
```

**Estructura de la respuesta:**

```json
{
  "precio_estimado_hoy": "<precio en euros>",
  "precio_minimo_esperado": "<precio en euros>",
  "antelacion_optima_dias": "<número de días>",
  "ahorro_estimado_eur": "<ahorro en euros>",
  "ahorro_estimado_pct": "<ahorro porcentual>",
  "recomendacion": "COMPRA HOY o ESPERA X días",
  "curva": [
    { "dias_anticipacion": "<días>",
      "precio_estimado": "<precio>"
    }
  ]
}
```

> El esquema Pydantic valida que `origen` y `destino` pertenezcan a las cinco ciudades permitidas y que no coincidan entre sí (un trayecto no puede tener el mismo origen y destino).

### 6.2. Docker — empaquetado y despliegue

`Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY models/ models/
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
- **Métricas de negocio** (objetivo a futuro, NO medible con datos de 2019): ¿los usuarios que siguen la recomendación ahorran realmente ese 15 %? Requiere usuarios reales y no se evalúa en este TFM.

### 7.3. Deriva de datos (data drift)

El modelo entrenado sobre datos 2019 se degrada porque el mercado post-liberalización (Ouigo, Iryo desde 2021-2022) cambia la política tarifaria de Renfe. Detección propuesta:

- **PSI** (Population Stability Index) por variable.
- **KS-test** sobre la distribución de `price` y `dias_anticipacion`.
- Si drift > umbral → alarma + trigger de reentrenamiento.

---

## 8. Conclusiones y roadmap

Este trabajo demuestra cómo transformar un notebook académico en un sistema MLOps reproducible. El caso de uso final cubre **corredores bidireccionales** entre Madrid, Barcelona, Sevilla, Valencia y Málaga —cada ciudad puede ser origen y destino— y utiliza XGBoost como modelo de regresión tabular supervisada. El pipeline procesa 13.025.112 registros y obtiene un MAE de 4,07 €, un RMSE de 6,17 €, un MAPE de 7,67 % y un R² de 0,94. La incorporación de la variable `origen`, al confirmarse la bidireccionalidad de los corredores, mejoró todas las métricas respecto a la versión inicial.

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
3. **A/B testing** entre modelos (Random Forest vs XGBoost vs regresión lineal) sobre tráfico real.
4. **Ampliar el dataset** a datos post-liberalización con Ouigo e Iryo.
5. **Ampliar la cobertura de corredores** a nuevas ciudades, reutilizando el patrón ya validado con Málaga.
6. **Migrar el tracking** a Azure ML o Databricks cuando el volumen lo justifique.
7. **Evolucionar la interfaz** hacia un agente conversacional (Copilot Studio + LLM).

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

# Arquitectura & ADRs — Renfe Price Optimizer

Este documento recoge las **decisiones de arquitectura** (ADR = Architectural Decision Record) del proyecto. Se versiona con Git para que quede trazable **por qué** se eligió cada pieza.

---

## ADR-001 · Fuente de datos: Kaggle en lugar de scraping propio

**Estado**: aceptado.
**Contexto**: el anteproyecto contemplaba scrapear Renfe, Ouigo e Iryo durante ~3 meses.
**Decisión**: usar el dataset público de Kaggle *Spanish Rail Tickets Pricing – Renfe* (abril–agosto 2019).
**Consecuencias**: cobertura limitada a Renfe pre-liberalización, pero longitudinalidad garantizada; el foco pasa a la arquitectura y al modelado.

---

## ADR-002 · Repositorio y control de versiones

**Estado**: aceptado.
**Decisión**: **Git + GitHub**, con protección de `main` y flujo por Pull Request.
**Alternativas descartadas**: GitLab (menos integración con Actions gratuitas), Bitbucket.

---

## ADR-003 · Versionado de datos y modelos

**Estado**: aceptado.
**Decisión**: **DVC + remoto Google Drive** para datos y modelos pesados.
**Motivo**: Git no está preparado para binarios grandes; DVC guarda punteros pequeños en Git y sincroniza el contenido con Drive.

---

## ADR-004 · Tracking de experimentos

**Estado**: aceptado.
**Decisión**: **MLflow open source**, tracking local en `./mlruns`.
**Alternativas**: Weights & Biases (requiere cuenta cloud), Neptune.ai.
**Consecuencia**: portabilidad total; el tribunal puede lanzar `mlflow ui` y ver todo.

---

## ADR-005 · Servicio de predicción

**Estado**: aceptado.
**Decisión**: **FastAPI** para la API, **Streamlit** para la UI.
**Motivo**: FastAPI genera docs automáticas, valida con Pydantic y rinde muy bien; Streamlit permite una demo funcional al tribunal sin JS.

---

## ADR-006 · Empaquetado y despliegue

**Estado**: aceptado.
**Decisión**: **Docker + docker-compose** con tres servicios (api, streamlit, mlflow).
**Motivo**: un solo `docker compose up --build` reproduce todo el sistema.

---

## ADR-007 · CI/CD

**Estado**: aceptado.
**Decisión**: **GitHub Actions** con job único (`quality`) que ejecuta ruff, pytest y smoke train.
**Alternativas descartadas**: Jenkins (excesivo), CircleCI (menor integración).

---

## ADR-008 · Entorno Python

**Estado**: aceptado.
**Decisión**: **uv + pyproject.toml**, con `requirements.txt` de respaldo.
**Motivo**: uv resuelve dependencias en segundos, lockfile determinista, sustituye pip+venv.

---

## ADR-009 · Elección de modelo baseline

**Estado**: aceptado.
**Decisión**: **XGBoost** como modelo por defecto, con `baseline` (linear) y `random_forest` disponibles vía `params.yaml`.
**Motivo**: la literatura previa (Bandyopadhyay 2024, Aditi 2020) coincide en que ML tabular rinde mejor que series clásicas en este problema.

---

## Diagrama de contexto (C4 nivel 1)

```
                   ┌───────────────────────────────────────┐
                   │        Sistema: Renfe Price           │
                   │             Optimizer                 │
                   └───────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐            ┌──────────────┐          ┌──────────────┐
│ Viajero final │            │ Broker/Agreg.│          │  Data team   │
│ (usa UI web)  │            │  (usa API)   │          │ (mantiene ML)│
└───────────────┘            └──────────────┘          └──────────────┘
        │                           │                           │
        ▼                           ▼                           ▼
   Streamlit UI                 FastAPI /predict          MLflow UI · DVC
```

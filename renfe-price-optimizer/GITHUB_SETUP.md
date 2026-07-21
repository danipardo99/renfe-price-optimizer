# 🚀 Guía paso a paso — Del ZIP al repo en GitHub (VS Code + WSL2)

**TFM MLOps · Renfe Price Optimizer**
**Objetivo**: crear el repo en GitHub desde cero y dejarlo funcionando en tu VS Code con el pipeline MLflow activo.

**Tiempo estimado**: 25–40 min si es la primera vez, 10 min si ya tienes Git y VS Code configurados.

---

## PARTE 0 · Requisitos previos (una vez en la vida)

### 0.1. Cuenta de GitHub

Si no la tienes: <https://github.com/signup>. Usa tu correo académico o personal.

### 0.2. Instalar Git

**Windows:**
```powershell
winget install --id Git.Git -e --source winget
```

Comprueba:
```bash
git --version
```

### 0.3. (Recomendado) Instalar WSL2 + Ubuntu

Abre **PowerShell como administrador**:
```powershell
wsl --install
```
Reinicia. Al arrancar Ubuntu por primera vez, crea usuario y contraseña.

Dentro de Ubuntu:
```bash
sudo apt update && sudo apt install git curl -y
```

### 0.4. Instalar VS Code + extensiones

Descarga: <https://code.visualstudio.com/>

Extensiones obligatorias (Ctrl+Shift+X):
- **Python** (Microsoft)
- **Pylance**
- **Jupyter**
- **GitLens**
- **Docker**
- **Ruff**
- **GitHub Pull Requests**
- **Remote - WSL** *(si usas WSL)*

### 0.5. (Recomendado) Instalar uv

Gestor rápido de entornos Python. Dentro de Ubuntu (o Windows):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Recarga la terminal o:
source $HOME/.local/bin/env
uv --version
```

### 0.6. Configurar tu identidad Git

```bash
git config --global user.name "Daniel Pardo"
git config --global user.email "tu.correo@dominio.com"
git config --global init.defaultBranch main
```

---

## PARTE 1 · Crear el repositorio en GitHub

1. Abre <https://github.com/new>.
2. Rellena:
   - **Repository name**: `renfe-price-optimizer`
   - **Description**: *TFM MLOps — Predicción precios Renfe Madrid-Barcelona*
   - **Public** (para poder mostrarlo al tribunal)
   - **NO** marques *Add README* ni *Add .gitignore* (los traemos del ZIP)
   - **License**: MIT (o la que prefieras)
3. Click en **Create repository**.
4. GitHub te llevará a una pantalla con instrucciones. **No cierres esa pestaña**: la URL (`https://github.com/<usuario>/renfe-price-optimizer.git`) la usaremos en el paso 3.

---

## PARTE 2 · Descomprimir el ZIP y abrirlo en VS Code

1. Descarga el ZIP entregado con este trabajo (`renfe-price-optimizer.zip`).
2. Descomprímelo en una carpeta cómoda, por ejemplo:
   - **Windows/WSL**: `\\wsl$\Ubuntu\home\<tu-usuario>\proyectos\renfe-price-optimizer`
   - **Windows nativo**: `C:\dev\renfe-price-optimizer`
   - **macOS/Linux**: `~/proyectos/renfe-price-optimizer`

3. Abre VS Code y usa **File → Open Folder** para abrir la carpeta descomprimida.

   Si usas WSL: `Ctrl+Shift+P` → **WSL: Reopen Folder in WSL**.

4. Abre la terminal integrada: `Ctrl+ñ` (o *Terminal → New Terminal*).

Comprueba que estás dentro del proyecto:
```bash
pwd            # debe terminar en /renfe-price-optimizer
ls             # deberías ver README.md, pyproject.toml, src/, tests/…
```

---

## PARTE 3 · Inicializar Git y hacer el primer push

Desde la terminal de VS Code, dentro del proyecto:

```bash
# 1. Inicializar el repositorio local
git init
git branch -M main

# 2. Añadir todos los archivos y hacer el primer commit
git add .
git status                                     # revisa que .env no aparezca
git commit -m "chore: bootstrap del proyecto Renfe Price Optimizer"

# 3. Conectar con GitHub (usa la URL de la Parte 1)
git remote add origin https://github.com/<TU-USUARIO>/renfe-price-optimizer.git

# 4. Subir por primera vez
git push -u origin main
```

**Si te pide autenticarte**: usa un **Personal Access Token** de GitHub (no la contraseña de la web).

> 🛡️ Cómo crear un token: GitHub → *Settings* (foto perfil) → *Developer settings* → *Personal access tokens* → *Tokens (classic)* → *Generate new token (classic)* → marca `repo` → copia el token y úsalo como contraseña cuando Git te la pida.

Ve a tu repo en GitHub y verás **todos los archivos**. Enhorabuena: tu proyecto ya vive en la nube.

---

## PARTE 4 · Proteger `main` y activar el flujo de PRs

En GitHub, entra a tu repo → **Settings → Branches → Add branch protection rule**:

- **Branch name pattern**: `main`
- ✅ Require a pull request before merging
- ✅ Require status checks to pass before merging → selecciona `renfe-optimizer-ci`
- Save changes.

A partir de ahora, nadie (ni tú) podrá hacer `git push origin main` directamente. Todo cambio va por rama + PR.

---

## PARTE 5 · Crear el entorno Python y ejecutar el pipeline

Desde la terminal de VS Code:

```bash
# 1. Crear entorno con uv (o con venv + pip si prefieres)
uv sync

# 2. Copiar el .env
cp .env.example .env

# 3. Verificar tests
uv run pytest -v

# 4. Generar dataset procesado
uv run python -m renfe_optimizer.features

# 5. Entrenar el modelo (registra el run en MLflow)
uv run python -m renfe_optimizer.train --config params.yaml

# 6. Ver los experimentos en el navegador
uv run mlflow ui
# Abre http://localhost:5000
```

Si prefieres pip clásico:
```bash
python -m venv .venv
source .venv/bin/activate            # Linux/WSL/macOS
# .venv\Scripts\activate             # Windows PowerShell
pip install -r requirements.txt
pytest -v
python -m renfe_optimizer.features
python -m renfe_optimizer.train --config params.yaml
mlflow ui
```

---

## PARTE 6 · Levantar la API + Streamlit

En dos terminales distintas de VS Code:

**Terminal 1** — API:
```bash
uv run uvicorn renfe_optimizer.api.main:app --reload
# Abre http://localhost:8000/docs
```

**Terminal 2** — UI:
```bash
uv run streamlit run streamlit_app/app.py
# Abre http://localhost:8501
```

O todo con Docker en un solo comando:
```bash
docker compose up --build
# API:       http://localhost:8000/docs
# Streamlit: http://localhost:8501
# MLflow:    http://localhost:5000
```

---

## PARTE 7 · Flujo diario de trabajo (bloques 2 y 3)

Ejemplo: añadir la variable *es_festivo_nacional*.

```bash
# 1. Actualizar main
git checkout main
git pull origin main

# 2. Crear rama
git checkout -b feature/festivo-nacional

# 3. Editar src/renfe_optimizer/features.py y tests/test_features.py

# 4. Ejecutar tests locales
uv run ruff check .
uv run pytest -v

# 5. Commit y push
git add .
git commit -m "feat(features): añade indicador de festivo nacional"
git push origin feature/festivo-nacional

# 6. En GitHub → Compare & pull request → Create pull request
# 7. Espera a que CI pase en verde → Merge → Delete branch
```

---

## PARTE 8 · Comprobar que GitHub Actions funciona

Al hacer el primer push del ZIP, GitHub Actions habrá lanzado el workflow automáticamente.

En tu repo → pestaña **Actions**. Verás algo así:

- 🟡 En curso → espera 1–2 min.
- 🟢 Verde → todo correcto. La rama es mergeable.
- 🔴 Rojo → abre el log del job y busca la primera línea con `Error:`. Corrige, haz commit y vuelve a push.

---

## PARTE 9 · (Opcional) Configurar DVC con Google Drive

Solo cuando el dataset completo pese > 100 MB y no quepa en Git:

```bash
uv add --dev dvc dvc-gdrive
dvc init
git add .dvc .dvcignore && git commit -m "chore: inicializa DVC"

# Empezar a versionar el CSV grande
dvc add data/raw/renfe_2019_full.csv
git add data/raw/renfe_2019_full.csv.dvc .gitignore
git commit -m "data: versiona dataset Renfe 2019 con DVC"

# Crear carpeta en Google Drive y copiar su ID
dvc remote add --default drive gdrive://<ID-DE-LA-CARPETA>
git add .dvc/config && git commit -m "chore: configura remoto DVC en Google Drive"

dvc push        # sube el dato pesado a Drive
```

Un compañero luego solo tendrá que hacer:
```bash
dvc pull
```

---

## PARTE 10 · Checklist final antes de defender el TFM

- [ ] El repo está en GitHub, público, con README claro.
- [ ] `main` está protegido y solo se mergea por PR.
- [ ] El último run de GitHub Actions está en verde.
- [ ] `uv sync && pytest && python -m renfe_optimizer.train` funciona de punta a punta.
- [ ] MLflow UI muestra al menos 3 runs (baseline, RF, XGBoost).
- [ ] FastAPI responde en `/docs` y `/predict`.
- [ ] Streamlit muestra una recomendación con datos reales.
- [ ] `docker compose up` levanta todo el stack.
- [ ] `.env` NO está en Git; `.env.example` sí.
- [ ] MEMORIA_MLOPS.md está commiteada y actualizada.

---

## 🆘 Problemas típicos

| Error | Causa | Solución |
|---|---|---|
| `Permission denied (publickey)` al push | SSH sin configurar | Usa HTTPS + Personal Access Token, o configura SSH: `ssh-keygen -t ed25519` y añade la clave pública a GitHub |
| `Please tell me who you are` | Falta `git config` | Ejecuta los `git config --global user.name/email` de la Parte 0.6 |
| `remote origin already exists` | Ya añadiste el remote antes | `git remote set-url origin <url>` |
| `main → main (fetch first)` | GitHub creó un README y tu local no | `git pull origin main --allow-unrelated-histories` |
| `ModuleNotFoundError: renfe_optimizer` | El entorno no está activo | `uv sync` o `source .venv/bin/activate` |
| `ruff check` falla en CI pero no en local | Versiones distintas | Fija la versión en `requirements.txt` |
| Streamlit no encuentra la API | Puerto ocupado o CORS | Comprueba que uvicorn está corriendo en 8000 |

---

**¡Listo!** Con esto tienes el repo montado, el pipeline funcionando y todo listo para defender.

*Autores: Alejandro Montenegro · Laura Nieto · Daniel Pardo · Julio 2026*

# Manual de Instalación

**Proyecto:** Predicción de Enfermedades Cardiovasculares mediante Aprendizaje Automático  
**Autor:** Jhandry Santiago Chimbo Rivera  
**Versión:** 1.0 — JULIO 2026

---

## Tabla de Contenidos

1. [Requisitos del Sistema](#1-requisitos-del-sistema)
2. [Herramientas Previas Necesarias](#2-herramientas-previas-necesarias)
3. [Obtener el Código Fuente](#3-obtener-el-código-fuente)
4. [Crear y Activar el Entorno Virtual](#4-crear-y-activar-el-entorno-virtual)
5. [Instalar Dependencias](#5-instalar-dependencias)
6. [Obtener los Datos de Entrada](#6-obtener-los-datos-de-entrada)
7. [Verificación de la Instalación](#7-verificación-de-la-instalación)
8. [Solución de Problemas Comunes](#8-solución-de-problemas-comunes)

---

## 1. Requisitos del Sistema

| Componente | Versión mínima | Recomendada |
|------------|---------------|-------------|
| Sistema Operativo | Windows 10 / Ubuntu 20.04 / macOS 11 | Windows 11 / Ubuntu 22.04 |
| Python | 3.10 | **3.12** |
| RAM | 8 GB | 16 GB |
| Espacio en disco | 2 GB libres | 5 GB libres |
| Procesador | x86-64 | x86-64 multi-núcleo |

> [!NOTE]
> Se recomienda usar Python 3.12 para compatibilidad total con las librerías del proyecto (`xgboost`, `lightgbm`, `optuna`).

---

## 2. Herramientas Previas Necesarias

Antes de continuar, asegúrate de tener instalados los siguientes programas:

### 2.1 Python 3.12

Descarga el instalador oficial desde:  
**https://www.python.org/downloads/**

**Windows:** Durante la instalación, marca la casilla **"Add Python to PATH"**.

Verifica la instalación:

```bash
python --version
# Esperado: Python 3.12.x
```

### 2.2 Git

Descarga e instala Git desde:  
**https://git-scm.com/downloads**

Verifica la instalación:

```bash
git --version
# Esperado: git version 2.x.x
```

### 2.3 DVC (Data Version Control)

DVC se instalará automáticamente junto con las dependencias del proyecto (está incluido en `requirements.txt`). Sin embargo, también puedes instalarlo de forma independiente:

```bash
pip install dvc
```

Verifica la instalación:

```bash
dvc --version
# Esperado: 3.x.x
```

---

## 3. Obtener el Código Fuente

### Opción A — Clonar con Git (recomendado)

```bash
git clone https://github.com/JhandryChimbo/Prediccion_Cardiovascular.git
cd Prediccion_Cardiovascular
```

### Opción B — Descargar como ZIP

1. Ve al repositorio en GitHub.
2. Haz clic en **Code → Download ZIP**.
3. Extrae el archivo ZIP en la carpeta de tu elección.
4. Abre una terminal y navega hasta la carpeta extraída:

```bash
cd ruta/a/Prediccion_Cardiovascular
```

---

## 4. Crear y Activar el Entorno Virtual

Se recomienda usar un entorno virtual para aislar las dependencias del proyecto.

### Windows (PowerShell)

```powershell
# Crear el entorno virtual
python -m venv env

# Activar el entorno virtual
.\env\Scripts\Activate.ps1
```

> [!IMPORTANT]
> Si PowerShell muestra el error *"la ejecución de scripts está deshabilitada"*, ejecuta primero:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Windows (CMD)

```cmd
python -m venv env
env\Scripts\activate.bat
```

### Linux / macOS

```bash
python3 -m venv env
source env/bin/activate
```

Cuando el entorno esté activo, verás el prefijo `(env)` al inicio de tu línea de comandos:

```
(env) C:\Prediccion_Cardiovascular>
```

---

## 5. Instalar Dependencias

Con el entorno virtual activado, actualiza `pip` e instala todas las librerías del proyecto:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Librerías que se instalarán

| Librería | Descripción |
|----------|-------------|
| `pandas` | Manipulación y análisis de datos |
| `numpy` | Cómputo numérico |
| `scikit-learn` | Algoritmos ML y herramientas de preprocesamiento |
| `xgboost` | Algoritmo XGBoost |
| `lightgbm` | Algoritmo LightGBM |
| `imbalanced-learn` | Técnicas de resampling (SMOTE, undersampling) |
| `optuna` | Optimización bayesiana de hiperparámetros |
| `matplotlib` | Visualización de gráficos |
| `seaborn` | Visualización estadística |
| `jupyter` | Entorno de Notebooks interactivos |
| `dvc` | Control de versiones de datos y pipelines |
| `mlflow` | Tracking de experimentos (flujo alternativo) |

> [!NOTE]
> La instalación puede tomar entre 5 y 15 minutos según la velocidad de tu conexión a internet.

---

## 6. Obtener los Datos de Entrada

El dataset de entrada (`cardio_train.csv`) no está almacenado directamente en Git, sino gestionado por **DVC**.

### Opción A — Mediante DVC Pull (si tienes acceso al remoto configurado)

```bash
dvc pull
```

Esto descarga automáticamente el archivo `data/raw/cardio_train.csv`.

### Opción B — Descarga manual desde Kaggle

Si no tienes acceso al remoto DVC, descarga el dataset directamente:

1. Ve a: **https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset**
2. Descarga el archivo `cardio_train.csv`.
3. Colócalo manualmente en la carpeta:

```
data/raw/cardio_train.csv
```

> [!IMPORTANT]
> El archivo debe llamarse exactamente `cardio_train.csv` y estar ubicado en `data/raw/`. De lo contrario, el pipeline no podrá encontrarlo.

---

## 7. Verificación de la Instalación

Una vez completados los pasos anteriores, verifica que todo esté correctamente instalado:

```bash
# Verificar Python y librerías clave
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, optuna; print('OK — Todas las librerías instaladas correctamente')"

# Verificar DVC
dvc status

# Verificar que el dataset esté disponible
python -c "import pandas as pd; df = pd.read_csv('data/raw/cardio_train.csv', sep=';'); print(f'Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas')"
```

Salida esperada:

```
OK — Todas las librerías instaladas correctamente
Data and pipelines are up to date.
Dataset cargado: 70000 filas, 13 columnas
```

---

## 8. Solución de Problemas Comunes

### Error: `python` no encontrado (Windows)

**Causa:** Python no fue agregado al PATH durante la instalación.  
**Solución:** Reinstala Python y marca la casilla **"Add Python to PATH"**, o agrega la ruta manualmente en las variables de entorno de Windows.

---

### Error: `pip install` falla con `lightgbm` o `xgboost`

**Causa:** Puede faltar Microsoft Visual C++ Build Tools (Windows).  
**Solución:** Descarga e instala desde:  
**https://visualstudio.microsoft.com/visual-cpp-build-tools/**

---

### Error: `dvc pull` falla por falta de acceso al remoto

**Causa:** No tienes credenciales configuradas para el almacenamiento remoto DVC.  
**Solución:** Usa la **Opción B** para obtener el dataset directamente desde Kaggle (ver sección 6).

---

### Error: `.\env\Scripts\Activate.ps1` no se puede cargar (PowerShell)

**Causa:** La política de ejecución de PowerShell está restringida.  
**Solución:**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### Notebooks de Jupyter no abren correctamente

**Causa:** Jupyter no está instalado o el kernel no está registrado.  
**Solución:**

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=env --display-name "Python (TIC-2026)"
```

---

> [!TIP]
> Una vez completada la instalación exitosamente, consulta el **Manual de Ejecución** (`MANUAL_EJECUCION.md`) para aprender a correr el pipeline y reproducir los experimentos.

---

**Última actualización:** Mayo 2026  
**Versión del documento:** 1.0

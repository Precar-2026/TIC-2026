# Requisitos del Software

**Proyecto:** Predicción de Enfermedades Cardiovasculares mediante Aprendizaje Automático  
**Autor:** Jhandry Santiago Chimbo Rivera  
**Versión:** 1.0 — Julio 2026

---

## Tabla de Contenidos

1. [Introducción](#1-introducción)
2. [Requisitos del Sistema Operativo](#2-requisitos-del-sistema-operativo)
3. [Requisitos de Hardware](#3-requisitos-de-hardware)
4. [Requisitos de Software Base](#4-requisitos-de-software-base)
5. [Dependencias de Python](#5-dependencias-de-python)
6. [Requisitos de Datos](#6-requisitos-de-datos)
7. [Requisitos de Red y Conectividad](#7-requisitos-de-red-y-conectividad)
8. [Requisitos Funcionales](#8-requisitos-funcionales)
9. [Requisitos No Funcionales](#9-requisitos-no-funcionales)
10. [Restricciones y Supuestos](#10-restricciones-y-supuestos)

---

## 1. Introducción

Este documento especifica los requisitos de software necesarios para instalar, configurar y ejecutar el sistema de predicción de riesgo cardiovascular. El sistema implementa un pipeline de aprendizaje automático supervisado que procesa datos clínicos, entrena múltiples modelos y genera artefactos reproducibles.

### Alcance

El sistema abarca:

- Preprocesamiento y limpieza de datos clínicos.
- Ingeniería y selección de características.
- Entrenamiento y optimización de cuatro algoritmos de clasificación.
- Evaluación y comparación experimental de modelos.
- Visualización de resultados mediante Notebooks interactivos.

### Convenciones de prioridad

| Etiqueta | Significado |
|----------|-------------|
| **OBLIGATORIO** | El sistema no funciona sin este requisito. |
| **RECOMENDADO** | Mejora el rendimiento o la experiencia, pero no es indispensable. |
| **OPCIONAL** | Habilita funcionalidades adicionales o alternativas. |

---

## 2. Requisitos del Sistema Operativo

| Sistema Operativo | Versión mínima | Prioridad |
|-------------------|---------------|-----------|
| Windows | 10 (64 bits) | **OBLIGATORIO** (una de las tres opciones) |
| Ubuntu / Debian | 20.04 LTS | **OBLIGATORIO** (una de las tres opciones) |
| macOS | 11 Big Sur | **OBLIGATORIO** (una de las tres opciones) |

> [!NOTE]
> El proyecto fue desarrollado y probado principalmente en **Windows 11 (64 bits)**. Se garantiza compatibilidad con Linux y macOS, pero las rutas de comandos pueden variar (ver `MANUAL_INSTALACION.md`).

> [!IMPORTANT]
> Solo se soportan sistemas operativos de **64 bits**. Los sistemas de 32 bits son incompatibles con librerías como `xgboost` y `lightgbm`.

---

## 3. Requisitos de Hardware

### Mínimos (ejecución básica del pipeline)

| Componente | Especificación mínima |
|------------|----------------------|
| Procesador | x86-64, 2 núcleos, 2.0 GHz |
| Memoria RAM | 8 GB |
| Almacenamiento | 3 GB libres en disco |
| Tipo de almacenamiento | HDD o SSD |

### Recomendados (ejecución con `--trials 50` o superior)

| Componente | Especificación recomendada |
|------------|---------------------------|
| Procesador | x86-64, 4+ núcleos, 2.5 GHz o superior |
| Memoria RAM | 16 GB |
| Almacenamiento | 5 GB libres en SSD |
| Tipo de almacenamiento | SSD (reduce tiempos de I/O del pipeline) |

> [!NOTE]
> La optimización bayesiana con Optuna (`--trials 50`) y la validación cruzada estratificada de 5 folds son las etapas más demandantes en CPU y RAM. Con 8 GB de RAM y un procesador de 4 núcleos, el pipeline completo tarda aproximadamente entre 1 y 3 horas.

---

## 4. Requisitos de Software Base

Los siguientes programas deben estar instalados en el sistema antes de configurar el proyecto.

### 4.1 Python

| Atributo | Valor |
|----------|-------|
| Versión mínima | 3.10 |
| Versión recomendada | **3.12** |
| Arquitectura | 64 bits |
| Prioridad | **OBLIGATORIO** |
| Descarga | https://www.python.org/downloads/ |

> [!IMPORTANT]
> Python debe estar agregado al `PATH` del sistema. En Windows, marcar la casilla **"Add Python to PATH"** durante la instalación.

---

### 4.2 Git

| Atributo | Valor |
|----------|-------|
| Versión mínima | 2.30 |
| Prioridad | **OBLIGATORIO** para clonar el repositorio |
| Descarga | https://git-scm.com/downloads |

---

### 4.3 pip (gestor de paquetes de Python)

| Atributo | Valor |
|----------|-------|
| Versión mínima | 23.0 |
| Prioridad | **OBLIGATORIO** |
| Actualización | `pip install --upgrade pip` |

---

### 4.4 DVC (Data Version Control)

| Atributo | Valor |
|----------|-------|
| Versión mínima | 3.0 |
| Prioridad | **OBLIGATORIO** para reproducir el pipeline completo |
| Instalación | Incluido en `requirements.txt` o `pip install dvc` |
| Sitio oficial | https://dvc.org/ |

---

### 4.5 Microsoft Visual C++ Build Tools *(solo Windows)*

| Atributo | Valor |
|----------|-------|
| Versión mínima | Visual Studio Build Tools 2019 |
| Prioridad | **RECOMENDADO** en Windows |
| Descarga | https://visualstudio.microsoft.com/visual-cpp-build-tools/ |
| Motivo | Requerido por algunas versiones de `lightgbm` y `xgboost` en Windows |

---

### 4.6 Jupyter Notebook / JupyterLab

| Atributo | Valor |
|----------|-------|
| Prioridad | **OBLIGATORIO** para ejecutar los notebooks de análisis |
| Instalación | Incluido en `requirements.txt` (`jupyter`) |
| Versión mínima | Notebook 6.x o JupyterLab 3.x |

---

## 5. Dependencias de Python

Todas las librerías se instalan automáticamente mediante:

```bash
pip install -r requirements.txt
```

### Tabla completa de dependencias

| Librería | Versión mínima | Prioridad | Función en el proyecto |
|----------|---------------|-----------|------------------------|
| `pandas` | 1.5 | **OBLIGATORIO** | Manipulación y análisis de datos tabulares |
| `numpy` | 1.23 | **OBLIGATORIO** | Cómputo numérico y operaciones matriciales |
| `scikit-learn` | 1.2 | **OBLIGATORIO** | Preprocesamiento, RFECV, validación cruzada, métricas y Regresión Logística |
| `xgboost` | 1.7 | **OBLIGATORIO** | Algoritmo XGBoost para clasificación |
| `lightgbm` | 3.3 | **OBLIGATORIO** | Algoritmo LightGBM para clasificación y RFECV |
| `imbalanced-learn` | 0.10 | **OBLIGATORIO** | SMOTE y técnicas de undersampling |
| `optuna` | 3.0 | **OBLIGATORIO** | Optimización bayesiana de hiperparámetros (TPE + MedianPruner) |
| `matplotlib` | 3.6 | **OBLIGATORIO** | Generación de gráficos y figuras |
| `seaborn` | 0.12 | **OBLIGATORIO** | Visualización estadística (heatmaps, distribuciones) |
| `jupyter` | 1.0 | **OBLIGATORIO** | Entorno de notebooks interactivos |
| `dvc` | 3.0 | **OBLIGATORIO** | Versionado de datos y orquestación del pipeline |
| `mlflow` | 2.0 | **OPCIONAL** | Tracking alternativo de experimentos |

> [!NOTE]
> Las versiones específicas no están fijadas en `requirements.txt` para permitir compatibilidad con las versiones más recientes. Si se requiere una reproducción exacta, se recomienda usar `pip freeze > requirements_lock.txt` después de la primera instalación exitosa.

---

## 6. Requisitos de Datos

### Dataset principal

| Atributo | Valor |
|----------|-------|
| Nombre | Cardiovascular Disease Dataset |
| Archivo | `cardio_train.csv` |
| Ubicación esperada | `data/raw/cardio_train.csv` |
| Tamaño aproximado | ~5 MB |
| Formato | CSV separado por punto y coma (`;`) |
| Registros | 70,000 filas |
| Columnas | 13 variables clínicas |
| Fuente | https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset |
| Prioridad | **OBLIGATORIO** |

### Variables del dataset

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `id` | int | Identificador único del paciente |
| `age` | int | Edad en días |
| `gender` | int | Género (1 = mujer, 2 = hombre) |
| `height` | int | Altura en centímetros |
| `weight` | float | Peso en kilogramos |
| `ap_hi` | int | Presión arterial sistólica (mmHg) |
| `ap_lo` | int | Presión arterial diastólica (mmHg) |
| `cholesterol` | int | Colesterol (1 = normal, 2 = sobre lo normal, 3 = muy sobre lo normal) |
| `gluc` | int | Glucosa (1 = normal, 2 = sobre lo normal, 3 = muy sobre lo normal) |
| `smoke` | int | Fumador (0 = no, 1 = sí) |
| `alco` | int | Consumo de alcohol (0 = no, 1 = sí) |
| `active` | int | Actividad física (0 = no, 1 = sí) |
| `cardio` | int | **Variable objetivo** — Enfermedad cardiovascular (0 = no, 1 = sí) |

---

## 7. Requisitos de Red y Conectividad

| Acción | Conectividad requerida | Prioridad |
|--------|----------------------|-----------|
| Clonar el repositorio con `git clone` | Acceso a internet | **OBLIGATORIO** (solo la primera vez) |
| Instalar dependencias con `pip install -r requirements.txt` | Acceso a internet | **OBLIGATORIO** (solo la primera vez) |
| Descargar datos con `dvc pull` | Acceso al remoto DVC configurado | **RECOMENDADO** |
| Descargar dataset manualmente desde Kaggle | Acceso a internet | **ALTERNATIVO** si no hay remoto DVC |
| Ejecución del pipeline (`dvc repro`, scripts Python) | **Sin conexión** | — |

> [!NOTE]
> Una vez instaladas todas las dependencias y disponible el dataset, el pipeline completo se ejecuta de forma completamente **offline**.

---

## 8. Requisitos Funcionales

Los siguientes requisitos describen las capacidades que el software debe proveer:

| ID | Requisito | Prioridad |
|----|-----------|-----------|
| RF-01 | El sistema debe limpiar el dataset eliminando registros con valores fisiológicamente inválidos usando rangos configurables. | **OBLIGATORIO** |
| RF-02 | El sistema debe generar variables derivadas clínicas (IMC, presión de pulso, presión arterial media, etc.). | **OBLIGATORIO** |
| RF-03 | El sistema debe realizar selección de características mediante RFECV usando LightGBM como estimador base. | **OBLIGATORIO** |
| RF-04 | El sistema debe entrenar y optimizar los modelos RF, LR, XGB y LGBM mediante búsqueda bayesiana con Optuna. | **OBLIGATORIO** |
| RF-05 | El sistema debe aplicar validación cruzada estratificada con semillas fijas para garantizar reproducibilidad. | **OBLIGATORIO** |
| RF-06 | El sistema debe ajustar el umbral de clasificación usando solo el conjunto de validación. | **OBLIGATORIO** |
| RF-07 | El sistema debe soportar técnicas de resampling: SMOTE, undersampling aleatorio o ninguno. | **OBLIGATORIO** |
| RF-08 | El sistema debe guardar todos los artefactos de cada experimento en carpetas numeradas (`ExperimentoN`). | **OBLIGATORIO** |
| RF-09 | El sistema debe generar métricas en validación y test: AUC-ROC, F1-Score, Precisión, Recall, MCC. | **OBLIGATORIO** |
| RF-10 | El sistema debe permitir la comparación de todos los experimentos mediante el notebook `05_comparativa_experimentos_tesis.ipynb`. | **OBLIGATORIO** |
| RF-11 | El sistema debe soportar la eliminación selectiva de características mediante el parámetro `--drop_features`. | **RECOMENDADO** |
| RF-12 | El sistema debe permitir filtrar el dataset solo a pacientes hipertensos (`--use_hypertensive_only`). | **RECOMENDADO** |
| RF-13 | El sistema debe generar figuras exportadas automáticamente en `report/figures/`. | **RECOMENDADO** |

---

## 9. Requisitos No Funcionales

| ID | Requisito | Categoría | Prioridad |
|----|-----------|-----------|-----------|
| RNF-01 | El pipeline completo debe ser reproducible: la misma configuración debe producir los mismos resultados. | Reproducibilidad | **OBLIGATORIO** |
| RNF-02 | No debe existir fuga de información entre los conjuntos de entrenamiento, validación y test. | Integridad | **OBLIGATORIO** |
| RNF-03 | El escalado (scaler) debe ajustarse únicamente sobre `X_train` y aplicarse sobre `X_val` y `X_test`. | Integridad | **OBLIGATORIO** |
| RNF-04 | Las semillas aleatorias deben estar fijas en todos los componentes del pipeline (split, CV, Optuna). | Reproducibilidad | **OBLIGATORIO** |
| RNF-05 | Los artefactos de cada experimento deben estar aislados para permitir comparaciones consistentes. | Trazabilidad | **OBLIGATORIO** |
| RNF-06 | El tiempo de ejecución del pipeline completo no debe superar las 4 horas en hardware recomendado. | Rendimiento | **RECOMENDADO** |
| RNF-07 | Los modelos entrenados deben ser serializables y reutilizables en formato `.joblib`. | Portabilidad | **OBLIGATORIO** |
| RNF-08 | El sistema debe registrar métricas de sobreajuste (gap train-validación) para cada experimento. | Trazabilidad | **RECOMENDADO** |
| RNF-09 | El código debe ejecutarse sin errores en Python 3.10, 3.11 y 3.12. | Compatibilidad | **RECOMENDADO** |

---

## 10. Restricciones y Supuestos

### Restricciones

- El proyecto **no incluye** una interfaz gráfica de usuario (GUI) ni una API de inferencia en producción. Su uso es exclusivamente académico e investigativo.
- El dataset `cardio_train.csv` **no está versionado directamente en Git**. Se gestiona con DVC.
- Los archivos generados en `data/processed/`, `models/` y `report/` **no se incluyen en el repositorio Git** (están en `.gitignore`).
- El umbral de clasificación **solo debe ajustarse usando el conjunto de validación**, nunca el de test.

### Supuestos

- El usuario tiene conocimientos básicos de Python y uso de la línea de comandos.
- El dataset original tiene la estructura y formato especificados en la sección 6.
- El entorno de ejecución tiene acceso a internet al menos durante la instalación inicial.
- Los valores de las semillas aleatorias utilizadas en el código (`random_state`) no serán modificados entre experimentos comparativos.

---

> [!TIP]
> Para comenzar con la instalación del entorno, consulta el archivo `MANUAL_INSTALACION.md`.  
> Para ejecutar el pipeline y los experimentos, consulta el archivo `MANUAL_EJECUCION.md`.

---

**Última actualización:** Julio 2026  
**Versión del documento:** 1.0

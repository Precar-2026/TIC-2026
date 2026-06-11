# Prediccion de Enfermedades Cardiovasculares mediante Aprendizaje Automatico

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-945DD6.svg)](https://dvc.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Optimization-3AAED8.svg)](https://optuna.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

---

## Resumen Ejecutivo

Este proyecto implementa un sistema de prediccion de riesgo cardiovascular con aprendizaje automatico supervisado, orientado a uso academico (tesis) y enfocado en reproducibilidad experimental.

El flujo cubre:

- Limpieza clinicamente guiada de datos (rangos fisiologicos configurables)
- Ingenieria de caracteristicas cardiovasculares con opciones de ablacion
- Optimizacion bayesiana con Optuna y validacion cruzada estratificada
- Generacion de artefactos por experimento para evaluacion robusta en test
- Orquestacion reproducible del pipeline con DVC

Dataset base: 70,000 registros de pacientes con variables demograficas, presion arterial, IMC, colesterol, glucosa y habitos de vida.

---
> [!IMPORTANT]
> ## Estado Actual (Mayo 2026)
>
> - Pipeline DVC activo con limpieza, feature engineering, RFECV y entrenamiento secuencial de `RF`, `LR`, `XGB`, `LGBM`.
> - Comparativa consolidada sobre test en `notebooks/05_comparativa_experimentos_tesis.ipynb`.
> - Se detectan y comparan 20 experimentos (`Experimento1` a `Experimento20`).
> - Ultimo corte comparativo (archivo `report/tables/comparativa_experimentos_test.csv`):
>   - Ganador por `AUC-ROC(test)`: `Experimento17 - XGBoost` (`AUC-ROC: 0.8059`, `F1: 0.7418`)
>   - Ganador por `F1(test)`: `Experimento9 - LightGBM` (`F1: 0.7436`, `AUC-ROC: 0.8044`)
> 
> Nota: Estos valores dependen del ultimo conjunto de artefactos locales disponible y pueden cambiar tras nuevas ejecuciones.

---

## Tabla de Contenidos

1. [Introduccion](#introduccion)
2. [Objetivos](#objetivos)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Metodologia](#metodologia)
5. [Pipeline de Datos y Entrenamiento](#pipeline-de-datos-y-entrenamiento)
6. [Notebooks y Reportes](#notebooks-y-reportes)
7. [Resultados y Experimentacion](#resultados-y-experimentacion)
8. [Configuracion del Entorno](#configuracion-del-entorno)
9. [Reproducibilidad](#reproducibilidad)
10. [Tecnologias y Herramientas](#tecnologias-y-herramientas)
11. [Autor y Contacto](#autor-y-contacto)
12. [Referencias](#referencias)

---

## Introduccion

Las enfermedades cardiovasculares (ECV) son una de las principales causas de mortalidad global. Este proyecto busca apoyar la deteccion temprana de riesgo mediante modelos predictivos que combinan rendimiento, trazabilidad y consistencia metodologica.

El enfoque integra:

- EDA para entender patrones y calidad del dato
- Feature engineering con criterio clinico (AHA/ACC)
- Optimizacion de hiperparametros con Optuna
- Evaluacion separada en validacion y test (sin fuga de informacion)
- Trazabilidad de experimentos por carpeta y artefactos versionables

---

## Objetivos

### Objetivo General

Evaluar el rendimiento medido a través de **F1-Score** y **AUC-ROC** de los algoritmos **Regresión Logística, Random Forest, XGBoost** y **LightGBM** para la detección de riesgo cardiovascular en pacientes hipertensos, utilizando el dataset Cardiovascular Disease.

### Objetivos Especificos

1. Preprocesar el conjunto de datos Cardiovascular Disease aplicando técnicas de limpieza e ingeniería de características para el entrenamiento de los modelos predictivos.
2. Entrenar los modelos predictivos basados en los algoritmos de **Regresión Logística, Random Forest, XGBoost** y **LightGBM** para la detección de riesgo cardiovascular en pacientes hipertensos, evaluando su desempeño mediante **F1-Score** y **AUC-ROC**.

---

## Estructura del Proyecto

```text
Prediccion_Cardiovascular/
|
|-- data/
|   |-- raw/
|   |   |-- cardio_train.csv.dvc
|   |-- processed/
|       |-- cardio_clean.csv
|       |-- model_inputs/
|       |-- model_inputs_rfecv/
|       |-- __tmp_full_feature_snapshot/
|
|-- src/
|   |-- data/
|   |   |-- make_dataset.py
|   |-- features/
|   |   |-- build_features.py
|   |   |-- select_features.py
|   |-- models/
|       |-- train_model.py
|
|-- models/
|   |-- best_model_LR.joblib
|   |-- best_model_RF.joblib
|   |-- best_model_XGB.joblib
|   |-- best_model_LGBM.joblib
|   |-- scaler.joblib
|   |-- Experimento1/
|   |-- ...
|   |-- Experimento18/
|
|-- notebooks/
|   |-- 01_eda_cardio.ipynb
|   |-- 02_featuring_cardio.ipynb
|   |-- 03_training_visualization.ipynb
|   |-- 04_evaluacion.ipynb
|   |-- 05_comparativa_experimentos_tesis.ipynb
|
|-- report/
|   |-- figures/
|   |-- tables/
|
|-- dvc.yaml
|-- requirements.txt
|-- README.md
```

### Archivos generados localmente (no versionados en Git)

- `env/`
- `data/raw/*.csv` (si se obtiene por DVC remoto)
- `data/processed/`
- `models/*.joblib` y `models/Experimento*/`

---

## Metodologia

### 1. Preprocesamiento de Datos

Script principal: `src/data/make_dataset.py`

Funciones clave:

- Eliminacion de columna `id`
- Eliminacion de duplicados
- Filtros fisiologicos configurables:
  - `altura`: min/max
  - `peso`: min/max
  - `ap_hi`: min/max
  - `ap_lo`: min/max
  - Consistencia de presion (`ap_hi > ap_lo`, o `>=` con flag)
- Variables derivadas iniciales:
  - `edad_años`
  - `imc`

Parametros utiles:

- `--altura_min`, `--altura_max`
- `--peso_min`, `--peso_max`
- `--ap_hi_min`, `--ap_hi_max`
- `--ap_lo_min`, `--ap_lo_max`
- `--allow_equal_bp`

### 2. Ingenieria de Caracteristicas

Script principal: `src/features/build_features.py`

Caracteristicas derivadas destacadas:

- `pulse_pressure`
- `map`
- `pressure_ratio`
- Categorias clinicas de IMC, edad e hipertension
- Interacciones (`imc_edad_interaction`, `presion_imc_interaction`, etc.)
- Transformaciones (`edad_squared`, `imc_squared`, `log_ap_hi`, `col_gluc_ratio`)

Importante:

- Se incorpora soporte de ablacion via `--drop_features`.
- Se puede desactivar FE avanzada con `--disable_advanced_features`.
- Se puede usar subpoblacion hipertensa con `--use_hypertensive_only`.
- Escalado configurable con `--scaler_type {robust,standard,none}`.

### 3. Seleccion de Caracteristicas (RFECV)

Script principal: `src/features/select_features.py`

Detalles clave:

- RFECV con `LightGBM` y `ROC-AUC` como métrica.
- Usa solo `X_train/y_train` para seleccionar variables y aplica el subset a `X_val/X_test`.
- Genera:
  - `data/processed/model_inputs_rfecv/`
  - `report/figures/rfecv_results.png`
  - `data/processed/model_inputs_rfecv/selected_features.pkl`

### 4. Modelado y Optimizacion

Script principal del pipeline: `src/models/train_model.py`

Modelos soportados:

- `LR` (Logistic Regression)
- `RF` (Random Forest)
- `XGB` (XGBoost)
- `LGBM` (LightGBM)

Estrategia de entrenamiento:

- Optuna (TPE) con poda (`MedianPruner`)
- CV estratificada (`--cv_folds`)
- Funcion objetivo compuesta:
  - Calidad en validacion (`ROC-AUC` + `F1`)
  - Penalizacion por sobreajuste (gap train-val)
- Ajuste de umbral sobre validacion (opcional)

Resampling soportado:

- `--resampling_method none|smote|undersample`
- Compatibilidad legacy con `--use_smote`
- Control de ratios:
  - `--smote_sampling_strategy`
  - `--undersample_sampling_strategy`

Control experimental:

- `--reset_experiment`
- `--no_threshold_tuning`
- `--drop_uncertain_cases`
- `--uncertainty_quantile`

### 5. Artefactos por Experimento

Cada `ExperimentoN` guarda:

- Modelos entrenados por algoritmo
- `training_artifacts/` con:
  - `optuna_trials_<modelo>.csv`
  - `val_metrics_<modelo>.csv`
  - `best_params_<modelo>.json`
  - `threshold_<modelo>.json`
  - `val_predictions_<modelo>.csv`
  - `overfitting_diagnostics_<modelo>.json`
  - `problematic_validation_samples_<modelo>.csv`
  - `model_inputs_snapshot/` (X_test/y_test por experimento)

El snapshot por experimento evita incompatibilidades al comparar corridas con distintos sets de features.

---

## Pipeline de Datos y Entrenamiento

### Ejecucion Completa (DVC)

```bash
dvc repro
```

El `dvc.yaml` actual ejecuta:

1. Limpieza (`make_dataset.py`) con filtros fisiologicos explicitos
2. Feature engineering (`build_features.py`) con:
   - `--scaler_type robust`
   - `--drop_features log_ap_hi,pressure_ratio,col_gluc_ratio`
3. Seleccion de caracteristicas (`select_features.py`) y salida a `model_inputs_rfecv`
4. Entrenamiento de `RF`, `LR`, `XGB`, `LGBM` con:
  - `--cv_folds 5`
  - `--trials 50`
  - `--resampling_method smote`

### Ejecucion Manual por Etapas

#### 1) Limpieza

```bash
python src/data/make_dataset.py \
  --input data/raw/cardio_train.csv \
  --output data/processed/cardio_clean.csv \
  --altura_min 130 --altura_max 220 \
  --peso_min 30 --peso_max 200 \
  --ap_hi_min 80 --ap_hi_max 240 \
  --ap_lo_min 50 --ap_lo_max 140
```

#### 2) Features

```bash
python src/features/build_features.py \
  --input data/processed/cardio_clean.csv \
  --output_dir data/processed/model_inputs \
  --scaler_path models/scaler.joblib \
  --scaler_type robust \
  --drop_features log_ap_hi,pressure_ratio,col_gluc_ratio
```

#### 3) Seleccion de caracteristicas (RFECV)

```bash
python src/features/select_features.py \
  --input_dir data/processed/model_inputs \
  --output_dir data/processed/model_inputs_rfecv
```

#### 4) Entrenamiento (ejemplo XGBoost)

```bash
python src/models/train_model.py \
  --input_dir data/processed/model_inputs_rfecv \
  --model XGB \
  --models_dir models \
  --cv_folds 5 \
  --trials 50 \
  --resampling_method smote \
  --smote_sampling_strategy 1.0
```

### Parametros principales de `train_model.py`

| Parametro | Descripcion | Default |
|-----------|-------------|---------|
| `--input_dir` | Directorio con `X_train/y_train/X_val/y_val` | - |
| `--model` | Tipo (`LR`, `RF`, `XGB`, `LGBM`) | - |
| `--models_dir` | Carpeta para modelos y experimentos | `models` |
| `--trials` | Numero de trials Optuna | `80` |
| `--cv_folds` | Folds de validacion cruzada | `5` |
| `--no_threshold_tuning` | Desactiva ajuste de umbral en validacion | `False` |
| `--reset_experiment` | Reinicia contador de `ExperimentoN` | `False` |
| `--drop_uncertain_cases` | Elimina casos inciertos de train y reentrena | `False` |
| `--uncertainty_quantile` | Cuantil de incertidumbre para limpieza | `0.10` |
| `--resampling_method` | `none`, `smote`, `undersample` | `none` |
| `--smote_sampling_strategy` | Ratio para SMOTE | `1.0` |
| `--undersample_sampling_strategy` | Ratio para undersampling | `1.0` |
| `--use_smote` | Alias legacy (tiene prioridad si se activa) | `False` |

---

## Notebooks y Reportes

### Notebooks

- `notebooks/01_eda_cardio.ipynb`: Analisis exploratorio
- `notebooks/02_featuring_cardio.ipynb`: Validacion de feature engineering
- `notebooks/03_training_visualization.ipynb`: Visualizacion de entrenamiento y artefactos
- `notebooks/04_evaluacion.ipynb`: Evaluacion formal en test
- `notebooks/05_comparativa_experimentos_tesis.ipynb`: Comparativa global entre experimentos

### Salidas de reporte

- Tablas:
  - `report/tables/comparativa_experimentos_test.csv`
  - `report/tables/ganadores_por_experimento_test.csv`
  - `report/tables/matriz_experimentos_recomendada.csv`
- Figuras:
  - Heatmaps de AUC/F1
  - Tendencias por experimento
  - ROC de ganadores
  - Matrices de confusion
  - Importancia de variables

---

## Resultados y Experimentacion

### Resumen del ultimo corte comparativo

Segun `report/tables/comparativa_experimentos_test.csv`:

- Modelos evaluados: 80
- Experimentos cubiertos: 20
- Ganador por `AUC-ROC(test)`: `Experimento17 - XGBoost`
  - `AUC-ROC(test): 0.8059`
  - `F1(test): 0.7418`
  - `MCC(test): 0.4368`
  - `Delta Test-Val (AUC): +0.0044`
- Ganador por `F1(test)`: `Experimento9 - LightGBM`
  - `F1(test): 0.7436`
  - `AUC-ROC(test): 0.8044`
  - `MCC(test): 0.4496`
  - `Delta Test-Val (AUC): +0.0036`

### Parametros, argumentos y criterios considerados

**Criterios comunes de optimizacion y evaluacion**

- Optuna con `TPE` y `MedianPruner`.
- Funcion objetivo compuesta: `0.70 * ROC-AUC + 0.30 * F1 - 0.15 * gap_auc`.
- CV estratificada (`cv_folds=5`).
- Ajuste de umbral en validacion y aplicacion directa en test.
- Resampling con `SMOTE` (`sampling_strategy=1.0`).
- `drop_uncertain_cases=False`.

**Ganador por AUC-ROC: Experimento17 - XGBoost**

- Argumentos clave: `cv_folds=5`, `trials=10`, `resampling_method=smote`, `smote_sampling_strategy=1.0`, `threshold=0.34`.
- Mejores hiperparametros:
  - `n_estimators=1393`, `max_depth=4`, `learning_rate=0.0064319`
  - `subsample=0.8413`, `colsample_bytree=0.7147`, `min_child_weight=12`
  - `gamma=3.1005`, `reg_alpha=1.6079`, `reg_lambda=1.6024`

**Ganador por F1: Experimento9 - LightGBM**

- Argumentos clave: `cv_folds=5`, `trials=30`, `smote_sampling_strategy=1.0`, `threshold=0.365`.
- Mejores hiperparametros:
  - `n_estimators=1334`, `learning_rate=0.0027697`, `num_leaves=20`, `max_depth=6`
  - `min_child_samples=69`, `subsample=0.8680`, `colsample_bytree=0.8311`
  - `reg_alpha=2.0043`, `reg_lambda=0.0003584`

### Criterios de interpretacion clinica

- Priorizar `Recall` alto para minimizar falsos negativos (pacientes en riesgo no detectados).
- Mantener `Precision` en rango aceptable para evitar sobrealerta.
- Usar `MCC` y `ROC-AUC` como metricas de equilibrio global.

---

## Configuracion del Entorno

### Requisitos

- Python 3.12+
- Git
- DVC

### Instalacion

#### 1) Clonar

```bash
git clone https://github.com/JhandryChimbo/Prediccion_Cardiovascular.git
cd Prediccion_Cardiovascular
```

#### 2) Entorno virtual

Windows (PowerShell):

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv env
source env/bin/activate
```

#### 3) Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4) Datos

```bash
dvc pull
```

Si no tienes acceso al remoto DVC, coloca manualmente `cardio_train.csv` en `data/raw/`.

---

## Reproducibilidad

Este proyecto prioriza reproducibilidad mediante:

1. Codigo versionado con Git
2. Pipeline declarativo en `dvc.yaml`
3. Artefactos de entrenamiento por experimento (`ExperimentoN/training_artifacts`)
4. Seeds fijas para split y CV
5. Separacion estricta de validacion y test

### Buenas practicas de evaluacion (importante)

- Ajustar umbral solo en validacion y aplicarlo tal cual en test.
- No re-optimizar umbral usando test.
- Comparar experimentos usando snapshots de test por experimento cuando existan.
- Mantener coherencia de features entre entrenamiento y evaluacion.

### Reproduccion rapida desde cero

```bash
git clone https://github.com/JhandryChimbo/Prediccion_Cardiovascular.git
cd Prediccion_Cardiovascular
python -m venv env
# Windows: .\env\Scripts\Activate.ps1
# Linux/macOS: source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
dvc pull
dvc repro
```

---

## Tecnologias y Herramientas

Dependencias principales (ver `requirements.txt`):

- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`, `lightgbm`
- `optuna`
- `imbalanced-learn`
- `matplotlib`, `seaborn`
- `jupyter`
- `dvc`
- `mlflow` (disponible para flujos alternativos)

### Nota sobre MLflow

El flujo DVC principal actual usa `train_model.py` y prioriza artefactos locales por experimento. `mlflow` se mantiene en dependencias para soportar flujos alternativos de tracking si se integran manualmente.

---

## Autor y Contacto

**Autor:** Jhandry Santiago Chimbo Rivera
**Proyecto:** Trabajo de Tesis - Prediccion de Enfermedades Cardiovasculares  
**Fecha:** Mayo 2026

Para consultas academicas o tecnicas, contactar a traves de los canales institucionales del autor.

---

<!-- ## Referencias

### Bases cientificas

1. World Health Organization (2021). Cardiovascular diseases (CVDs).  
   https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

2. Arnett DK, et al. (2019). 2019 ACC/AHA Guideline on the Primary Prevention of Cardiovascular Disease. *Circulation*, 140:e596-e646.

3. Framingham Heart Study. Risk Score Calculations.  
   https://www.framinghamheartstudy.org

### Referencias tecnicas

4. Akiba T, et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. KDD.
5. Chen T, Guestrin C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
6. Ke G, et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS. -->

### Dataset

Cardiovascular Disease Dataset (Kaggle):  
   https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

---

## Licencia

Proyecto academico de tesis. Uso orientado a investigacion y fines educativos, sujeto a politicas institucionales.

---

## Agradecimientos

A la comunidad open-source de ciencia de datos y machine learning, y en especial a los equipos de `scikit-learn`, `XGBoost`, `LightGBM`, `Optuna`, `DVC` y `MLflow`.

---

**Ultima actualizacion:** Mayo 2026  
**Version del documento:** 1.2

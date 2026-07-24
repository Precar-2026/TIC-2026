# Manual de Ejecución

**Proyecto:** Predicción de Enfermedades Cardiovasculares mediante Aprendizaje Automático  
**Autor:** Jhandry Santiago Chimbo Rivera  
**Versión:** 1.0 — JULIO 2026

---

## Tabla de Contenidos

1. [Antes de Ejecutar](#1-antes-de-ejecutar)
2. [Activar el Entorno Virtual](#2-activar-el-entorno-virtual)
3. [Ejecución Completa con DVC](#3-ejecución-completa-con-dvc)
4. [Ejecución Manual por Etapas](#4-ejecución-manual-por-etapas)
5. [Parámetros Avanzados de Entrenamiento](#5-parámetros-avanzados-de-entrenamiento)
6. [Ejecución de Notebooks](#6-ejecución-de-notebooks)
7. [Flujo de Experimentación](#7-flujo-de-experimentación)
8. [Revisión de Resultados](#8-revisión-de-resultados)
9. [Referencia de Parámetros](#9-referencia-de-parámetros)

---

## 1. Antes de Ejecutar

Asegúrate de haber completado el proceso de instalación descrito en `MANUAL_INSTALACION.md`. En particular, verifica que:

- [x] El entorno virtual `env/` fue creado y está disponible.
- [x] Las dependencias fueron instaladas con `pip install -r requirements.txt`.
- [x] El archivo `data/raw/cardio_train.csv` existe (obtenido por `dvc pull` o descarga manual).

---

## 2. Activar el Entorno Virtual

Antes de ejecutar cualquier comando, activa el entorno virtual desde la raíz del proyecto.

### Windows (PowerShell)

```powershell
.\env\Scripts\Activate.ps1
```

### Windows (CMD)

```cmd
env\Scripts\activate.bat
```

### Linux / macOS

```bash
source env/bin/activate
```

Confirmación: deberás ver el prefijo `(env)` en tu terminal:

```
(env) C:\...\Prediccion_Cardiovascular>
```

---

## 3. Ejecución Completa con DVC

La forma más rápida y recomendada de reproducir todo el experimento es usando **DVC**, que orquesta automáticamente todas las etapas en el orden correcto.

```bash
dvc repro
```

### ¿Qué hace `dvc repro`?

DVC ejecuta las siguientes etapas secuencialmente, omitiendo las que no han cambiado:

| Etapa | Script | Descripción |
|-------|--------|-------------|
| `limpieza_datos` | `src/data/make_dataset.py` | Limpieza fisiológica del CSV original |
| `ingenieria_caracteristicas` | `src/features/build_features.py` | Generación de variables derivadas y escalado |
| `seleccion_caracteristicas` | `src/features/select_features.py` | Selección con RFECV usando LightGBM |
| `train_random_forest` | `src/models/train_model.py` | Entrenamiento y optimización de Random Forest |
| `train_logistic_regression` | `src/models/train_model.py` | Entrenamiento y optimización de Regresión Logística |
| `train_xgboost` | `src/models/train_model.py` | Entrenamiento y optimización de XGBoost |
| `train_lightgbm` | `src/models/train_model.py` | Entrenamiento y optimización de LightGBM |

### Verificar el estado del pipeline

```bash
dvc status
```

Salida esperada si todo está al día:

```
Data and pipelines are up to date.
```

> [!NOTE]
> El pipeline completo puede tardar entre **30 minutos y varias horas** dependiendo del número de `trials` de Optuna y la potencia de tu equipo.

---

## 4. Ejecución Manual por Etapas

Si deseas ejecutar o modificar etapas individualmente, usa los siguientes comandos:

### Etapa 1 — Limpieza de Datos

```bash
python src/data/make_dataset.py \
  --input data/raw/cardio_train.csv \
  --output data/processed/cardio_clean.csv \
  --altura_min 130 --altura_max 220 \
  --peso_min 30 --peso_max 200 \
  --ap_hi_min 80 --ap_hi_max 240 \
  --ap_lo_min 50 --ap_lo_max 140
```

**En Windows (una sola línea):**

```powershell
python src/data/make_dataset.py --input data/raw/cardio_train.csv --output data/processed/cardio_clean.csv --altura_min 130 --altura_max 220 --peso_min 30 --peso_max 200 --ap_hi_min 80 --ap_hi_max 240 --ap_lo_min 50 --ap_lo_max 140
```

**Salida generada:** `data/processed/cardio_clean.csv`

---

### Etapa 2 — Ingeniería de Características

```bash
python src/features/build_features.py \
  --input data/processed/cardio_clean.csv \
  --output_dir data/processed/model_inputs \
  --scaler_path models/scaler.joblib \
  --scaler_type robust \
  --drop_features log_ap_hi,pressure_ratio,col_gluc_ratio
```

**Salidas generadas:**
- `data/processed/model_inputs/` (X_train, X_val, X_test, y_train, y_val, y_test)
- `models/scaler.joblib`

---

### Etapa 3 — Selección de Características (RFECV)

```bash
python src/features/select_features.py \
  --input_dir data/processed/model_inputs \
  --output_dir data/processed/model_inputs_rfecv
```

**Salidas generadas:**
- `data/processed/model_inputs_rfecv/` (datasets con features seleccionadas)
- `data/processed/model_inputs_rfecv/selected_features.pkl`
- `report/figures/rfecv_results.png`

---

### Etapa 4 — Entrenamiento de Modelos

Cada modelo se entrena de forma independiente. A continuación se muestran los comandos para cada uno:

#### Random Forest (RF)

```bash
python src/models/train_model.py \
  --input_dir data/processed/model_inputs_rfecv \
  --model RF \
  --models_dir models \
  --cv_folds 5 \
  --trials 50 \
  --resampling_method smote \
  --smote_sampling_strategy 1.0 \
  --reset_experiment
```

#### Regresión Logística (LR)

```bash
python src/models/train_model.py \
  --input_dir data/processed/model_inputs_rfecv \
  --model LR \
  --models_dir models \
  --cv_folds 5 \
  --trials 50 \
  --resampling_method smote \
  --smote_sampling_strategy 1.0
```

#### XGBoost (XGB)

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

#### LightGBM (LGBM)

```bash
python src/models/train_model.py \
  --input_dir data/processed/model_inputs_rfecv \
  --model LGBM \
  --models_dir models \
  --cv_folds 5 \
  --trials 50 \
  --resampling_method smote \
  --smote_sampling_strategy 1.0
```

> [!TIP]
> Usa `--trials 10` para una ejecución rápida de prueba. Para resultados óptimos (como los reportados en la tesis), usa `--trials 50` o superior.

---

## 5. Parámetros Avanzados de Entrenamiento

### Cambiar el método de resampling

```bash
# Sin resampling
python src/models/train_model.py ... --resampling_method none

# Con undersampling
python src/models/train_model.py ... --resampling_method undersample --undersample_sampling_strategy 0.8
```

### Entrenar solo en pacientes hipertensos

```bash
python src/features/build_features.py ... --use_hypertensive_only
```

### Desactivar el ajuste de umbral

```bash
python src/models/train_model.py ... --no_threshold_tuning
```

### Reiniciar el contador de experimentos

El flag `--reset_experiment` reinicia el conteo de carpetas `ExperimentoN` desde 1. **Úsalo solo al inicio de una nueva serie experimental**, ya que sobrescribirá los experimentos previos.

```bash
python src/models/train_model.py ... --reset_experiment
```

### Eliminar casos inciertos del entrenamiento

```bash
python src/models/train_model.py ... --drop_uncertain_cases --uncertainty_quantile 0.10
```

### Usar escalado estándar en lugar de robusto

```bash
python src/features/build_features.py ... --scaler_type standard
```

---

## 6. Ejecución de Notebooks

Los notebooks permiten realizar análisis exploratorio, visualizar el entrenamiento y comparar experimentos de forma interactiva.

### Iniciar Jupyter Notebook

```bash
jupyter notebook
```

Se abrirá automáticamente el navegador en `http://localhost:8888`. Navega a la carpeta `notebooks/` y abre el notebook deseado.

### Descripción de los Notebooks

| Notebook | Descripción | Cuándo ejecutarlo |
|----------|-------------|-------------------|
| `01_eda_cardio.ipynb` | Análisis exploratorio de datos (EDA) | Antes del pipeline, para entender el dataset |
| `02_featuring_cardio.ipynb` | Validación del feature engineering | Después de la Etapa 2 |
| `03_training_visualization.ipynb` | Visualización de curvas de entrenamiento y artefactos por experimento | Después de entrenar modelos |
| `04_evaluacion.ipynb` | Evaluación formal de modelos en el conjunto de test | Después del entrenamiento completo |
| `05_comparativa_experimentos_tesis.ipynb` | Comparativa global entre todos los experimentos | Para obtener el resumen final de la tesis |

### Ejecutar un notebook desde la terminal (sin interfaz)

```bash
jupyter nbconvert --to notebook --execute notebooks/05_comparativa_experimentos_tesis.ipynb --output notebooks/05_comparativa_experimentos_tesis_ejecutado.ipynb
```

---

## 7. Flujo de Experimentación

Para reproducir o crear nuevos experimentos de forma controlada, sigue este flujo:

```
1. Modificar parámetros en dvc.yaml (o ejecutar manualmente con flags distintos)
        ↓
2. Ejecutar pipeline: dvc repro  (o scripts individuales)
        ↓
3. Revisar artefactos en: models/ExperimentoN/training_artifacts/
        ↓
4. Ejecutar notebooks/04_evaluacion.ipynb para evaluar en test
        ↓
5. Ejecutar notebooks/05_comparativa_experimentos_tesis.ipynb para comparar
        ↓
6. Revisar tablas resumen en: report/tables/
```

### Estructura de artefactos por experimento

Cada experimento genera la siguiente estructura dentro de `models/ExperimentoN/`:

```text
models/
└── ExperimentoN/
    ├── best_model_RF.joblib
    ├── best_model_LR.joblib
    ├── best_model_XGB.joblib
    ├── best_model_LGBM.joblib
    └── training_artifacts/
        ├── optuna_trials_<modelo>.csv       ← Historial de trials de Optuna
        ├── val_metrics_<modelo>.csv         ← Métricas en validación
        ├── best_params_<modelo>.json        ← Mejores hiperparámetros encontrados
        ├── threshold_<modelo>.json          ← Umbral de clasificación ajustado
        ├── val_predictions_<modelo>.csv     ← Predicciones en validación
        ├── overfitting_diagnostics_<modelo>.json ← Diagnóstico de sobreajuste
        └── model_inputs_snapshot/           ← Snapshot de X_test/y_test
```

---

## 8. Revisión de Resultados

### Tablas comparativas

Los resultados consolidados se guardan en `report/tables/`:

| Archivo | Contenido |
|---------|-----------|
| `comparativa_experimentos_test.csv` | Métricas de todos los experimentos sobre test |
| `ganadores_por_experimento_test.csv` | El mejor modelo de cada experimento |
| `matriz_experimentos_recomendada.csv` | Resumen ejecutivo recomendado |

### Visualizar tablas rápidamente

```python
import pandas as pd
df = pd.read_csv('report/tables/comparativa_experimentos_test.csv')
print(df.sort_values('AUC-ROC(test)', ascending=False).head(10))
```

### Figuras generadas

Las gráficas se guardan en `report/figures/` e incluyen:

- Heatmaps de AUC/F1 por experimento
- Curvas ROC de los modelos ganadores
- Matrices de confusión
- Importancia de variables (feature importance)
- Resultados de RFECV (`rfecv_results.png`)

---

## 9. Referencia de Parámetros

### `train_model.py` — Parámetros completos

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--input_dir` | str | — | Directorio con datasets preprocesados (requerido) |
| `--model` | str | — | Algoritmo: `LR`, `RF`, `XGB`, `LGBM` (requerido) |
| `--models_dir` | str | `models` | Carpeta de salida para modelos y experimentos |
| `--trials` | int | `80` | Número de trials de Optuna |
| `--cv_folds` | int | `5` | Número de folds de validación cruzada estratificada |
| `--resampling_method` | str | `none` | Método: `none`, `smote`, `undersample` |
| `--smote_sampling_strategy` | float | `1.0` | Ratio de balanceo para SMOTE |
| `--undersample_sampling_strategy` | float | `1.0` | Ratio de balanceo para undersampling |
| `--no_threshold_tuning` | flag | `False` | Desactiva el ajuste de umbral en validación |
| `--reset_experiment` | flag | `False` | Reinicia el contador de `ExperimentoN` |
| `--drop_uncertain_cases` | flag | `False` | Elimina casos inciertos del conjunto de entrenamiento |
| `--uncertainty_quantile` | float | `0.10` | Cuantil de incertidumbre para la limpieza de casos |
| `--use_smote` | flag | `False` | Alias legacy para activar SMOTE (tiene prioridad) |

### `build_features.py` — Parámetros principales

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--input` | str | — | Archivo CSV limpio de entrada (requerido) |
| `--output_dir` | str | — | Directorio de salida de los datasets (requerido) |
| `--scaler_path` | str | — | Ruta para guardar el scaler entrenado |
| `--scaler_type` | str | `robust` | Tipo de escalado: `robust`, `standard`, `none` |
| `--drop_features` | str | — | Features a eliminar, separadas por comas |
| `--disable_advanced_features` | flag | `False` | Desactiva la generación de features avanzadas |
| `--use_hypertensive_only` | flag | `False` | Filtra el dataset a solo pacientes hipertensos |

### `make_dataset.py` — Filtros fisiológicos

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--input` | str | — | Ruta al CSV original (`cardio_train.csv`) |
| `--output` | str | — | Ruta de salida del CSV limpio |
| `--altura_min` | int | `130` | Altura mínima permitida (cm) |
| `--altura_max` | int | `220` | Altura máxima permitida (cm) |
| `--peso_min` | int | `30` | Peso mínimo permitido (kg) |
| `--peso_max` | int | `200` | Peso máximo permitido (kg) |
| `--ap_hi_min` | int | `80` | Presión sistólica mínima (mmHg) |
| `--ap_hi_max` | int | `240` | Presión sistólica máxima (mmHg) |
| `--ap_lo_min` | int | `50` | Presión diastólica mínima (mmHg) |
| `--ap_lo_max` | int | `140` | Presión diastólica máxima (mmHg) |
| `--allow_equal_bp` | flag | `False` | Permite registros con `ap_hi == ap_lo` |

---

> [!IMPORTANT]
> Para una reproducción exacta de los experimentos reportados en la tesis, usa `--trials 50`, `--cv_folds 5` y `--resampling_method smote` con los filtros fisiológicos por defecto definidos en `dvc.yaml`.

---

**Última actualización:** Mayo 2026  
**Versión del documento:** 1.0

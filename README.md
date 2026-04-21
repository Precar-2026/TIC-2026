# Predicción de Enfermedades Cardiovasculares mediante Aprendizaje Automático

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-945DD6.svg)](https://dvc.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

---

## Resumen Ejecutivo

Este proyecto implementa un sistema de predicción de enfermedades cardiovasculares basado en técnicas de aprendizaje automático supervisado. El objetivo principal es desarrollar modelos predictivos robustos y clínicamente interpretables que puedan asistir en la detección temprana de riesgo cardiovascular a partir de variables demográficas, clínicas y de estilo de vida.

El proyecto forma parte de una investigación académica de tesis que aborda la aplicación de metodologías avanzadas de ciencia de datos en el ámbito de la medicina preventiva cardiovascular.

**Palabras clave:** Enfermedades cardiovasculares, Machine Learning, Optimización bayesiana, MLflow, DVC, Ingeniería de características médicas.

---

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Objetivos](#objetivos)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Metodología](#metodología)
5. [Tecnologías y Herramientas](#tecnologías-y-herramientas)
6. [Configuración del Entorno](#configuración-del-entorno)
7. [Pipeline de Datos y Entrenamiento](#pipeline-de-datos-y-entrenamiento)
8. [Resultados y Experimentación](#resultados-y-experimentación)
9. [Reproducibilidad](#reproducibilidad)
10. [Autor y Contacto](#autor-y-contacto)
11. [Referencias](#referencias)

---

## Introducción

Las enfermedades cardiovasculares (ECV) representan la principal causa de mortalidad a nivel mundial, según la Organización Mundial de la Salud (OMS). La detección temprana de individuos en riesgo es fundamental para implementar intervenciones preventivas efectivas y reducir la carga de morbilidad asociada.

Este proyecto desarrolla un sistema predictivo que utiliza algoritmos de aprendizaje automático para identificar pacientes con alto riesgo cardiovascular. El enfoque metodológico combina:

- **Análisis Exploratorio de Datos (EDA)** para comprender patrones y relaciones clínicas
- **Ingeniería de características médicamente fundamentada** basada en guías clínicas (AHA/ACC)
- **Optimización bayesiana de hiperparámetros** mediante Optuna
- **Gestión de experimentos** con MLflow para trazabilidad y reproducibilidad
- **Control de versiones de datos** con DVC

El dataset utilizado contiene información de 70,000 pacientes con variables como edad, presión arterial, índice de masa corporal, colesterol, glucosa, hábitos de tabaquismo y actividad física.

---

## Objetivos

### Objetivo General

Desarrollar e implementar modelos predictivos de aprendizaje automático para la detección temprana de enfermedades cardiovasculares, evaluando su rendimiento y aplicabilidad clínica.

### Objetivos Específicos

1. **Realizar un análisis exploratorio de datos** para identificar patrones, distribuciones y relaciones en las variables clínicas del dataset cardiovascular.

2. **Implementar un pipeline de preprocesamiento robusto** que incluya limpieza de datos basada en criterios fisiológicos y médicos.

3. **Diseñar e implementar características derivadas** (feature engineering) basadas en conocimiento médico cardiovascular, incluyendo:
   - Pulse Pressure (indicador de rigidez arterial)
   - Mean Arterial Pressure (MAP)
   - Rate Pressure Product (RPP - índice de trabajo cardíaco)
   - Índice de Masa Corporal (IMC) y categorías clínicas
   - Interacciones entre factores de riesgo

4. **Entrenar y optimizar múltiples algoritmos de clasificación**, incluyendo:
   - Regresión Logística (baseline interpretable)
   - Random Forest
   - XGBoost
   - LightGBM

5. **Optimizar hiperparámetros** mediante búsqueda bayesiana (Optuna) para maximizar el rendimiento predictivo.

6. **Implementar un sistema de tracking de experimentos** con MLflow para garantizar reproducibilidad y trazabilidad.

7. **Evaluar y comparar modelos** utilizando métricas apropiadas para problemas de clasificación médica (Accuracy, Precision, Recall, F1-Score, ROC-AUC, MCC).

8. **Garantizar la reproducibilidad** mediante control de versiones de código (Git) y datos (DVC).

---

## Estructura del Proyecto

```
Prediccion_Cardiovascular/
│
├── data/                           
│   ├── raw/                        # Datos originales sin procesar
│   │   ├── .gitkeep                # (los archivos .csv se obtienen con DVC)
│   │   └── cardio_train.csv.dvc    # Archivo DVC para versionar datos
│   │
│   ├── processed/                  # Datos procesados (generados localmente)
│   │   └── .gitkeep                # (archivos generados por el pipeline)
│   │
│   └── report/                     # Reportes generados
│
├── src/                            # Código fuente del proyecto
│   ├── data/
│   │   └── make_dataset.py         # Script de limpieza y preprocesamiento
│   │
│   ├── features/
│   │   └── build_features.py       # Script de ingeniería de características
│   │
│   └── models/
│       └── train_model.py          # Script de entrenamiento y optimización
│
├── models/                         
│   └── .gitkeep                    # Los modelos se generan tras entrenar
│
├── notebooks/                      # Notebooks Jupyter 
│   ├── 01_eda_cardio.ipynb         
│   ├── 02_featuring_cardio.ipynb   
│   └── 03_evaluacion.ipynb                                   
│
├── .gitignore                      # Archivos ignorados por Git
├── dvc.yaml                        # Pipeline DVC 
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Este archivo
```

**Nota importante:** Los siguientes elementos se generan localmente y **NO** están en el repositorio Git:
- `env/` - Entorno virtual de Python
- `data/raw/*.csv` - Datos originales (se obtienen con `dvc pull`)
- `data/processed/` - Datos procesados (se generan con el pipeline)
- `models/` - Modelos entrenados (se generan con el pipeline o entrenamiento)
- `mlartifacts/`, `mlruns/` - Artefactos de MLflow (generados localmente)
- `*.joblib`, `*.pkl` - Archivos de modelos serializados

---

## Metodología

### 1. Análisis Exploratorio de Datos (EDA)

**Notebook:** `01_eda_cardio.ipynb`

- Análisis de distribuciones de variables numéricas y categóricas
- Identificación de valores atípicos (outliers) y anomalías
- Análisis de correlaciones entre variables
- Estratificación por grupos de riesgo
- Visualizaciones clínicas (distribuciones de presión arterial, IMC, edad)
- Identificación de rangos fisiológicos aceptables

**Hallazgos clave:**
- Dataset naturalmente balanceado (50% con ECV, 50% sin ECV)
- Identificación de valores fisiológicamente imposibles
- Fuerte correlación entre presión arterial y riesgo cardiovascular
- Importancia del IMC y edad como factores predictivos

### 2. Preprocesamiento de Datos

**Script:** `src/data/make_dataset.py`

**Filtros aplicados:**
- **Altura:** 130-220 cm
- **Peso:** 30-200 kg
- **Presión sistólica:** 80-240 mmHg
- **Presión diastólica:** 50-140 mmHg
- **Validación médica:** presión_sistólica > presión_diastólica

**Transformaciones:**
- Conversión de edad de días a años
- Cálculo de Índice de Masa Corporal (IMC)
- Eliminación de registros fisiológicamente inverosímiles

### 3. Ingeniería de Características

**Script:** `src/features/build_features.py`

**Características médicas derivadas:**

1. **Pulse Pressure (PP):** Diferencia entre presión sistólica y diastólica
   - Indicador de rigidez arterial
   
2. **Mean Arterial Pressure (MAP):** Presión arterial promedio
   - Fórmula: MAP = PD + (PP/3)
   
3. **Rate Pressure Product (RPP):** Índice de trabajo cardíaco
   - Aproximación del consumo de oxígeno del miocardio
   
4. **Categorías de IMC:** Clasificación OMS
   - Bajo peso (<18.5), Normal (18.5-24.9), Sobrepeso (25-29.9), Obesidad (≥30)
   
5. **Grupos de edad:** Estratificación por riesgo cardiovascular
   - Joven (<45), Adulto (45-60), Mayor (>60)
   
6. **Índices de riesgo combinados:**
   - Conteo de factores de riesgo (tabaquismo, alcohol, inactividad)
   - Índice de presión arterial
   
7. **Interacciones entre variables:**
   - Productos y ratios entre características clave

**Técnicas de escalado:**
- RobustScaler: Robusto a valores atípicos residuales
- Normalización por cuartiles (percentil 25-75)

**Selección de características:**
- Filtro VarianceThreshold
- Análisis de correlación para reducir multicolinealidad

### 4. Modelado y Optimización

**Script:** `src/models/train_model.py`

**Modelos implementados:**

| Modelo | Justificación | Trials Optuna |
|--------|---------------|---------------|
| **Logistic Regression** | Baseline interpretable, apropiado para relaciones lineales | 50 |
| **Random Forest** | Maneja bien variables ordinales, no requiere normalización estricta | 85 |
| **XGBoost** | Alto rendimiento, captura relaciones complejas, robusto | 100 |
| **LightGBM** | Eficiente computacionalmente, alternativa competitiva a XGBoost | 100 |

**Estrategia de optimización:**
- **Búsqueda bayesiana** con Optuna (TPE Sampler)
- **Validación cruzada estratificada** (5-fold StratifiedKFold)
- **Early stopping** para evitar sobreajuste
- **Función objetivo:** Maximizar F1-Score ponderado

**Métricas de evaluación:**
- **Accuracy:** Proporción de predicciones correctas
- **Precision:** Capacidad de no etiquetar negativos como positivos
- **Recall (Sensibilidad):** Capacidad de identificar positivos correctamente
- **F1-Score:** Media armónica de precision y recall
- **ROC-AUC:** Área bajo la curva ROC
- **MCC (Matthews Correlation Coefficient):** Métrica robusta para clasificación binaria

### 5. Gestión de Experimentos

**MLflow:** Sistema de tracking de experimentos
- URI de tracking: `http://127.0.0.1:5000`
- Nombre de experimento: `Tesis_Cardio_Prediccion`
- Registro automático de hiperparámetros, métricas y modelos
- Versionado de modelos con tags y etapa (staging/production)

**DVC:** Control de versiones de datos
- Pipeline declarativo en `dvc.yaml`
- Reproducibilidad garantizada de todo el flujo de trabajo
- Tracking de cambios en datasets

---

## Tecnologías y Herramientas

### Lenguaje y Entorno
- **Python 3.12**
- **Entorno virtual** (venv)

### Bibliotecas de Ciencia de Datos
| Biblioteca | Versión | Propósito |
|------------|---------|-----------|
| `pandas` | Latest | Manipulación de datos |
| `numpy` | Latest | Operaciones numéricas |
| `scikit-learn` | Latest | Algoritmos ML y preprocesamiento |
| `matplotlib` | Latest | Visualización |
| `seaborn` | Latest | Visualización estadística |

### Frameworks de Machine Learning
| Framework | Propósito |
|-----------|-----------|
| `xgboost` | Gradient boosting optimizado |
| `lightgbm` | Gradient boosting eficiente |
| `optuna` | Optimización bayesiana de hiperparámetros |
| `imbalanced-learn` | Manejo de datasets desbalanceados (disponible) |

### MLOps y Gestión de Experimentos
| Herramienta | Propósito |
|-------------|-----------|
| `mlflow` | Tracking de experimentos, registro de modelos |
| `dvc` | Control de versiones de datos y pipelines |
| `joblib` | Serialización eficiente de modelos |

### Análisis y Desarrollo
- **Jupyter Notebook:** Análisis exploratorio e interactivo
- **Git:** Control de versiones de código

---

## Configuración del Entorno

### Requisitos Previos

- Python 3.12 o superior
- Git
- DVC (Data Version Control)

### Instalación

#### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd Prediccion_Cardiovascular
```

#### 2. Crear y activar entorno virtual

**Windows (PowerShell):**
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3 -m venv env
source env/bin/activate
```

#### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Obtener datos con DVC

```bash
# Obtener datos versionados desde el storage remoto
dvc pull
```

**Nota:** Los datos no están en Git debido a su tamaño. DVC los descarga desde el almacenamiento remoto configurado. Si no tienes acceso al storage remoto de DVC, coloca manualmente `cardio_train.csv` en `data/raw/`.

#### 5. Configurar MLflow (opcional, para tracking local)

```bash
# Iniciar servidor MLflow en terminal separado
mlflow ui --host 127.0.0.1 --port 5000
```

Acceder a la interfaz web en: http://127.0.0.1:5000

---

## Pipeline de Datos y Entrenamiento

### Ejecución Completa con DVC

El proyecto utiliza DVC para orquestar todo el pipeline de forma reproducible:

```bash
# Ejecutar pipeline completo
dvc repro
```

Este comando ejecuta secuencialmente:
1. Limpieza de datos
2. Ingeniería de características
3. Entrenamiento de Random Forest
4. Entrenamiento de Logistic Regression
5. Entrenamiento de XGBoost
6. Entrenamiento de LightGBM

### Ejecución Manual por Etapas

#### 1. Limpieza de Datos

```bash
python src/data/make_dataset.py \
    --input data/raw/cardio_train.csv \
    --output data/processed/cardio_clean.csv
```

#### 2. Ingeniería de Características

```bash
python src/features/build_features.py \
    --input data/processed/cardio_clean.csv \
    --output_dir data/processed/model_inputs \
    --scaler_path models/scaler.joblib
```

#### 3. Entrenamiento de Modelos

**Random Forest:**
```bash
python src/models/train_model.py \
    --input_dir data/processed/model_inputs \
    --model RF \
    --models_dir models \
    --trials 85 \
    --reset_experiment
```

**Logistic Regression:**
```bash
python src/models/train_model.py \
    --input_dir data/processed/model_inputs \
    --model LR \
    --models_dir models \
    --trials 50
```

**XGBoost:**
```bash
python src/models/train_model.py \
    --input_dir data/processed/model_inputs \
    --model XGB \
    --models_dir models \
    --trials 100
```

**LightGBM:**
```bash
python src/models/train_model.py \
    --input_dir data/processed/model_inputs \
    --model LGBM \
    --models_dir models \
    --trials 100
```

### Parámetros de Entrenamiento

| Parámetro | Descripción | Por defecto |
|-----------|-------------|-------------|
| `--input_dir` | Directorio con datos de entrada | - |
| `--model` | Tipo de modelo (LR, RF, XGB, LGBM) | - |
| `--models_dir` | Directorio para guardar modelos | `models/` |
| `--trials` | Número de trials Optuna | 100 |
| `--reset_experiment` | Reiniciar número de experimento | False |
| `--resampling_method` | Balanceo de clases: `none`, `smote`, `undersample` | `none` |
| `--undersample_sampling_strategy` | Ratio final en undersampling (minoría/mayoría) | `1.0` |

Compatibilidad: `--use_smote` sigue disponible como alias legacy y tiene prioridad si se activa.

---

## Resultados y Experimentación

### Estructura de Experimentos

**Nota:** Los modelos entrenados se generan localmente y **NO** se suben a GitHub (están en `.gitignore`). 

Después de entrenar, el proyecto mantiene un historial de experimentos en tu máquina local:

```
models/                          # (generado localmente, no en Git)
├── scaler.joblib                # Escalador entrenado
├── best_model_LR.joblib         # Mejor modelo Logistic Regression
├── best_model_RF.joblib         # Mejor modelo Random Forest
├── best_model_XGB.joblib        # Mejor modelo XGBoost
├── best_model_LGBM.joblib       # Mejor modelo LightGBM
│
└── Experimento[n]/              # Historial de experimentos previos
    ├── best_model_LR.joblib
    ├── best_model_RF.joblib
    ├── best_model_XGB.joblib
    └── best_model_LGBM.joblib
```

Los modelos se generan automáticamente al ejecutar el pipeline de entrenamiento.

### Evaluación de Modelos

**Notebook de evaluación:** `notebooks/03_evaluacion.ipynb`

Este notebook contiene:
- Carga de modelos entrenados
- Evaluación en conjunto de prueba
- Matrices de confusión
- Curvas ROC y PR (Precision-Recall)
- Comparación de métricas entre modelos
- Análisis de importancia de características
- Interpretabilidad de predicciones

### Acceso a Resultados en MLflow

1. Iniciar servidor MLflow:
   ```bash
   mlflow ui --host 127.0.0.1 --port 5000
   ```

2. Navegar a http://127.0.0.1:5000

3. Seleccionar experimento: `Tesis_Cardio_Prediccion`

4. Explorar:
   - Comparación de runs
   - Hiperparámetros óptimos
   - Métricas de performance
   - Artefactos y modelos registrados

### Interpretación de Resultados

Los modelos se evalúan considerando el **contexto médico**:

- **Recall alto:** Prioridad en identificar pacientes en riesgo (minimizar falsos negativos)
- **Precision aceptable:** Evitar alarmas falsas innecesarias
- **ROC-AUC:** Capacidad discriminativa global del modelo
- **MCC:** Métrica balanceada que considera verdaderos/falsos positivos y negativos

---

## Reproducibilidad

### Garantías de Reproducibilidad

Este proyecto implementa las mejores prácticas de reproducibilidad científica:

1. **Control de versiones de código:** Git (scripts, notebooks, configuración)
2. **Control de versiones de datos:** DVC (datasets versionados)
3. **Gestión de dependencias:** `requirements.txt` con versiones específicas
4. **Seeds aleatorias:** Definidas en scripts para reproducir splits y optimizaciones
5. **Pipeline declarativo:** `dvc.yaml` define dependencias y salidas de cada etapa
6. **Tracking de experimentos:** MLflow registra todos los hiperparámetros y métricas

### Archivos Versionados en Git

**Incluidos en el repositorio:**
- ✅ Código fuente (`src/`)
- ✅ Notebooks de análisis (`notebooks/`)
- ✅ Pipeline DVC (`dvc.yaml`)
- ✅ Dependencias (`requirements.txt`)
- ✅ Archivos DVC (`.dvc`) para versionado de datos
- ✅ Documentación (`README.md`)

**Generados localmente (NO en Git):**
- ❌ Datos (`data/raw/`, `data/processed/`) - se obtienen con DVC o pipeline
- ❌ Modelos (`models/`, `*.joblib`) - se generan tras entrenar
- ❌ Entorno virtual (`env/`) - se crea localmente
- ❌ Artefactos MLflow (`mlartifacts/`, `mlruns/`) - se generan localmente

### Reproducir Resultados

Para reproducir exactamente los resultados desde un repositorio limpio:

```bash
# 1. Clonar repositorio (solo contiene código, no datos ni modelos)
git clone <URL_DEL_REPOSITORIO>
cd Prediccion_Cardiovascular

# 2. Crear y activar entorno virtual
python -m venv env

# Windows (PowerShell):
.\env\Scripts\Activate.ps1

# Linux/macOS:
# source env/bin/activate

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Obtener datos con DVC (si tienes acceso al storage remoto)
dvc pull

# 4.1. Alternativa: Colocar datos manualmente si no tienes acceso a DVC remote
# Coloca cardio_train.csv en data/raw/

# 5. Ejecutar pipeline completo (genera datos procesados y entrena modelos)
dvc repro

# 6. (Opcional) Ver resultados en MLflow
mlflow ui --host 127.0.0.1 --port 5000
# Navega a http://127.0.0.1:5000
```

**Resultado esperado:**
- Datos procesados en `data/processed/`
- Modelos entrenados en `models/`
- Experimentos registrados en MLflow (directorio `mlruns/` local)

---

## Autor y Contacto

**Autor:** Jhandry U  
**Institución:** [Nombre de la Universidad/Institución]  
**Proyecto:** Trabajo de Tesis - Predicción de Enfermedades Cardiovasculares  
**Fecha:** Marzo 2026  

Para consultas académicas o técnicas sobre este proyecto, contactar a través de: [correo@ejemplo.com]

---

## Referencias

### Bases Científicas

1. **World Health Organization (2021).** Cardiovascular diseases (CVDs). [https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))

2. **American Heart Association (2019).** 2019 ACC/AHA Guideline on the Primary Prevention of Cardiovascular Disease. Circulation, 140:e596-e646.

3. **Framingham Heart Study.** Risk Score Calculations. [https://www.framinghamheartstudy.org](https://www.framinghamheartsudy.org)

### Referencias Técnicas

4. **Bergstra, J., & Bengio, Y. (2012).** Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research, 13, 281-305.

5. **Akiba, T., et al. (2019).** Optuna: A Next-generation Hyperparameter Optimization Framework. KDD 2019.

6. **Chen, T., & Guestrin, C. (2016).** XGBoost: A Scalable Tree Boosting System. KDD 2016.

7. **Ke, G., et al. (2017).** LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS 2017.

### Dataset

8. **Cardiovascular Disease Dataset.** Kaggle. [https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

### Herramientas

9. **MLflow Documentation.** [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)

10. **DVC Documentation.** [https://dvc.org/doc](https://dvc.org/doc)

11. **Scikit-learn: Machine Learning in Python.** Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

---

## Licencia

Este proyecto es parte de un trabajo académico de tesis y está sujeto a las políticas de la institución educativa correspondiente. El código está disponible con fines educativos y de investigación.

---

## Agradecimientos

Se agradece a la comunidad open-source de ciencia de datos y machine learning por las herramientas y bibliotecas que hicieron posible este proyecto. Especialmente a los desarrolladores de scikit-learn, XGBoost, LightGBM, Optuna, MLflow y DVC.

---

**Última actualización:** Marzo 2026  
**Versión del documento:** 1.0

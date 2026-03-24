"""
Script de Entrenamiento y Optimización de Modelos - Enfermedades Cardiovasculares

Este módulo implementa el entrenamiento y optimización de modelos de clasificación
para la predicción de enfermedades cardiovasculares. Se basa en los hallazgos del
Análisis Exploratorio de Datos (EDA) y utiliza:

1. Optuna: Búsqueda bayesiana de hiperparámetros
2. MLflow: Seguimiento de experimentos y registro de modelos
3. Validación cruzada estratificada para evaluación robusta

Modelos implementados (justificación basada en EDA):
- Regresión Logística (LR): Baseline interpretable, apropiado para relaciones lineales
- Random Forest (RF): Maneja bien variables ordinales y no requiere normalización estricta
- XGBoost (XGB): Alto rendimiento, maneja relaciones complejas, robusto a outliers
- LightGBM (LGBM): Eficiente computacionalmente, alternativa a XGBoost

Decisión: No se aplica balanceo (SMOTE) ya que el dataset está naturalmente balanceado
según el EDA (50% con enfermedad, 50% sin enfermedad).

Autor: Jhandry U
Fecha: Marzo 2026
"""

import pandas as pd
import numpy as np
import argparse
import logging
import os
import joblib
import warnings
import re
import shutil
import json
from typing import Dict, Any

import optuna
from optuna.samplers import TPESampler

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)

# Suprimir warnings innecesarios
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuración de MLflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Tesis_Cardio_Prediccion"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def get_current_experiment_number(models_dir: str) -> int:
    """
    Obtiene o crea el número de experimento actual para el pipeline.
    
    Si existe un archivo .current_experiment.txt, usa ese número para que todos 
    los modelos del pipeline se guarden en la misma carpeta de experimento.
    Si no existe, crea uno nuevo basado en las carpetas existentes.
    
    Args:
        models_dir: Directorio donde se almacenan los experimentos
        
    Returns:
        Número del experimento actual (int)
    """
    os.makedirs(models_dir, exist_ok=True)
    
    current_exp_file = os.path.join(models_dir, '.current_experiment.txt')
    
    # Si existe el archivo de experimento actual, usar ese número
    if os.path.exists(current_exp_file):
        try:
            with open(current_exp_file, 'r') as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            pass
    
    # Si no existe, crear uno nuevo
    # Buscar el número más alto de experimentos existentes
    pattern = re.compile(r'^Experimento(\d+)$')
    experiment_numbers = []
    
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                match = pattern.match(item)
                if match:
                    experiment_numbers.append(int(match.group(1)))
    
    # Calcular el siguiente número
    next_number = max(experiment_numbers) + 1 if experiment_numbers else 1
    
    # Guardar el número actual en el archivo
    with open(current_exp_file, 'w') as f:
        f.write(str(next_number))
    
    return next_number


def reset_experiment_counter(models_dir: str) -> None:
    """
    Elimina el archivo de experimento actual para iniciar uno nuevo.
    
    Llamar esta función después de completar todos los entrenamientos
    del pipeline actual si deseas que la próxima ejecución use un nuevo número.
    
    Args:
        models_dir: Directorio donde se almacenan los experimentos
    """
    current_exp_file = os.path.join(models_dir, '.current_experiment.txt')
    if os.path.exists(current_exp_file):
        os.remove(current_exp_file)
        logger.info("Contador de experimentos reiniciado. Próxima ejecución usará nuevo número.")


class ModelTrainer:
    """
    Clase para entrenamiento y optimización de modelos de clasificación.
    
    Implementa búsqueda de hiperparámetros con Optuna, validación cruzada
    estratificada y seguimiento de experimentos con MLflow.
    """
    
    # Mapeo de nombres de modelos
    MODEL_NAMES = {
        'LR': 'Regresión Logística',
        'RF': 'Random Forest',
        'XGB': 'XGBoost',
        'LGBM': 'LightGBM'
    }
    
    # Configuración de validación cruzada
    CV_FOLDS = 10
    RANDOM_STATE = 42
    
    def __init__(self, input_dir: str, model_name: str, models_dir: str, n_trials: int = 50):
        """
        Inicializa el entrenador de modelos.
        
        Args:
            input_dir: Directorio con datos de entrenamiento y prueba
            model_name: Código del modelo a entrenar ('LR', 'RF', 'XGB', 'LGBM')
            models_dir: Directorio donde se guardarán los modelos entrenados
            n_trials: Número de iteraciones para la optimización con Optuna
        """
        self.input_dir = input_dir
        self.model_name = model_name.upper()
        self.model_full_name = self.MODEL_NAMES.get(self.model_name, self.model_name)
        self.models_dir = models_dir
        self.n_trials = n_trials
        
        # Datasets
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
        # Mejores parámetros y modelo
        self.best_params = None
        self.best_model = None
        self.metrics = {}
        self.val_metrics = {}
        self.study = None
        self.best_cv_roc_auc = None

        # Predicciones de validación para análisis posterior
        self.y_val_pred = None
        self.y_val_pred_proba = None
        self.y_train_pred = None
        self.y_train_pred_proba = None
        
        # Número de experimento
        self.experiment_number = None
        self.experiment_dir = None
        
    def load_data(self) -> None:
        """
        Carga los datos de entrenamiento y validación desde archivos CSV.
        """
        logger.info(f"Cargando datos desde: {self.input_dir}")
        
        self.X_train = pd.read_csv(os.path.join(self.input_dir, 'X_train.csv'))
        self.y_train = pd.read_csv(os.path.join(self.input_dir, 'y_train.csv')).values.ravel()
        self.X_val = pd.read_csv(os.path.join(self.input_dir, 'X_val.csv'))
        self.y_val = pd.read_csv(os.path.join(self.input_dir, 'y_val.csv')).values.ravel()
        
        total = self.X_train.shape[0] + self.X_val.shape[0]
        logger.info("Datos cargados:")
        logger.info(f"  Entrenamiento: {self.X_train.shape[0]:,} registros ({self.X_train.shape[0]/total*100:.1f}%), {self.X_train.shape[1]} características")
        logger.info(f"  Validación: {self.X_val.shape[0]:,} registros ({self.X_val.shape[0]/total*100:.1f}%), {self.X_val.shape[1]} características")
        
        # Verificar balance en todos los conjuntos
        train_balance = pd.Series(self.y_train).value_counts(normalize=True) * 100
        val_balance = pd.Series(self.y_val).value_counts(normalize=True) * 100
        logger.info(f"  Balance de clases - Train: {train_balance[0]:.1f}% / {train_balance[1]:.1f}%")
        logger.info(f"  Balance de clases - Val: {val_balance[0]:.1f}% / {val_balance[1]:.1f}%")
    
    def create_model(self, params: Dict[str, Any]) -> Any:
        """
        Crea una instancia del modelo con los parámetros especificados.
        
        Args:
            params: Diccionario con hiperparámetros del modelo
            
        Returns:
            Instancia del modelo configurado
        """
        if self.model_name == 'LR':
            return LogisticRegression(
                **params,
                max_iter=1000,
                random_state=self.RANDOM_STATE
            )
        
        elif self.model_name == 'RF':
            return RandomForestClassifier(
                **params,
                random_state=self.RANDOM_STATE,
                n_jobs=-1
            )
        
        elif self.model_name == 'XGB':
            return XGBClassifier(
                **params,
                random_state=self.RANDOM_STATE,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0,
                n_jobs=-1
            )
        
        elif self.model_name == 'LGBM':
            return LGBMClassifier(
                **params,
                random_state=self.RANDOM_STATE,
                verbose=-1,
                n_jobs=-1
            )
        
        else:
            raise ValueError(f"Modelo '{self.model_name}' no soportado. Opciones: {list(self.MODEL_NAMES.keys())}")
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define el espacio de búsqueda de hiperparámetros según el modelo.
        
        Args:
            trial: Objeto trial de Optuna
            
        Returns:
            Diccionario con hiperparámetros sugeridos
        """
        if self.model_name == 'LR':
            return {
                'C': trial.suggest_float('C', 1e-4, 100.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
            }
        
        elif self.model_name == 'RF':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
        
        elif self.model_name == 'XGB':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
            }
        
        elif self.model_name == 'LGBM':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'num_leaves': trial.suggest_int('num_leaves', 10, 150),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
            }
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Función objetivo para la optimización con Optuna.
        
        Utiliza validación cruzada estratificada para evaluar hiperparámetros.
        
        Args:
            trial: Objeto trial de Optuna
            
        Returns:
            Score promedio de validación cruzada (ROC-AUC)
        """
        # Obtener hiperparámetros sugeridos
        params = self.define_search_space(trial)
        
        # Crear modelo
        model = self.create_model(params)
        
        # Validación cruzada estratificada
        cv = StratifiedKFold(n_splits=self.CV_FOLDS, shuffle=True, random_state=self.RANDOM_STATE)
        
        # Usar ROC-AUC como métrica de optimización (apropiada para clasificación balanceada)
        scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=cv, scoring='roc_auc', n_jobs=-1
        )
        
        return scores.mean()
    
    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """
        Ejecuta la optimización de hiperparámetros con Optuna.
        
        Returns:
            Diccionario con los mejores hiperparámetros encontrados
        """
        logger.info(f"Iniciando optimización de hiperparámetros para {self.model_full_name}")
        logger.info("Método de búsqueda: Bayesiana (TPE)")
        logger.info(f"Número de trials: {self.n_trials}")
        logger.info(f"Validación cruzada: {self.CV_FOLDS} folds estratificados")
        logger.info("Métrica de optimización: ROC-AUC")
        
        # Crear estudio de Optuna
        sampler = TPESampler(seed=self.RANDOM_STATE)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f"{self.model_name}_optimization"
        )
        
        # Ejecutar optimización
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=False
        )

        self.study = study
        
        # Obtener mejores parámetros
        self.best_params = study.best_params
        best_score = study.best_value
        self.best_cv_roc_auc = best_score
        
        logger.info("Optimización completada")
        logger.info(f"Mejor ROC-AUC (CV): {best_score:.4f}")
        logger.info("Mejores hiperparámetros:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        
        return self.best_params
    
    def train_final_model(self) -> Any:
        """
        Entrena el modelo final con los mejores hiperparámetros.
        
        Returns:
            Modelo entrenado
        """
        logger.info("Entrenando modelo final con mejores hiperparámetros...")
        
        self.best_model = self.create_model(self.best_params)
        self.best_model.fit(self.X_train, self.y_train)
        
        logger.info("Modelo entrenado exitosamente")
        return self.best_model
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evalúa el modelo en los conjuntos de validación con múltiples métricas.
        
        Returns:
            Diccionario con todas las métricas calculadas
        """
        # Evaluar en validación
        logger.info("Evaluando modelo en conjunto de validación...")
        y_val_pred = self.best_model.predict(self.X_val)
        y_val_pred_proba = self.best_model.predict_proba(self.X_val)[:, 1] if hasattr(self.best_model, 'predict_proba') else y_val_pred

        self.y_val_pred = y_val_pred
        self.y_val_pred_proba = y_val_pred_proba

        # Predicciones de entrenamiento para diagnóstico de sobreajuste
        y_train_pred = self.best_model.predict(self.X_train)
        y_train_pred_proba = self.best_model.predict_proba(self.X_train)[:, 1] if hasattr(self.best_model, 'predict_proba') else y_train_pred
        self.y_train_pred = y_train_pred
        self.y_train_pred_proba = y_train_pred_proba
        
        self.val_metrics = {
            'val_accuracy': accuracy_score(self.y_val, y_val_pred),
            'val_precision': precision_score(self.y_val, y_val_pred, zero_division=0),
            'val_recall': recall_score(self.y_val, y_val_pred, zero_division=0),
            'val_f1_score': f1_score(self.y_val, y_val_pred, zero_division=0),
            'val_roc_auc': roc_auc_score(self.y_val, y_val_pred_proba),
            'val_mcc': matthews_corrcoef(self.y_val, y_val_pred)
        }
        
        
        return self.val_metrics

    def analyze_overfitting_and_problematic_data(self) -> Dict[str, Any]:
        """
        Analiza señales de sobreajuste y detecta muestras problemáticas en validación.

        Returns:
            Diccionario con:
            - overfitting: métricas de train/val y brechas
            - problematic_samples: DataFrame con casos conflictivos de validación
        """
        if self.y_val_pred is None or self.y_val_pred_proba is None:
            raise RuntimeError("Debes ejecutar evaluate_model() antes del análisis de sobreajuste.")

        if self.y_train_pred is None or self.y_train_pred_proba is None:
            raise RuntimeError("No hay predicciones de entrenamiento para análisis de sobreajuste.")

        # Métricas de entrenamiento para comparar contra validación
        train_metrics = {
            'train_accuracy': accuracy_score(self.y_train, self.y_train_pred),
            'train_precision': precision_score(self.y_train, self.y_train_pred, zero_division=0),
            'train_recall': recall_score(self.y_train, self.y_train_pred, zero_division=0),
            'train_f1_score': f1_score(self.y_train, self.y_train_pred, zero_division=0),
            'train_roc_auc': roc_auc_score(self.y_train, self.y_train_pred_proba),
            'train_mcc': matthews_corrcoef(self.y_train, self.y_train_pred)
        }

        gaps = {
            'gap_accuracy': train_metrics['train_accuracy'] - self.val_metrics['val_accuracy'],
            'gap_precision': train_metrics['train_precision'] - self.val_metrics['val_precision'],
            'gap_recall': train_metrics['train_recall'] - self.val_metrics['val_recall'],
            'gap_f1_score': train_metrics['train_f1_score'] - self.val_metrics['val_f1_score'],
            'gap_roc_auc': train_metrics['train_roc_auc'] - self.val_metrics['val_roc_auc'],
            'gap_mcc': train_metrics['train_mcc'] - self.val_metrics['val_mcc']
        }

        f1_gap = gaps['gap_f1_score']
        auc_gap = gaps['gap_roc_auc']
        if f1_gap >= 0.08 or auc_gap >= 0.08:
            overfitting_risk = 'alto'
        elif f1_gap >= 0.04 or auc_gap >= 0.04:
            overfitting_risk = 'medio'
        else:
            overfitting_risk = 'bajo'

        # Casos problemáticos en validación:
        # 1) Errores de alta confianza (posibles outliers/etiquetas ruidosas)
        # 2) Casos muy inciertos (frontera de decisión)
        val_df = self.X_val.copy().reset_index(drop=True)
        val_df['row_id_val'] = np.arange(len(val_df))
        val_df['y_true'] = self.y_val
        val_df['y_pred'] = self.y_val_pred
        val_df['y_pred_proba'] = self.y_val_pred_proba
        val_df['is_error'] = (val_df['y_true'] != val_df['y_pred']).astype(int)
        val_df['pred_confidence'] = np.where(
            val_df['y_pred'] == 1,
            val_df['y_pred_proba'],
            1 - val_df['y_pred_proba']
        )
        val_df['uncertainty'] = np.abs(val_df['y_pred_proba'] - 0.5)

        high_conf_threshold = float(val_df['pred_confidence'].quantile(0.90))
        high_conf_errors = val_df[
            (val_df['is_error'] == 1) & (val_df['pred_confidence'] >= high_conf_threshold)
        ].copy()
        high_conf_errors['problem_type'] = 'error_alta_confianza'
        high_conf_errors['problem_score'] = high_conf_errors['pred_confidence']

        low_uncertainty_threshold = float(val_df['uncertainty'].quantile(0.10))
        uncertain_cases = val_df[
            val_df['uncertainty'] <= low_uncertainty_threshold
        ].copy()
        uncertain_cases['problem_type'] = 'caso_incierto'
        uncertain_cases['problem_score'] = 1 - (2 * uncertain_cases['uncertainty'])

        problematic_samples = pd.concat([high_conf_errors, uncertain_cases], ignore_index=True)
        problematic_samples = problematic_samples.drop_duplicates(subset=['row_id_val'])
        problematic_samples = problematic_samples.sort_values('problem_score', ascending=False)

        analysis = {
            'overfitting': {
                **train_metrics,
                **self.val_metrics,
                **gaps,
                'overfitting_risk': overfitting_risk,
                'high_conf_error_threshold': high_conf_threshold,
                'low_uncertainty_threshold': low_uncertainty_threshold,
                'n_problematic_samples': int(problematic_samples.shape[0])
            },
            'problematic_samples': problematic_samples
        }

        return analysis

    def save_training_artifacts(self) -> str:
        """
        Guarda artefactos de entrenamiento para análisis visual en notebooks.

        Incluye:
        - Historial de trials de Optuna
        - Métricas de validación
        - Mejores hiperparámetros
        - Predicciones de validación (y_true, y_pred, y_pred_proba)

        Returns:
            Ruta al directorio donde se guardaron los artefactos
        """
        if self.experiment_dir is None:
            raise RuntimeError("No se puede guardar artefactos sin directorio de experimento.")

        artifacts_dir = os.path.join(self.experiment_dir, "training_artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # 1) Historial de trials de Optuna
        if self.study is not None:
            trials_df = self.study.trials_dataframe(attrs=("number", "value", "state", "params"))
            trials_path = os.path.join(artifacts_dir, f"optuna_trials_{self.model_name}.csv")
            trials_df.to_csv(trials_path, index=False)
            logger.info(f"Trials de Optuna guardados en: {trials_path}")

        # 2) Métricas de validación
        metrics_payload = {
            "model": self.model_name,
            "model_full_name": self.model_full_name,
            "best_cv_roc_auc": self.best_cv_roc_auc,
            **self.val_metrics,
        }
        metrics_df = pd.DataFrame([metrics_payload])
        metrics_path = os.path.join(artifacts_dir, f"val_metrics_{self.model_name}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Métricas de validación guardadas en: {metrics_path}")

        # 3) Mejores hiperparámetros
        params_path = os.path.join(artifacts_dir, f"best_params_{self.model_name}.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(self.best_params, f, indent=2, ensure_ascii=False)
        logger.info(f"Mejores hiperparámetros guardados en: {params_path}")

        # 4) Predicciones de validación
        if self.y_val_pred is not None and self.y_val_pred_proba is not None:
            preds_df = pd.DataFrame(
                {
                    "y_true": self.y_val,
                    "y_pred": self.y_val_pred,
                    "y_pred_proba": self.y_val_pred_proba,
                }
            )
            preds_path = os.path.join(artifacts_dir, f"val_predictions_{self.model_name}.csv")
            preds_df.to_csv(preds_path, index=False)
            logger.info(f"Predicciones de validación guardadas en: {preds_path}")

        # 5) Diagnóstico de sobreajuste y muestras problemáticas
        analysis = self.analyze_overfitting_and_problematic_data()

        overfit_path = os.path.join(artifacts_dir, f"overfitting_diagnostics_{self.model_name}.json")
        with open(overfit_path, "w", encoding="utf-8") as f:
            json.dump(analysis['overfitting'], f, indent=2, ensure_ascii=False)
        logger.info(f"Diagnóstico de sobreajuste guardado en: {overfit_path}")

        problematic_path = os.path.join(artifacts_dir, f"problematic_validation_samples_{self.model_name}.csv")
        analysis['problematic_samples'].to_csv(problematic_path, index=False)
        logger.info(f"Muestras problemáticas guardadas en: {problematic_path}")

        return artifacts_dir
    
    def log_to_mlflow(self) -> None:
        """
        Registra el experimento, métricas y modelo en MLflow.
        """
        logger.info("Registrando experimento en MLflow...")
        

        with mlflow.start_run(run_name=f"{self.model_name}_Optimized"):
            # Registrar parámetros
            mlflow.log_param("model_type", self.model_name)
            mlflow.log_param("model_name", self.model_full_name)
            mlflow.log_param("n_trials", self.n_trials)
            mlflow.log_param("cv_folds", self.CV_FOLDS)
            
            # Registrar número de experimento si está disponible
            if self.experiment_number is not None:
                mlflow.log_param("experiment_number", self.experiment_number)
                mlflow.log_param("experiment_folder", f"Experimento{self.experiment_number}")
            
            mlflow.log_params(self.best_params)
            
            # Registrar métricas de validación y prueba
            mlflow.log_metrics(self.val_metrics)
            mlflow.log_metrics(self.metrics)
            
            # Registrar modelo según su tipo
            if self.model_name == 'XGB':
                mlflow.xgboost.log_model(self.best_model, "model")
            elif self.model_name == 'LGBM':
                mlflow.lightgbm.log_model(self.best_model, "model")
            else:
                mlflow.sklearn.log_model(self.best_model, "model")
            
            logger.info("Experimento registrado en MLflow")
    
    def save_model(self) -> str:
        """
        Guarda el modelo entrenado en formato joblib en dos ubicaciones:
        1. Carpeta de experimento compartida (ExperimentoN) - todos los modelos del pipeline
        2. Ubicación estándar (models/best_model_{MODELO}.joblib) para compatibilidad con DVC
        
        Returns:
            Ruta del archivo guardado en la carpeta de experimento
        """
        # Asegurar que existe el directorio base de modelos
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Obtener el número de experimento actual (compartido entre todos los modelos del pipeline)
        self.experiment_number = get_current_experiment_number(self.models_dir)
        experiment_folder_name = f"Experimento{self.experiment_number}"
        self.experiment_dir = os.path.join(self.models_dir, experiment_folder_name)
        
        # Crear directorio del experimento
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Guardar modelo en carpeta de experimento (historial)
        experiment_model_path = os.path.join(self.experiment_dir, f"best_model_{self.model_name}.joblib")
        joblib.dump(self.best_model, experiment_model_path)
        logger.info(f"Modelo guardado en carpeta de experimento: {experiment_model_path}")
        
        # Copiar también a la ubicación estándar (para DVC)
        standard_model_path = os.path.join(self.models_dir, f"best_model_{self.model_name}.joblib")
        shutil.copy2(experiment_model_path, standard_model_path)
        logger.info(f"Modelo copiado a ubicación DVC: {standard_model_path}")
        logger.info(f"Experimento actual: {self.experiment_number}")
        
        return experiment_model_path
    
    def print_results_summary(self) -> None:
        """
        Imprime un resumen detallado de los resultados del entrenamiento.
        """
        logger.info("\n" + "="*80)
        logger.info(f"RESULTADOS DEL MODELO: {self.model_full_name}")
        logger.info("="*80)
        
        logger.info("\nMÉTRICAS EN VALIDACIÓN:")
        logger.info(f"  Accuracy:     {self.val_metrics['val_accuracy']:.4f}")
        logger.info(f"  Precision:    {self.val_metrics['val_precision']:.4f}")
        logger.info(f"  Recall:       {self.val_metrics['val_recall']:.4f}")
        logger.info(f"  F1-Score:     {self.val_metrics['val_f1_score']:.4f}")
        logger.info(f"  ROC-AUC:      {self.val_metrics['val_roc_auc']:.4f}")
        logger.info(f"  MCC:          {self.val_metrics['val_mcc']:.4f}")
        
        
        logger.info("\nMEJORES HIPERPARÁMETROS:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        
        logger.info("="*80 + "\n")
    
    def run_pipeline(self) -> None:
        """
        Ejecuta el pipeline completo de entrenamiento y evaluación.
        """
        logger.info(f"Iniciando pipeline de entrenamiento para {self.model_full_name}")
        
        # 1. Cargar datos
        self.load_data()
        
        # 2. Optimizar hiperparámetros
        self.optimize_hyperparameters()
        
        # 3. Entrenar modelo final
        self.train_final_model()
        
        # 4. Evaluar modelo
        self.evaluate_model()
        
        # 5. Guardar modelo (antes de MLflow para registrar el número de experimento)
        self.save_model()

        # 6. Guardar artefactos para visualización
        self.save_training_artifacts()
        
        # 7. Registrar en MLflow
        self.log_to_mlflow()
        
        # 8. Mostrar resultados
        self.print_results_summary()
        
        logger.info(f"Pipeline de entrenamiento completado para {self.model_full_name}")


def main():
    """
    Función principal para ejecutar el script desde línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description='Entrenamiento y optimización de modelos para predicción cardiovascular',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
    python train_model.py --input_dir data/processed/model_inputs \\
                          --model XGB \\
                          --models_dir models \\
                          --trials 50
                          
Modelos disponibles:
    LR    - Regresión Logística
    RF    - Random Forest
    XGB   - XGBoost
    LGBM  - LightGBM
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directorio con archivos X_train.csv, y_train.csv, X_val.csv, y_val.csv'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['LR', 'RF', 'XGB', 'LGBM'],
        help='Código del modelo a entrenar'
    )
    parser.add_argument(
        '--models_dir',
        type=str,
        default='models/',
        help='Directorio para guardar el modelo entrenado (default: models/)'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='Número de iteraciones para optimización con Optuna (default: 50)'
    )
    parser.add_argument(
        '--reset_experiment',
        action='store_true',
        help='Reinicia el contador de experimentos para iniciar uno nuevo'
    )
    
    args = parser.parse_args()
    
    # Reiniciar experimento si se solicitó
    if args.reset_experiment:
        reset_experiment_counter(args.models_dir)
        logger.info("Contador de experimentos reiniciado. Se creará un nuevo experimento.")
    
    # Validar que el directorio de entrada existe
    if not os.path.exists(args.input_dir):
        logger.error(f"El directorio de entrada no existe: {args.input_dir}")
        return
    
    # Validar que los archivos requeridos existen
    required_files = ['X_train.csv', 'y_train.csv', 'X_val.csv', 'y_val.csv']
    for filename in required_files:
        filepath = os.path.join(args.input_dir, filename)
        if not os.path.exists(filepath):
            logger.error(f"Archivo requerido no encontrado: {filepath}")
            return
    
    # Ejecutar pipeline
    trainer = ModelTrainer(
        input_dir=args.input_dir,
        model_name=args.model,
        models_dir=args.models_dir,
        n_trials=args.trials
    )
    trainer.run_pipeline()


if __name__ == "__main__":
    main()

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
import argparse
import logging
import os
import joblib
import warnings
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
    confusion_matrix
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
    CV_FOLDS = 5
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
        self.X_test = None
        self.y_test = None
        
        # Mejores parámetros y modelo
        self.best_params = None
        self.best_model = None
        self.metrics = {}
        self.val_metrics = {}
        
    def load_data(self) -> None:
        """
        Carga los datos de entrenamiento, validación y prueba desde archivos CSV.
        """
        logger.info(f"Cargando datos desde: {self.input_dir}")
        
        self.X_train = pd.read_csv(os.path.join(self.input_dir, 'X_train.csv'))
        self.y_train = pd.read_csv(os.path.join(self.input_dir, 'y_train.csv')).values.ravel()
        self.X_val = pd.read_csv(os.path.join(self.input_dir, 'X_val.csv'))
        self.y_val = pd.read_csv(os.path.join(self.input_dir, 'y_val.csv')).values.ravel()
        self.X_test = pd.read_csv(os.path.join(self.input_dir, 'X_test.csv'))
        self.y_test = pd.read_csv(os.path.join(self.input_dir, 'y_test.csv')).values.ravel()
        
        total = self.X_train.shape[0] + self.X_val.shape[0] + self.X_test.shape[0]
        logger.info("Datos cargados:")
        logger.info(f"  Entrenamiento: {self.X_train.shape[0]:,} registros ({self.X_train.shape[0]/total*100:.1f}%), {self.X_train.shape[1]} características")
        logger.info(f"  Validación: {self.X_val.shape[0]:,} registros ({self.X_val.shape[0]/total*100:.1f}%), {self.X_val.shape[1]} características")
        logger.info(f"  Prueba: {self.X_test.shape[0]:,} registros ({self.X_test.shape[0]/total*100:.1f}%), {self.X_test.shape[1]} características")
        
        # Verificar balance en todos los conjuntos
        train_balance = pd.Series(self.y_train).value_counts(normalize=True) * 100
        val_balance = pd.Series(self.y_val).value_counts(normalize=True) * 100
        test_balance = pd.Series(self.y_test).value_counts(normalize=True) * 100
        logger.info(f"  Balance de clases - Train: {train_balance[0]:.1f}% / {train_balance[1]:.1f}%")
        logger.info(f"  Balance de clases - Val: {val_balance[0]:.1f}% / {val_balance[1]:.1f}%")
        logger.info(f"  Balance de clases - Test: {test_balance[0]:.1f}% / {test_balance[1]:.1f}%")
    
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
                verbosity=0
            )
        
        elif self.model_name == 'LGBM':
            return LGBMClassifier(
                **params,
                random_state=self.RANDOM_STATE,
                verbose=-1
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
                'C': trial.suggest_float('C', 1e-4, 10.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
            }
        
        elif self.model_name == 'RF':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 10, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
        
        elif self.model_name == 'XGB':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5)
            }
        
        elif self.model_name == 'LGBM':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Función objetivo para la optimización con Optuna.
        
        Utiliza validación cruzada estratificada para evaluar hiperparámetros.
        
        Args:
            trial: Objeto trial de Optuna
            
        Returns:
            Score promedio de validación cruzada (ROC AUC)
        """
        # Obtener hiperparámetros sugeridos
        params = self.define_search_space(trial)
        
        # Crear modelo
        model = self.create_model(params)
        
        # Validación cruzada estratificada
        cv = StratifiedKFold(n_splits=self.CV_FOLDS, shuffle=True, random_state=self.RANDOM_STATE)
        
        # Usar ROC AUC como métrica de optimización (apropiada para clasificación balanceada)
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
        logger.info("Métrica de optimización: F1-Score")
        
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
        
        # Obtener mejores parámetros
        self.best_params = study.best_params
        best_score = study.best_value
        
        logger.info("Optimización completada")
        logger.info(f"Mejor F1-Score (CV): {best_score:.4f}")
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
        Evalúa el modelo en los conjuntos de validación y prueba con múltiples métricas.
        
        Returns:
            Diccionario con todas las métricas calculadas
        """
        # Evaluar en validación
        logger.info("Evaluando modelo en conjunto de validación...")
        y_val_pred = self.best_model.predict(self.X_val)
        y_val_pred_proba = self.best_model.predict_proba(self.X_val)[:, 1] if hasattr(self.best_model, 'predict_proba') else y_val_pred
        
        self.val_metrics = {
            'val_accuracy': accuracy_score(self.y_val, y_val_pred),
            'val_precision': precision_score(self.y_val, y_val_pred, zero_division=0),
            'val_recall': recall_score(self.y_val, y_val_pred, zero_division=0),
            'val_f1_score': f1_score(self.y_val, y_val_pred, zero_division=0),
            'val_roc_auc': roc_auc_score(self.y_val, y_val_pred_proba),
            'val_mcc': matthews_corrcoef(self.y_val, y_val_pred)
        }
        
        # Evaluar en prueba
        logger.info("Evaluando modelo en conjunto de prueba...")
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1] if hasattr(self.best_model, 'predict_proba') else y_pred
        
        # Calcular métricas de prueba
        self.metrics = {
            'test_accuracy': accuracy_score(self.y_test, y_pred),
            'test_precision': precision_score(self.y_test, y_pred, zero_division=0),
            'test_recall': recall_score(self.y_test, y_pred, zero_division=0),
            'test_f1_score': f1_score(self.y_test, y_pred, zero_division=0),
            'test_roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'test_mcc': matthews_corrcoef(self.y_test, y_pred)
        }
        
        # Matriz de confusión (solo para test)
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        self.metrics['true_negatives'] = int(tn)
        self.metrics['false_positives'] = int(fp)
        self.metrics['false_negatives'] = int(fn)
        self.metrics['true_positives'] = int(tp)
        
        # Métricas adicionales derivadas
        self.metrics['test_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        self.metrics['test_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return self.metrics
    
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
        Guarda el modelo entrenado en formato joblib.
        
        Returns:
            Ruta del archivo guardado
        """
        os.makedirs(self.models_dir, exist_ok=True)
        model_path = os.path.join(self.models_dir, f"best_model_{self.model_name}.joblib")
        
        joblib.dump(self.best_model, model_path)
        logger.info(f"Modelo guardado en: {model_path}")
        
        return model_path
    
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
        
        logger.info("\nMÉTRICAS EN PRUEBA (TEST):")
        logger.info(f"  Accuracy:     {self.metrics['test_accuracy']:.4f}")
        logger.info(f"  Precision:    {self.metrics['test_precision']:.4f}")
        logger.info(f"  Recall:       {self.metrics['test_recall']:.4f}")
        logger.info(f"  F1-Score:     {self.metrics['test_f1_score']:.4f}")
        logger.info(f"  ROC-AUC:      {self.metrics['test_roc_auc']:.4f}")
        logger.info(f"  MCC:          {self.metrics['test_mcc']:.4f}")
        
        logger.info("\nMÉTRICAS ADICIONALES (TEST):")
        logger.info(f"  Sensitivity:  {self.metrics['test_sensitivity']:.4f}")
        logger.info(f"  Specificity:  {self.metrics['test_specificity']:.4f}")
        
        logger.info("\nMATRIZ DE CONFUSIÓN (TEST):")
        logger.info(f"  Verdaderos Negativos:  {self.metrics['true_negatives']:,}")
        logger.info(f"  Falsos Positivos:      {self.metrics['false_positives']:,}")
        logger.info(f"  Falsos Negativos:      {self.metrics['false_negatives']:,}")
        logger.info(f"  Verdaderos Positivos:  {self.metrics['true_positives']:,}")
        
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
        
        # 5. Registrar en MLflow
        self.log_to_mlflow()
        
        # 6. Guardar modelo
        self.save_model()
        
        # 7. Mostrar resultados
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
        help='Directorio con archivos X_train.csv, y_train.csv, X_val.csv, y_val.csv, X_test.csv, y_test.csv'
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
    
    args = parser.parse_args()
    
    # Validar que el directorio de entrada existe
    if not os.path.exists(args.input_dir):
        logger.error(f"El directorio de entrada no existe: {args.input_dir}")
        return
    
    # Validar que los archivos requeridos existen
    required_files = ['X_train.csv', 'y_train.csv', 'X_val.csv', 'y_val.csv', 'X_test.csv', 'y_test.csv']
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

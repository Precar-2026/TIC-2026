"""
Script de Entrenamiento y Optimización de Modelos - Enfermedades Cardiovasculares

Utiliza Optuna para la búsqueda de hiperparámetros y MLflow para el
seguimiento de experimentos (Experiment Tracking) y registro de modelos.

Autor: Jhandry U
Fecha: Marzo 2026
"""

import pandas as pd
import argparse
import logging
import os
import joblib
import optuna
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, accuracy_score

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurar el servidor local de MLflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Tesis_Cardio_Prediccion")

class ModelTrainer:
    def __init__(self, input_dir: str, model_name: str, models_dir: str, n_trials: int = 20):
        self.input_dir = input_dir
        self.model_name = model_name.upper()
        self.models_dir = models_dir
        self.n_trials = n_trials
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
    def load_data(self):
        logger.info(f"Cargando datos desde: {self.input_dir}")
        self.X_train = pd.read_csv(os.path.join(self.input_dir, 'X_train.csv'))
        self.y_train = pd.read_csv(os.path.join(self.input_dir, 'y_train.csv')).values.ravel()
        self.X_test = pd.read_csv(os.path.join(self.input_dir, 'X_test.csv'))
        self.y_test = pd.read_csv(os.path.join(self.input_dir, 'y_test.csv')).values.ravel()

    def objective(self, trial):
        """Función objetivo para Optuna dependiendo del modelo."""
        if self.model_name == 'LR':
            C = trial.suggest_float('C', 1e-4, 10.0, log=True)
            solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
            clf = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
            
        elif self.model_name == 'RF':
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 5, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                         min_samples_split=min_samples_split, random_state=42, n_jobs=-1)
            
        elif self.model_name == 'XGB':
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
            clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                learning_rate=learning_rate, random_state=42, use_label_encoder=False, eval_metric='logloss')
            
        elif self.model_name == 'LGBM':
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            num_leaves = trial.suggest_int('num_leaves', 20, 100)
            learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
            clf = LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves, 
                                 learning_rate=learning_rate, random_state=42)
        else:
            raise ValueError(f"Modelo {self.model_name} no soportado.")

        # Entrenar y evaluar rápido (Usamos F1 como métrica a optimizar internamente)
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        return f1_score(self.y_test, preds)

    def run(self):
        self.load_data()
        
        logger.info(f"--- Iniciando optimización con Optuna para {self.model_name} ({self.n_trials} trials) ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        best_params = study.best_params
        logger.info(f"Mejores parámetros encontrados: {best_params}")

        # Entrenar el modelo final con los mejores parámetros y registrar en MLflow
        logger.info("Entrenando modelo final y registrando en MLflow...")
        
        # Iniciar corrida de MLflow
        with mlflow.start_run(run_name=f"{self.model_name}_Optimized"):
            # 1. Instanciar el mejor modelo
            if self.model_name == 'LR':
                best_clf = LogisticRegression(**best_params, max_iter=1000, random_state=42)
            elif self.model_name == 'RF':
                best_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
            elif self.model_name == 'XGB':
                best_clf = XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
            elif self.model_name == 'LGBM':
                best_clf = LGBMClassifier(**best_params, random_state=42)

            # 2. Entrenar
            best_clf.fit(self.X_train, self.y_train)
            
            # 3. Predecir
            preds = best_clf.predict(self.X_test)
            probs = best_clf.predict_proba(self.X_test)[:, 1] if hasattr(best_clf, "predict_proba") else preds
            
            # 4. Calcular métricas clave para la tesis
            metrics = {
                "f1_score": f1_score(self.y_test, preds),
                "mcc": matthews_corrcoef(self.y_test, preds),
                "auc_roc": roc_auc_score(self.y_test, probs),
                "accuracy": accuracy_score(self.y_test, preds)
            }
            
            # 5. Loguear todo en MLflow
            mlflow.log_params(best_params)
            mlflow.log_param("model_type", self.model_name)
            mlflow.log_metrics(metrics)
            
            # Guardar el modelo en MLflow y localmente
            os.makedirs(self.models_dir, exist_ok=True)
            model_path = os.path.join(self.models_dir, f"best_model_{self.model_name}.joblib")
            joblib.dump(best_clf, model_path)
            
            if self.model_name == 'XGB':
                mlflow.xgboost.log_model(best_clf, "model")
            elif self.model_name == 'LGBM':
                mlflow.lightgbm.log_model(best_clf, "model")
            else:
                mlflow.sklearn.log_model(best_clf, "model")

            logger.info("========================================")
            logger.info(f"RESULTADOS FINALES {self.model_name}:")
            logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"MCC:      {metrics['mcc']:.4f}")
            logger.info(f"AUC-ROC:  {metrics['auc_roc']:.4f}")
            logger.info("========================================")
            logger.info(f"Modelo físico guardado en: {model_path}")

def main():
    parser = argparse.ArgumentParser(description='Entrenamiento de Modelos con Optuna y MLflow')
    parser.add_argument('--input_dir', type=str, required=True, help='Directorio con X_train, y_train, etc.')
    parser.add_argument('--model', type=str, required=True, choices=['LR', 'RF', 'XGB', 'LGBM'], help='Modelo a entrenar')
    parser.add_argument('--models_dir', type=str, default='models/', help='Directorio para guardar el .joblib final')
    parser.add_argument('--trials', type=int, default=20, help='Número de iteraciones para Optuna')
    args = parser.parse_args()
    
    trainer = ModelTrainer(args.input_dir, args.model, args.models_dir, args.trials)
    trainer.run()

if __name__ == "__main__":
    main()
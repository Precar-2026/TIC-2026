"""
Script de Selección de Características - RFECV

Este módulo toma los datos procesados por 'build_features.py' y aplica 
Eliminación Recursiva de Características (RFE) con Validación Cruzada 
para determinar el subconjunto óptimo de variables predictoras.

Autor: Asesor Tesis Codigo
"""

import pandas as pd
import argparse
import logging
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def perform_rfecv(X_train: pd.DataFrame, y_train: pd.Series, output_dir: str) -> list:
    """
    Ejecuta RFECV y guarda el gráfico de rendimiento y la lista de características.
    """
    logger.info("Iniciando RFECV con LightGBM. Esto puede tomar unos minutos...")
    
    estimator = LGBMClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring='roc_auc',
        min_features_to_select=5,
        n_jobs=-1
    )
    
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.support_].tolist()
    logger.info(f"RFECV Completado. Características óptimas: {selector.n_features_} de {X_train.shape[1]}")
    
    # Generar y guardar gráfico
    plt.figure(figsize=(10, 6))
    plt.title('RFECV: Número de características vs. ROC AUC', fontsize=14)
    plt.xlabel('Número de características seleccionadas')
    plt.ylabel('ROC AUC (Validación Cruzada)')
    
    mean_scores = selector.cv_results_['mean_test_score']
    plt.plot(range(1, len(mean_scores) + 1), mean_scores, marker='o', color='b')
    plt.axvline(x=selector.n_features_, color='r', linestyle='--', label=f'Óptimo: {selector.n_features_} vars')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'rfecv_results.png')
    plt.savefig(plot_path)
    logger.info(f"Gráfico de RFECV guardado en: {plot_path}")
    
    return selected_features

def main():
    parser = argparse.ArgumentParser(description='Selección de características con RFECV')
    parser.add_argument('--input_dir', type=str, required=True, help='Directorio con X_train.csv y y_train.csv')
    parser.add_argument('--output_dir', type=str, required=True, help='Directorio para guardar los datasets finales')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Cargar datos de entrenamiento escalados
    logger.info("Cargando datasets de entrenamiento...")
    X_train = pd.read_csv(os.path.join(args.input_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(args.input_dir, 'y_train.csv')).iloc[:, 0]
    
    X_val = pd.read_csv(os.path.join(args.input_dir, 'X_val.csv'))
    X_test = pd.read_csv(os.path.join(args.input_dir, 'X_test.csv'))

    y_val_path = os.path.join(args.input_dir, 'y_val.csv')
    y_test_path = os.path.join(args.input_dir, 'y_test.csv')
    y_val = None
    y_test = None
    if os.path.exists(y_val_path):
        y_val = pd.read_csv(y_val_path).iloc[:, 0]
    else:
        logger.warning("y_val.csv no existe en el input_dir. Se omitira en la salida.")
    if os.path.exists(y_test_path):
        y_test = pd.read_csv(y_test_path).iloc[:, 0]
    else:
        logger.warning("y_test.csv no existe en el input_dir. Se omitira en la salida.")

    # 2. Ejecutar RFECV
    optimal_features = perform_rfecv(X_train, y_train, args.output_dir)

    # 3. Guardar la lista de características para reproducibilidad
    joblib.dump(optimal_features, os.path.join(args.output_dir, 'selected_features.pkl'))
    logger.info("Lista de características seleccionadas guardada en selected_features.pkl")

    # 4. Filtrar y guardar los datasets finales
    logger.info("Filtrando datasets con las características seleccionadas...")
    X_train_selected = X_train[optimal_features]
    X_val_selected = X_val[optimal_features]
    X_test_selected = X_test[optimal_features]

    X_train_selected.to_csv(os.path.join(args.output_dir, 'X_train.csv'), index=False)
    X_val_selected.to_csv(os.path.join(args.output_dir, 'X_val.csv'), index=False)
    X_test_selected.to_csv(os.path.join(args.output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(args.output_dir, 'y_train.csv'), index=False, header=True)
    if y_val is not None:
        y_val.to_csv(os.path.join(args.output_dir, 'y_val.csv'), index=False, header=True)
    if y_test is not None:
        y_test.to_csv(os.path.join(args.output_dir, 'y_test.csv'), index=False, header=True)
    
    logger.info("Datasets finales listos para el entrenamiento de modelos.")

if __name__ == "__main__":
    main()
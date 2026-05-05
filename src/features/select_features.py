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
import warnings
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
LOG_SEPARATOR = "=" * 80


def log_banner(title: str) -> None:
    logger.info(LOG_SEPARATOR)
    logger.info(title)
    logger.info(LOG_SEPARATOR)


def perform_rfecv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    figures_dir: str,
) -> tuple[list[str], str]:
    """
    Ejecuta RFECV y guarda el gráfico de rendimiento y la lista de características.
    """
    logger.info("Ejecutando RFECV con LightGBM (CV=5, métrica=ROC AUC).")
    
    estimator = LGBMClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        verbosity=-1,
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring='roc_auc',
        min_features_to_select=10,
        n_jobs=-1
    )
    
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.support_].tolist()
    
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
    
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, 'rfecv_results.png')
    plt.savefig(plot_path)
    plt.close()
    
    return selected_features, plot_path

def main():
    parser = argparse.ArgumentParser(description='Selección de características con RFECV')
    parser.add_argument('--input_dir', type=str, required=True, help='Directorio con X_train.csv y y_train.csv')
    parser.add_argument('--output_dir', type=str, required=True, help='Directorio para guardar los datasets finales')
    parser.add_argument(
        '--figures_dir',
        type=str,
        default=os.path.join('report', 'figures'),
        help='Directorio para guardar la figura de RFECV'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    warnings.filterwarnings(
        "ignore",
        message=r".*LightGBM.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*No further splits with positive gain.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*X does not have valid feature names.*",
        category=UserWarning,
    )
    logging.getLogger("lightgbm").setLevel(logging.ERROR)

    log_banner("INICIANDO SELECCIÓN DE CARACTERÍSTICAS (RFECV)")
    logger.info("Entrada: %s | Salida: %s", args.input_dir, args.output_dir)

    # 1. Cargar datos de entrenamiento escalados
    logger.info("Cargando datasets de entrenamiento...")
    X_train = pd.read_csv(os.path.join(args.input_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(args.input_dir, 'y_train.csv')).iloc[:, 0]
    
    X_val = pd.read_csv(os.path.join(args.input_dir, 'X_val.csv'))
    X_test = pd.read_csv(os.path.join(args.input_dir, 'X_test.csv'))

    logger.info(
        "Datos cargados: X_train=%s, X_val=%s, X_test=%s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )

    y_val_path = os.path.join(args.input_dir, 'y_val.csv')
    y_test_path = os.path.join(args.input_dir, 'y_test.csv')
    y_val = None
    y_test = None
    if os.path.exists(y_val_path):
        y_val = pd.read_csv(y_val_path).iloc[:, 0]
    else:
        logger.warning("No se encontró y_val.csv en input_dir; se omitirá en la salida.")
    if os.path.exists(y_test_path):
        y_test = pd.read_csv(y_test_path).iloc[:, 0]
    else:
        logger.warning("No se encontró y_test.csv en input_dir; se omitirá en la salida.")

    # 2. Ejecutar RFECV
    optimal_features, plot_path = perform_rfecv(
        X_train,
        y_train,
        args.figures_dir,
    )
    logger.info(
        "RFECV completado: %d de %d variables seleccionadas.",
        len(optimal_features),
        X_train.shape[1],
    )

    # 3. Guardar la lista de características para reproducibilidad
    selected_features_path = os.path.join(args.output_dir, 'selected_features.pkl')
    joblib.dump(optimal_features, selected_features_path)
    logger.info(
        "Artefactos guardados: %s, %s",
        selected_features_path,
        plot_path,
    )

    # 4. Filtrar y guardar los datasets finales
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

    logger.info(
        "Datasets filtrados exportados en: %s (X_train, X_val, X_test, y_*)",
        args.output_dir,
    )
    log_banner("SELECCIÓN DE CARACTERÍSTICAS COMPLETADA")

if __name__ == "__main__":
    main()
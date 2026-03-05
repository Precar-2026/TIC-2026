"""
Script de Ingeniería de Características - Enfermedades Cardiovasculares

Este módulo implementa el pipeline de ingeniería de características para el dataset
de enfermedades cardiovasculares, basándose en los hallazgos del Análisis Exploratorio
de Datos (EDA). Realiza:

1. Selección de variables basada en análisis de multicolinealidad
2. División estratificada en conjuntos de entrenamiento y prueba
3. Normalización apropiada para evitar data leakage
4. Preparación de datos para modelado predictivo

Decisiones clave basadas en EDA:
- No se aplica sobremuestreo (SMOTE): dataset naturalmente balanceado (50-50)
- Variables ordinales mantenidas: cholesterol y gluc conservan orden natural (1 < 2 < 3)
- Variables redundantes eliminadas: age, weight, height (reemplazadas por edad_años e imc)

Autor: Jhandry U
Fecha: Marzo 2026
"""

import pandas as pd
import argparse
import logging
import os
import joblib
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Clase para realizar ingeniería de características del dataset cardiovascular.
    
    Implementa selección de variables, división de datos y normalización basadas
    en hallazgos del EDA y mejores prácticas para evitar data leakage.
    """
    
    # Variables seleccionadas para el modelo (basadas en EDA)
    SELECTED_FEATURES = [
        # Variables derivadas (más interpretables que las originales)
        'edad_años',      # Conversión de age en días a años
        'imc',            # Índice de Masa Corporal (reemplaza weight y height)
        
        # Variables clínicas (alta correlación con objetivo)
        'ap_hi',          # Presión arterial sistólica
        'ap_lo',          # Presión arterial diastólica
        'cholesterol',    # Nivel de colesterol (ordinal: 1, 2, 3)
        'gluc',           # Nivel de glucosa (ordinal: 1, 2, 3)
        
        # Variables demográficas y de estilo de vida
        'gender',         # Sexo (1: mujer, 2: hombre)
        'smoke',          # Tabaquismo (0: no, 1: sí)
        'alco',           # Consumo de alcohol (0: no, 1: sí)
        'active'          # Actividad física (0: no, 1: sí)
    ]
    
    TARGET_VARIABLE = 'cardio'
    VAL_SIZE = 0.09  # 9% validación
    TEST_SIZE = 0.10  # 10% prueba
    RANDOM_STATE = 42
    
    def __init__(self, input_path: str, output_dir: str, scaler_path: str):
        """
        Inicializa el ingeniero de características.
        
        Args:
            input_path: Ruta del archivo CSV con datos limpios
            output_dir: Directorio donde se guardarán los conjuntos procesados
            scaler_path: Ruta donde se guardará el objeto StandardScaler
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.scaler_path = scaler_path
        self.df = None
        self.feature_stats = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Carga el dataset limpio desde la ruta especificada.
        
        Returns:
            DataFrame con los datos cargados
        """
        logger.info(f"Cargando datos limpios desde: {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        logger.info(f"Dataset cargado: {self.df.shape[0]:,} registros, {self.df.shape[1]} variables")
        return self.df
    
    def validate_data_quality(self) -> None:
        """
        Valida la calidad de los datos cargados.
        
        Verifica ausencia de valores nulos, duplicados y presencia de variables requeridas.
        """
        logger.info("Validando calidad de datos...")
        
        # Verificar valores nulos
        n_nulls = self.df.isnull().sum().sum()
        if n_nulls > 0:
            logger.warning(f"Se encontraron {n_nulls} valores nulos")
        else:
            logger.info("No se encontraron valores nulos")
        
        # Verificar duplicados
        n_duplicates = self.df.duplicated().sum()
        if n_duplicates > 0:
            logger.warning(f"Se encontraron {n_duplicates} registros duplicados")
        else:
            logger.info("No se encontraron registros duplicados")
        
        # Verificar presencia de variables requeridas
        required_vars = self.SELECTED_FEATURES + [self.TARGET_VARIABLE]
        missing_vars = [var for var in required_vars if var not in self.df.columns]
        
        if missing_vars:
            raise ValueError(f"Variables faltantes en el dataset: {missing_vars}")
        
        logger.info("Validación de calidad completada")
    
    def select_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Selecciona las variables relevantes para el modelado.
        
        Elimina variables redundantes identificadas en el análisis de multicolinealidad:
        - age: Reemplazada por edad_años
        - weight y height: Reemplazadas por imc
        
        Returns:
            Tupla con (X: variables predictoras, y: variable objetivo)
        """
        logger.info("Seleccionando variables para modelado...")
        
        X = self.df[self.SELECTED_FEATURES].copy()
        y = self.df[self.TARGET_VARIABLE].copy()
        
        logger.info(f"Variables seleccionadas: {len(self.SELECTED_FEATURES)}")
        for i, var in enumerate(self.SELECTED_FEATURES, 1):
            logger.info(f"  {i:2d}. {var}")
        
        # Almacenar estadísticas
        self.feature_stats['n_features'] = len(self.SELECTED_FEATURES)
        self.feature_stats['feature_names'] = self.SELECTED_FEATURES
        
        return X, y
    
    def check_class_balance(self, y: pd.Series, dataset_name: str = "Dataset") -> None:
        """
        Verifica y reporta el balance de clases.
        
        Args:
            y: Serie con la variable objetivo
            dataset_name: Nombre del conjunto para logging
        """
        distribution = y.value_counts().sort_index()
        proportions = y.value_counts(normalize=True).sort_index() * 100
        
        logger.info(f"Distribución de clases en {dataset_name}:")
        for class_val, count, prop in zip(distribution.index, distribution.values, proportions.values):
            class_name = "Sin enfermedad" if class_val == 0 else "Con enfermedad"
            logger.info(f"  {class_name}: {count:,} ({prop:.2f}%)")
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Divide el dataset en conjuntos de entrenamiento, validación y prueba.
        
        Utiliza estratificación para mantener la proporción de clases en todos los conjuntos.
        División: 81% entrenamiento, 9% validación, 10% prueba.
        
        Args:
            X: Variables predictoras
            y: Variable objetivo
            
        Returns:
            Tupla con (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Primera división: separar test (10%) del resto (90%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.TEST_SIZE,
            random_state=self.RANDOM_STATE,
            stratify=y
        )
        
        # Segunda división: separar validación (9%) de entrenamiento (81%)
        # val_size_adjusted: 9% del total es 10% del 90% restante
        val_size_adjusted = self.VAL_SIZE / (1 - self.TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.RANDOM_STATE,
            stratify=y_temp
        )
        
        total = len(X)
        logger.info("División completada:")
        logger.info(f"  - Entrenamiento: {X_train.shape[0]:,} registros ({X_train.shape[0]/total*100:.1f}%)")
        logger.info(f"  - Validación: {X_val.shape[0]:,} registros ({X_val.shape[0]/total*100:.1f}%)")
        logger.info(f"  - Prueba: {X_test.shape[0]:,} registros ({X_test.shape[0]/total*100:.1f}%)")
        
        # Verificar balance en todos los conjuntos
        self.check_class_balance(y_train, "Entrenamiento")
        self.check_class_balance(y_val, "Validación")
        self.check_class_balance(y_test, "Prueba")
        
        # Almacenar estadísticas
        self.feature_stats['train_size'] = X_train.shape[0]
        self.feature_stats['val_size'] = X_val.shape[0]
        self.feature_stats['test_size'] = X_test.shape[0]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
        """
        Normaliza las variables utilizando StandardScaler.
        
        Ajusta el escalador únicamente con el conjunto de entrenamiento y aplica
        la misma transformación a validación y prueba para evitar data leakage.
        
        Args:
            X_train: Variables predictoras de entrenamiento
            X_val: Variables predictoras de validación
            X_test: Variables predictoras de prueba
            
        Returns:
            Tupla con (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
        """
        logger.info("Normalizando variables (StandardScaler)...")
        
        scaler = StandardScaler()
        
        # Ajustar y transformar conjunto de entrenamiento
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transformar conjunto de validación (sin ajustar)
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        # Transformar conjunto de prueba (sin ajustar)
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Verificar normalización
        mean_train = X_train_scaled.mean().mean()
        std_train = X_train_scaled.std().mean()
        
        logger.info("Normalización aplicada correctamente")
        logger.info(f"Media del conjunto de entrenamiento: {mean_train:.6f} (esperado: ~0)")
        logger.info(f"Desviación estándar del conjunto de entrenamiento: {std_train:.6f} (esperado: ~1)")
        
        # Almacenar estadísticas
        self.feature_stats['scaling_mean'] = round(mean_train, 6)
        self.feature_stats['scaling_std'] = round(std_train, 6)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    def save_scaler(self, scaler: StandardScaler) -> None:
        """
        Guarda el objeto StandardScaler para uso en inferencia.
        
        Args:
            scaler: Objeto StandardScaler ajustado
        """
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(scaler, self.scaler_path)
        logger.info(f"Escalador guardado en: {self.scaler_path}")
    
    def save_processed_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                           X_test: pd.DataFrame, y_train: pd.Series, 
                           y_val: pd.Series, y_test: pd.Series) -> None:
        """
        Guarda los conjuntos de datos procesados.
        
        Args:
            X_train: Variables predictoras de entrenamiento (normalizadas)
            X_val: Variables predictoras de validación (normalizadas)
            X_test: Variables predictoras de prueba (normalizadas)
            y_train: Variable objetivo de entrenamiento
            y_val: Variable objetivo de validación
            y_test: Variable objetivo de prueba
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        X_train.to_csv(os.path.join(self.output_dir, 'X_train.csv'), index=False)
        X_val.to_csv(os.path.join(self.output_dir, 'X_val.csv'), index=False)
        X_test.to_csv(os.path.join(self.output_dir, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(self.output_dir, 'y_train.csv'), index=False, header=True)
        y_val.to_csv(os.path.join(self.output_dir, 'y_val.csv'), index=False, header=True)
        y_test.to_csv(os.path.join(self.output_dir, 'y_test.csv'), index=False, header=True)
        
        logger.info(f"Datasets procesados guardados en: {self.output_dir}")
        logger.info(f"  - X_train.csv: {X_train.shape}")
        logger.info(f"  - X_val.csv: {X_val.shape}")
        logger.info(f"  - X_test.csv: {X_test.shape}")
        logger.info(f"  - y_train.csv: {y_train.shape}")
        logger.info(f"  - y_val.csv: {y_val.shape}")
        logger.info(f"  - y_test.csv: {y_test.shape}")
    
    def generate_summary_report(self) -> Dict:
        """
        Genera un reporte resumen del proceso de ingeniería de características.
        
        Returns:
            Diccionario con estadísticas del proceso
        """
        total = self.feature_stats['train_size'] + self.feature_stats['val_size'] + self.feature_stats['test_size']
        
        logger.info("\n" + "="*70)
        logger.info("RESUMEN DE INGENIERÍA DE CARACTERÍSTICAS")
        logger.info(f"Variables seleccionadas: {self.feature_stats['n_features']}")
        logger.info(f"Total de registros: {total:,}")
        logger.info(f"Conjunto de entrenamiento: {self.feature_stats['train_size']:,} registros (81%)")
        logger.info(f"Conjunto de validación: {self.feature_stats['val_size']:,} registros (9%)")
        logger.info(f"Conjunto de prueba: {self.feature_stats['test_size']:,} registros (10%)")
        logger.info(f"Normalización - Media: {self.feature_stats['scaling_mean']}")
        logger.info(f"Normalización - Desv. Estándar: {self.feature_stats['scaling_std']}")
        logger.info("-"*70)
        logger.info("DECISIONES CLAVE:")
        logger.info("  - División estratificada: train 81%, validación 9%, test 10%")
        logger.info("  - No se aplicó SMOTE: dataset naturalmente balanceado")
        logger.info("  - Variables ordinales mantenidas sin one-hot encoding")
        logger.info("  - Variables redundantes eliminadas (age, weight, height)")
        logger.info("="*70)
        
        return self.feature_stats
    
    def run_pipeline(self) -> None:
        """
        Ejecuta el pipeline completo de ingeniería de características.
        """
        logger.info("Iniciando pipeline de ingeniería de características")
        
        # 1. Cargar datos
        self.load_data()
        
        # 2. Validar calidad
        self.validate_data_quality()
        
        # 3. Seleccionar variables
        X, y = self.select_features()
        
        # 4. Verificar balance de clases
        self.check_class_balance(y)
        
        # 5. Dividir datos
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # 6. Normalizar
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = self.normalize_features(
            X_train, X_val, X_test
        )
        
        # 7. Guardar escalador
        self.save_scaler(scaler)
        
        # 8. Guardar datos procesados
        self.save_processed_data(
            X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test
        )
        
        # 9. Generar reporte
        self.generate_summary_report()
        
        logger.info("Pipeline de ingeniería de características completado exitosamente")


def main():
    """
    Función principal para ejecutar el script desde línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description='Ingeniería de características para dataset de enfermedades cardiovasculares',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
    python build_features.py --input data/processed/cardio_clean.csv \\
                            --output_dir data/processed/model_inputs \\
                            --scaler_path models/scaler.pkl
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Ruta del archivo CSV con datos limpios'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directorio para guardar los conjuntos de entrenamiento y prueba'
    )
    parser.add_argument(
        '--scaler_path',
        type=str,
        required=True,
        help='Ruta para guardar el objeto StandardScaler'
    )
    
    args = parser.parse_args()
    
    # Validar que el archivo de entrada existe
    if not os.path.exists(args.input):
        logger.error(f"El archivo de entrada no existe: {args.input}")
        return
    
    # Ejecutar pipeline
    engineer = FeatureEngineer(args.input, args.output_dir, args.scaler_path)
    engineer.run_pipeline()


if __name__ == "__main__":
    main()
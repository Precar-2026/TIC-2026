"""
Script Avanzado de Ingeniería de Características - Enfermedades Cardiovasculares

Este módulo implementa un pipeline avanzado de feature engineering basado en:
1. Conocimiento médico cardiovascular
2. Interacciones entre variables clínicas
3. Transformaciones no lineales
4. Selección inteligente de características

Nuevas características derivadas:
- Pulse Pressure: Indicador de rigidez arterial
- Mean Arterial Pressure (MAP): Presión arterial promedio
- Rate Pressure Product (RPP): Índice de trabajo cardíaco
- Categorías de IMC: Clasificación clínica OMS
- Categorías de edad: Grupos de riesgo cardiovascular
- Índices de riesgo: Combinaciones de factores de riesgo
- Interacciones: Productos y ratios entre variables clave

Técnicas aplicadas:
- Feature creation basada en guías médicas (AHA/ACC)
- Polynomial features para variables no lineales
- Binning clínico de variables continuas
- RobustScaler para manejo de outliers
- Selección de features por correlación y varianza

Autor: Jhandry U
Fecha: Marzo 2026
"""

import pandas as pd
import numpy as np
import argparse
import logging
import os
import joblib
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Clase avanzada para ingeniería de características cardiovasculares.
    
    Implementa feature engineering basado en conocimiento médico, creación de
    interacciones, transformaciones no lineales y selección inteligente.
    """
    
    TARGET_VARIABLE = 'cardio'
    VAL_SIZE = 0.09
    TEST_SIZE = 0.10
    RANDOM_STATE = 42
    
    def __init__(self, input_path: str, output_dir: str, scaler_path: str):
        """
        Inicializa el ingeniero de características avanzado.
        
        Args:
            input_path: Ruta del archivo CSV con datos limpios
            output_dir: Directorio donde se guardarán los conjuntos procesados
            scaler_path: Ruta donde se guardará el objeto Scaler
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.scaler_path = scaler_path
        self.df = None
        self.feature_stats = {}
        
    def load_data(self) -> pd.DataFrame:
        """Carga el dataset limpio desde la ruta especificada."""
        logger.info(f"Cargando datos limpios desde: {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        logger.info(f"Dataset cargado: {self.df.shape[0]:,} registros, {self.df.shape[1]} variables")
        return self.df
    
    def validate_data_quality(self) -> None:
        """Valida la calidad de los datos cargados."""
        logger.info("Validando calidad de datos...")
        
        n_nulls = self.df.isnull().sum().sum()
        logger.info(f"Valores nulos: {n_nulls}")
        
        n_duplicates = self.df.duplicated().sum()
        logger.info(f"Registros duplicados: {n_duplicates}")
        
        if n_nulls > 0:
            logger.warning(f"Se encontraron {n_nulls} valores nulos que serán manejados")
        if n_duplicates > 0:
            logger.warning(f"Se encontraron {n_duplicates} duplicados")
        
        logger.info("Validación completada")

    def filter_hypertensive_patients(self) -> None:
        """Filtra para incluir ÚNICAMENTE pacientes hipertensos (≥130/80 mmHg)."""
        logger.info("Aplicando criterio: SOLO pacientes hipertensos...")
        initial_len = len(self.df)
        
        mask_hipertenso = (self.df['ap_hi'] >= 130) | (self.df['ap_lo'] >= 80)
        self.df = self.df[mask_hipertenso].copy()
        
        final_len = len(self.df)
        logger.info(f"Pacientes hipertensos: {final_len:,} (excluidos: {initial_len - final_len:,})")
        self.feature_stats['pacientes_hipertensos'] = final_len
    
    def create_advanced_features(self) -> None:
        """
        Crea características avanzadas basadas en conocimiento médico cardiovascular.
        
        Features cardiovasculares:
        - Pulse Pressure: Rigidez arterial (sistólica - diastólica)
        - Mean Arterial Pressure (MAP): 1/3 sistólica + 2/3 diastólica
        - Rate Pressure Product (RPP): Trabajo cardíaco estimado
        - Presión Diferencial Ratio: Indicador de elasticidad arterial
        
        Features categóricas (basadas en guías clínicas):
        - Categorías IMC (OMS)
        - Grupos de edad cardiovascular
        - Niveles de hipertensión (AHA)
        - Carga de factores de riesgo
        
        Features de interacción:
        - IMC × Edad: Riesgo combinado
        - Presión × IMC: Impacto vascular
       - Colesterol × Glucosa: Síndrome metabólico
        - Estilo de vida combinado: Smoking + Alcohol - Exercise
        """
        logger.info("Creando características cardiovasculares avanzadas...")
        
        # 1. Pulse Pressure (Presión de Pulso) - Indicador de rigidez arterial
        self.df['pulse_pressure'] = self.df['ap_hi'] - self.df['ap_lo']
        
        # 2. Mean Arterial Pressure (Presión Arterial Media)
        self.df['map'] = (self.df['ap_hi'] + 2 * self.df['ap_lo']) / 3
        
        # 3. Rate Pressure Product (Producto de frecuencia-presión) - trabajo cardíaco
        # Estimado usando edad como proxy de frecuencia cardíaca
        estimated_hr = 220 - (self.df['age'] / 365.25)
        self.df['rpp'] = (self.df['ap_hi'] * estimated_hr) / 1000  # Escalado
        
        # 4. Presión Diferencial Ratio (elasticidad arterial)
        self.df['pressure_ratio'] = self.df['pulse_pressure'] / self.df['ap_hi']
        
        # 5. Categorías de IMC (OMS)
        self.df['imc_categoria'] = pd.cut(
            self.df['imc'],
            bins=[0, 18.5, 25, 30, 35, 100],
            labels=[0, 1, 2, 3, 4],  # Bajo peso, Normal, Sobrepeso, Obesidad I, Obesidad II+
            include_lowest=True
        ).astype(int)
        
        # 6. Grupos de edad cardiovascular
        self.df['edad_grupo'] = pd.cut(
            self.df['age'] / 365.25,
            bins=[0, 40, 50, 60, 100],
            labels=[0, 1, 2, 3],  # <40, 40-49, 50-59, 60+
            include_lowest=True
        ).astype(int)
        
        # 7. Categorías de Hipertensión (AHA/ACC)
        # Basado en la presión más alta
        def categorize_hypertension(row):
            if row['ap_hi'] >= 180 or row['ap_lo'] >= 120:
                return 4  # Crisis hipertensiva
            elif row['ap_hi'] >= 140 or row['ap_lo'] >= 90:
                return 3  # Hipertensión Etapa 2
            elif row['ap_hi'] >= 130 or row['ap_lo'] >= 80:
                return 2  # Hipertensión Etapa 1
            else:
                return 1  # Elevada
        
        self.df['hipertension_nivel'] = self.df.apply(categorize_hypertension, axis=1)
        
        # 8. IMC × Edad (riesgo combinado de obesidad y edad)
        self.df['imc_edad_interaction'] = (self.df['imc'] * self.df['age'] / 365.25) / 100
        
        # 9. Presión × IMC (impacto vascular del sobrepeso)
        self.df['presion_imc_interaction'] = (self.df['map'] * self.df['imc']) / 100
        
        # 10. Colesterol × Glucosa (indicador de síndrome metabólico)
        self.df['colesterol_glucosa_interaction'] = self.df['cholesterol'] * self.df['gluc']
        
        # 11. Índice de estilo de vida (combinación de factores modificables)
        # Smoking y alcohol suman (malos), active resta (bueno)
        self.df['lifestyle_score'] = (
            self.df['smoke'] * 2 +  # Fumar pesa más
            self.df['alco'] * 1 -
            self.df['active'] * 1.5  # Ejercicio es muy protector
        )
        
        # 12. Carga total de factores de riesgo
        self.df['risk_factor_count'] = (
            (self.df['cholesterol'] > 1).astype(int) +
            (self.df['gluc'] > 1).astype(int) +
            self.df['smoke'] +
            (self.df['imc'] >= 25).astype(int) +
            (self.df['active'] == 0).astype(int)
        )
        
        # 13. Edad al cuadrado (relación no lineal con riesgo CV)
        self.df['edad_squared'] = (self.df['age'] / 365.25) ** 2
        
        # 14. IMC al cuadrado (impacto exponencial de obesidad)
        self.df['imc_squared'] = self.df['imc'] ** 2
        
        # 15. Log de presión sistólica (normalizar distribución)
        self.df['log_ap_hi'] = np.log1p(self.df['ap_hi'])
        
        # 16. Ratio Colesterol/Glucosa (balance metabólico)
        self.df['col_gluc_ratio'] = self.df['cholesterol'] / (self.df['gluc'] + 0.5)
        
        logger.info("Creadas 16 nuevas características cardiovasculares")
        logger.info("  - 4 features derivadas (pulse pressure, MAP, RPP, pressure ratio)")
        logger.info("  - 3 categorías clínicas (IMC, edad, hipertensión)")
        logger.info("  - 5 interacciones (IMC×edad, presión×IMC, col×gluc, lifestyle, risk count)")
        logger.info("  - 4 transformaciones no lineales (cuadrados, log, ratios)")
        logger.info(f"Total de features: {self.df.shape[1]}")
    
    def select_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Selecciona características finales eliminando variables redundantes y originales.
        
        Estrategia:
        1. Elimina variables originales reemplazadas por derivadas
        2. Mantiene todas las nuevas features creadas
        3. Elimina features con varianza muy baja
        
        Returns:
            Tupla con (X: variables predictoras, y: variable objetivo)
        """
        logger.info("Seleccionando características finales...")
        
        # Variables a eliminar (redundantes o reemplazadas)
        vars_to_drop = [
            # 'age',     # Reemplazada por edad_grupo y edad_squared
            # 'weight',  # Reemplazada por imc
            # 'height',  # Reemplazada por imc
        ]
        
        # Eliminar variables redundantes y target
        X = self.df.drop(columns=vars_to_drop + [self.TARGET_VARIABLE], errors='ignore')
        y = self.df[self.TARGET_VARIABLE].copy()
        
        # Eliminar features con varianza cero (si las hay)
        initial_features = X.shape[1]
        selector = VarianceThreshold(threshold=0.0)
        X_filtered = selector.fit_transform(X)
        
        # Recuperar nombres de features seleccionadas
        selected_feature_names = X.columns[selector.get_support()].tolist()
        X = pd.DataFrame(X_filtered, columns=selected_feature_names, index=X.index)
        
        removed = initial_features - X.shape[1]
        if removed > 0:
            logger.info(f"Eliminadas {removed} features con varianza cero")
        
        logger.info(f"Features seleccionadas: {X.shape[1]}")
        logger.info("Features principales:")
        for i, var in enumerate(selected_feature_names[:10], 1):
            logger.info(f"  {i:2d}. {var}")
        if len(selected_feature_names) > 10:
            logger.info(f"  ... y {len(selected_feature_names) - 10} más")
        
        self.feature_stats['n_features'] = X.shape[1]
        self.feature_stats['feature_names'] = selected_feature_names
        
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
    
    def normalize_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
        """
        Normaliza las variables utilizando RobustScaler.
        
        Utiliza RobustScaler en lugar de StandardScaler porque es menos sensible a
        outliers (usa mediana y rango intercuartil en lugar de media y desviación estándar).
        Esto es importante en datos médicos donde pueden existir valores extremos válidos.
        
        Args:
            X_train: Variables predictoras de entrenamiento
            X_val: Variables predictoras de validación
            X_test: Variables predictoras de prueba
            
        Returns:
            Tupla con (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
        """
        logger.info("Normalizando variables (RobustScaler - resistente a outliers)...")
        
        scaler = RobustScaler()
        
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
        
        # Verificar escalado
        median_train = X_train_scaled.median().mean()
        iqr_train = (X_train_scaled.quantile(0.75) - X_train_scaled.quantile(0.25)).mean()
        
        logger.info("Normalización aplicada correctamente")
        logger.info(f"Mediana del conjunto de entrenamiento: {median_train:.6f} (esperado: ~0)")
        logger.info(f"IQR promedio del conjunto de entrenamiento: {iqr_train:.6f} (esperado: ~1)")
        
        # Almacenar estadísticas
        self.feature_stats['scaling_median'] = round(median_train, 6)
        self.feature_stats['scaling_iqr'] = round(iqr_train, 6)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    def save_scaler(self, scaler: RobustScaler) -> None:
        """
        Guarda el objeto RobustScaler para uso en inferencia.
        
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
        Genera un reporte resumen del proceso de ingeniería de características avanzadas.
        
        Returns:
            Diccionario con estadísticas del proceso
        """
        total = self.feature_stats['train_size'] + self.feature_stats['val_size'] + self.feature_stats['test_size']
        
        logger.info("\n" + "="*80)
        logger.info("RESUMEN DE INGENIERÍA DE CARACTERÍSTICAS AVANZADAS")
        logger.info("="*80)
        logger.info(f"Features finales seleccionadas: {self.feature_stats['n_features']}")
        logger.info(f"Total de registros procesados: {total:,}")
        logger.info(f"   • Entrenamiento: {self.feature_stats['train_size']:,} registros (81%)")
        logger.info(f"   • Validación: {self.feature_stats['val_size']:,} registros (9%)")
        logger.info(f"   • Prueba: {self.feature_stats['test_size']:,} registros (10%)")
        logger.info("Normalización (RobustScaler):")
        logger.info(f"   • Mediana: {self.feature_stats['scaling_median']}")
        logger.info(f"   • IQR: {self.feature_stats['scaling_iqr']}")
        logger.info("-"*80)
        logger.info("TÉCNICAS DE FEATURE ENGINEERING APLICADAS:")
        logger.info("   1. Features Cardiovasculares Derivadas:")
        logger.info("      • Pulse Pressure (rigidez arterial)")
        logger.info("      • Mean Arterial Pressure (MAP)")
        logger.info("      • Rate Pressure Product (trabajo cardíaco)")
        logger.info("      • Pressure Ratio (elasticidad arterial)")
        logger.info("   2. Categorías Clínicas:")
        logger.info("      • IMC (clasificación OMS)")
        logger.info("      • Grupos de edad cardiovascular")
        logger.info("      • Niveles de hipertensión (AHA/ACC)")
        logger.info("   3. Features de Interacción:")
        logger.info("      • IMC × Edad (riesgo combinado)")
        logger.info("      • Presión × IMC (impacto vascular)")
        logger.info("      • Colesterol × Glucosa (síndrome metabólico)")
        logger.info("      • Lifestyle score (factores modificables)")
        logger.info("      • Risk factor count (carga de riesgo)")
        logger.info("   4. Transformaciones No Lineales:")
        logger.info("      • Edad² e IMC² (relaciones cuadráticas)")
        logger.info("      • Log(Presión Sistólica)")
        logger.info("      • Ratios metabólicos")
        logger.info("-"*80)
        logger.info("DECISIONES TÉCNICAS:")
        logger.info("   • RobustScaler: Resistente a outliers médicos")
        logger.info("   • División estratificada: Mantiene balance de clases")
        logger.info("   • Sin SMOTE: Dataset balanceado (Se puede considerar en futuras iteraciones)")
        logger.info("   • Variables ordinales preservadas (cholesterol, gluc)")
        logger.info("="*80 + "\n")
        
        return self.feature_stats
    
    def run_pipeline(self) -> None:
        """
        Ejecuta el pipeline completo de ingeniería de características avanzadas.
        """
        logger.info("="*80)
        logger.info("INICIANDO PIPELINE DE FEATURE ENGINEERING AVANZADO")
        logger.info("="*80)
        
        # 1. Cargar datos
        self.load_data()
        
        # 2. Validar calidad
        self.validate_data_quality()

        # 3. Filtrar pacientes hipertensos
        # self.filter_hypertensive_patients()
        
        # 4. Crear características avanzadas
        self.create_advanced_features()
        
        # 5. Seleccionar variables finales
        X, y = self.select_features()
        
        # 6. Verificar balance de clases
        self.check_class_balance(y)
        
        # 7. Dividir datos
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # 8. Normalizar con RobustScaler
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = self.normalize_features(
            X_train, X_val, X_test
        )
        
        # 9. Guardar escalador
        self.save_scaler(scaler)
        
        # 10. Guardar datos procesados
        self.save_processed_data(
            X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test
        )
        
        # 11. Generar reporte
        self.generate_summary_report()
        
        logger.info("="*80)
        logger.info("PIPELINE DE FEATURE ENGINEERING COMPLETADO EXITOSAMENTE")
        logger.info("="*80)


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
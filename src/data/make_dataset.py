"""
Script de Preprocesamiento de Datos - Enfermedades Cardiovasculares

Este módulo implementa el pipeline de limpieza y preprocesamiento del dataset
de enfermedades cardiovasculares, aplicando filtros de calidad de datos basados
en criterios fisiológicos y médicos identificados durante el análisis exploratorio.

Autor: Jhandry U
Fecha: Marzo 2026
"""

import pandas as pd
import argparse
import logging
import os
from typing import Tuple, Dict

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clase para realizar la limpieza y preprocesamiento del dataset cardiovascular.
    
    Implementa filtros de calidad de datos basados en rangos fisiológicos y 
    criterios médicos establecidos, además de crear variables derivadas relevantes
    para el análisis predictivo.
    """
    
    # Constantes por defecto de rangos fisiológicos basados en EDA y criterios médicos
    ALTURA_MIN_DEFAULT = 130  # cm
    ALTURA_MAX_DEFAULT = 220  # cm
    PESO_MIN_DEFAULT = 30     # kg
    PESO_MAX_DEFAULT = 200    # kg
    PRESION_SISTOLICA_MIN_DEFAULT = 80   # mmHg
    PRESION_SISTOLICA_MAX_DEFAULT = 240  # mmHg
    PRESION_DIASTOLICA_MIN_DEFAULT = 50  # mmHg
    PRESION_DIASTOLICA_MAX_DEFAULT = 140 # mmHg
    DIAS_POR_ANIO = 365.25
    
    def __init__(
        self,
        input_path: str,
        output_path: str,
        altura_min: int = ALTURA_MIN_DEFAULT,
        altura_max: int = ALTURA_MAX_DEFAULT,
        peso_min: int = PESO_MIN_DEFAULT,
        peso_max: int = PESO_MAX_DEFAULT,
        presion_sistolica_min: int = PRESION_SISTOLICA_MIN_DEFAULT,
        presion_sistolica_max: int = PRESION_SISTOLICA_MAX_DEFAULT,
        presion_diastolica_min: int = PRESION_DIASTOLICA_MIN_DEFAULT,
        presion_diastolica_max: int = PRESION_DIASTOLICA_MAX_DEFAULT,
        allow_equal_bp: bool = False,
    ):
        """
        Inicializa el limpiador de datos.
        
        Args:
            input_path: Ruta del archivo CSV con datos crudos
            output_path: Ruta donde se guardará el archivo procesado
            altura_min: Altura mínima aceptada (cm)
            altura_max: Altura máxima aceptada (cm)
            peso_min: Peso mínimo aceptado (kg)
            peso_max: Peso máximo aceptado (kg)
            presion_sistolica_min: Presión sistólica mínima aceptada (mmHg)
            presion_sistolica_max: Presión sistólica máxima aceptada (mmHg)
            presion_diastolica_min: Presión diastólica mínima aceptada (mmHg)
            presion_diastolica_max: Presión diastólica máxima aceptada (mmHg)
            allow_equal_bp: Si True permite ap_hi == ap_lo; si False exige ap_hi > ap_lo
        """
        self.input_path = input_path
        self.output_path = output_path
        self.altura_min = altura_min
        self.altura_max = altura_max
        self.peso_min = peso_min
        self.peso_max = peso_max
        self.presion_sistolica_min = presion_sistolica_min
        self.presion_sistolica_max = presion_sistolica_max
        self.presion_diastolica_min = presion_diastolica_min
        self.presion_diastolica_max = presion_diastolica_max
        self.allow_equal_bp = allow_equal_bp
        self.df_original = None
        self.df_clean = None
        self.cleaning_report = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Carga el dataset desde la ruta especificada.
        
        Returns:
            DataFrame con los datos cargados
        """
        logger.info(f"Cargando datos desde: {self.input_path}")
        self.df_original = pd.read_csv(self.input_path, sep=';')
        logger.info(f"Dataset cargado: {self.df_original.shape[0]:,} registros, {self.df_original.shape[1]} variables")
        return self.df_original
    
    def remove_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina la columna 'id' que no aporta valor predictivo.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame sin la columna 'id'
        """
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
            logger.info("Columna 'id' eliminada")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina registros duplicados exactos.
        
        Args:
            df: DataFrame con posibles duplicados
            
        Returns:
            DataFrame sin duplicados
        """
        n_duplicados = df.duplicated().sum()
        if n_duplicados > 0:
            df = df.drop_duplicates()
            logger.info(f"Registros duplicados eliminados: {n_duplicados:,}")
            self.cleaning_report['duplicados_eliminados'] = n_duplicados
        else:
            logger.info("No se encontraron registros duplicados")
            self.cleaning_report['duplicados_eliminados'] = 0
        return df
    
    def filter_anthropometric_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Filtra registros con medidas antropométricas fuera de rangos fisiológicos.
        
        Aplica filtros para altura y peso basados en rangos médicos establecidos.
        
        Args:
            df: DataFrame a filtrar
            
        Returns:
            Tupla con (DataFrame filtrado, número de registros eliminados)
        """
        mask = (
            (df['height'] >= self.altura_min) & 
            (df['height'] <= self.altura_max) &
            (df['weight'] >= self.peso_min) & 
            (df['weight'] <= self.peso_max)
        )
        n_eliminados = (~mask).sum()
        df_filtered = df[mask].copy()
        
        logger.info(f"Registros eliminados por medidas antropométricas atípicas: {n_eliminados:,}")
        self.cleaning_report['antropometricos_eliminados'] = n_eliminados
        return df_filtered, n_eliminados
    
    def filter_blood_pressure_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Filtra registros con valores de presión arterial atípicos o inválidos.
        
        Aplica tres criterios:
        1. Presión sistólica dentro del rango fisiológico
        2. Presión diastólica dentro del rango fisiológico
        3. Presión sistólica estrictamente mayor que diastólica
        
        Args:
            df: DataFrame a filtrar
            
        Returns:
            Tupla con (DataFrame filtrado, número de registros eliminados)
        """
        mask_rango_sistolica = (df['ap_hi'] >= self.presion_sistolica_min) & (df['ap_hi'] <= self.presion_sistolica_max)
        mask_rango_diastolica = (df['ap_lo'] >= self.presion_diastolica_min) & (df['ap_lo'] <= self.presion_diastolica_max)
        mask_logica = df['ap_hi'] >= df['ap_lo'] if self.allow_equal_bp else df['ap_hi'] > df['ap_lo']
        
        mask_completa = mask_rango_sistolica & mask_rango_diastolica & mask_logica
        n_eliminados = (~mask_completa).sum()
        df_filtered = df[mask_completa].copy()
        
        logger.info(f"Registros eliminados por valores de presión arterial atípicos: {n_eliminados:,}")
        self.cleaning_report['presion_eliminados'] = n_eliminados
        return df_filtered, n_eliminados
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea variables derivadas relevantes para el análisis.
        
        Variables creadas:
        - edad_años: Conversión de edad de días a años
        - imc: Índice de Masa Corporal (kg/m²)
        
        Args:
            df: DataFrame con variables originales
            
        Returns:
            DataFrame con variables derivadas añadidas
        """
        # Edad en años
        df['edad_años'] = (df['age'] / self.DIAS_POR_ANIO).round(1)
        
        # Índice de Masa Corporal
        df['imc'] = df['weight'] / ((df['height'] / 100) ** 2)
        df['imc'] = df['imc'].round(5)
        
        logger.info("Variables derivadas creadas: edad_años, imc")
        return df
    
    def generate_cleaning_report(self) -> Dict:
        """
        Genera un reporte detallado del proceso de limpieza.
        
        Returns:
            Diccionario con estadísticas del proceso de limpieza
        """
        n_original = self.df_original.shape[0]
        n_final = self.df_clean.shape[0]
        n_total_eliminados = n_original - n_final
        porcentaje_retenido = (n_final / n_original) * 100
        
        report = {
            'registros_originales': n_original,
            'registros_finales': n_final,
            'total_eliminados': n_total_eliminados,
            'porcentaje_retenido': round(porcentaje_retenido, 2),
            **self.cleaning_report
        }
        
        logger.info("\n" + "="*60)
        logger.info("REPORTE DE LIMPIEZA")
        logger.info("="*60)
        logger.info(f"Registros originales: {report['registros_originales']:,}")
        logger.info(f"Registros finales: {report['registros_finales']:,}")
        logger.info(f"Total eliminados: {report['total_eliminados']:,} ({100-porcentaje_retenido:.2f}%)")
        logger.info(f"Porcentaje retenido: {report['porcentaje_retenido']}%")
        logger.info("-"*60)
        logger.info(f"  Duplicados: {report.get('duplicados_eliminados', 0):,}")
        logger.info(f"  Datos antropométricos atípicos: {report.get('antropometricos_eliminados', 0):,}")
        logger.info(f"  Presión arterial atípica: {report.get('presion_eliminados', 0):,}")
        logger.info("="*60)
        
        return report
    
    def save_cleaned_data(self) -> None:
        """
        Guarda el dataset limpio en la ruta de salida especificada.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.df_clean.to_csv(self.output_path, index=False)
        logger.info(f"Dataset limpio guardado en: {self.output_path}")
    
    def run_pipeline(self) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de limpieza de datos.
        
        Returns:
            DataFrame limpio y procesado
        """
        logger.info("Iniciando pipeline de limpieza de datos")
        
        # 1. Cargar datos
        df = self.load_data()
        
        # 2. Eliminar columna ID
        df = self.remove_id_column(df)
        
        # 3. Eliminar duplicados
        df = self.remove_duplicates(df)
        
        # 4. Filtrar valores antropométricos atípicos
        df, _ = self.filter_anthropometric_outliers(df)
        
        # 5. Filtrar valores de presión arterial atípicos
        df, _ = self.filter_blood_pressure_outliers(df)
        
        # 6. Crear variables derivadas
        df = self.create_derived_features(df)
        
        # 7. Almacenar resultado
        self.df_clean = df
        
        # 8. Generar reporte
        self.generate_cleaning_report()
        
        # 9. Guardar datos limpios
        self.save_cleaned_data()
        
        logger.info("Pipeline de limpieza completado exitosamente")
        return self.df_clean


def main():
    """
    Función principal para ejecutar el script desde línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description='Limpieza y preprocesamiento del dataset de enfermedades cardiovasculares',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Ruta del archivo CSV con datos crudos'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Ruta donde se guardará el archivo CSV procesado'
    )
    parser.add_argument('--altura_min', type=int, default=DataCleaner.ALTURA_MIN_DEFAULT, help='Altura mínima válida (cm)')
    parser.add_argument('--altura_max', type=int, default=DataCleaner.ALTURA_MAX_DEFAULT, help='Altura máxima válida (cm)')
    parser.add_argument('--peso_min', type=int, default=DataCleaner.PESO_MIN_DEFAULT, help='Peso mínimo válido (kg)')
    parser.add_argument('--peso_max', type=int, default=DataCleaner.PESO_MAX_DEFAULT, help='Peso máximo válido (kg)')
    parser.add_argument('--ap_hi_min', type=int, default=DataCleaner.PRESION_SISTOLICA_MIN_DEFAULT, help='Presión sistólica mínima válida (mmHg)')
    parser.add_argument('--ap_hi_max', type=int, default=DataCleaner.PRESION_SISTOLICA_MAX_DEFAULT, help='Presión sistólica máxima válida (mmHg)')
    parser.add_argument('--ap_lo_min', type=int, default=DataCleaner.PRESION_DIASTOLICA_MIN_DEFAULT, help='Presión diastólica mínima válida (mmHg)')
    parser.add_argument('--ap_lo_max', type=int, default=DataCleaner.PRESION_DIASTOLICA_MAX_DEFAULT, help='Presión diastólica máxima válida (mmHg)')
    parser.add_argument('--allow_equal_bp', action='store_true', help='Permite registros con ap_hi == ap_lo')
    
    args = parser.parse_args()
    
    # Validar que el archivo de entrada existe
    if not os.path.exists(args.input):
        logger.error(f"El archivo de entrada no existe: {args.input}")
        return
    
    # Ejecutar pipeline
    cleaner = DataCleaner(
        input_path=args.input,
        output_path=args.output,
        altura_min=args.altura_min,
        altura_max=args.altura_max,
        peso_min=args.peso_min,
        peso_max=args.peso_max,
        presion_sistolica_min=args.ap_hi_min,
        presion_sistolica_max=args.ap_hi_max,
        presion_diastolica_min=args.ap_lo_min,
        presion_diastolica_max=args.ap_lo_max,
        allow_equal_bp=args.allow_equal_bp,
    )
    cleaner.run_pipeline()


if __name__ == "__main__":
    main()
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pandas_profiling import ProfileReport
import plotly.express as px
from typing import Dict, List, Optional
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExploratoryDataAnalysis:
    def __init__(self, data_dir: str = '../Data/processed', 
                 reports_dir: str = '../../reports'):
        """
        Inicializa la clase de análisis exploratorio de datos.
        
        Args:
            data_dir (str): Directorio que contiene los datos procesados
            reports_dir (str): Directorio para guardar los reportes
        """
        self.data_dir = Path(data_dir)
        self.reports_dir = Path(reports_dir)
        self.figures_dir = self.reports_dir / 'figures'
        self._create_directories()
        self.data: Dict[str, pd.DataFrame] = {}
        
        # Configuración de estilos
        plt.style.use('seaborn')
        sns.set_theme(style="whitegrid")
    
    def _create_directories(self):
        """Crea los directorios necesarios si no existen."""
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def load_data(self, file_path: Optional[str] = None):
        """
        Carga los datos desde un archivo o directorio.
        
        Args:
            file_path (str, optional): Ruta al archivo o directorio. Si es None, usa data_dir.
        """
        path = Path(file_path) if file_path else self.data_dir
        
        if path.is_file():
            self._load_single_file(path)
        else:
            self._load_from_directory(path)
    
    def _load_single_file(self, file_path: Path):
        """Carga un único archivo de datos."""
        try:
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Formato de archivo no soportado: {file_path.suffix}")
            
            self.data[file_path.stem] = df
            logger.info(f"Datos cargados desde {file_path.name}: {df.shape[0]} filas, {df.shape[1]} columnas")
            
        except Exception as e:
            logger.error(f"Error al cargar {file_path}: {str(e)}")
    
    def _load_from_directory(self, directory: Path):
        """Carga todos los archivos de datos de un directorio."""
        file_extensions = ['.parquet', '.csv', '.xlsx', '.xls']
        files = [f for f in directory.iterdir() if f.suffix.lower() in file_extensions]
        
        if not files:
            logger.warning(f"No se encontraron archivos de datos en {directory}")
            return
        
        logger.info(f"Cargando {len(files)} archivos de datos...")
        for file in files:
            self._load_single_file(file)
    
    def generate_profile_report(self, output_file: str = 'profile_report.html'):
        """
        Genera un informe de perfil de datos usando pandas-profiling.
        
        Args:
            output_file (str): Nombre del archivo de salida
        """
        if not self.data:
            logger.warning("No hay datos cargados para generar el informe")
            return
        
        output_path = self.reports_dir / output_file
        
        # Para múltiples conjuntos de datos, creamos un informe para cada uno
        for name, df in self.data.items():
            try:
                profile = ProfileReport(
                    df,
                    title=f"Análisis Exploratorio - {name}",
                    explorative=True,
                    minimal=False
                )
                
                report_path = output_path.parent / f"{output_path.stem}_{name}{output_path.suffix}"
                profile.to_file(report_path)
                logger.info(f"Informe generado: {report_path}")
                
            except Exception as e:
                logger.error(f"Error al generar el informe para {name}: {str(e)}")
    
    def analyze_numerical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza un análisis estadístico de las columnas numéricas.
        
        Args:
            df (pd.DataFrame): DataFrame a analizar
            
        Returns:
            pd.DataFrame: Estadísticas descriptivas
        """
        # Seleccionar solo columnas numéricas
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No se encontraron columnas numéricas para analizar")
            return pd.DataFrame()
        
        # Calcular estadísticas descriptivas
        stats = df[numeric_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
        
        # Calcular valores faltantes
        missing = df[numeric_cols].isnull().sum()
        stats.loc['missing'] = missing
        
        # Calcular asimetría y curtosis
        stats.loc['skew'] = df[numeric_cols].skew()
        stats.loc['kurtosis'] = df[numeric_cols].kurtosis()
        
        return stats.T
    
    def analyze_categorical_columns(self, df: pd.DataFrame, max_categories: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Analiza las columnas categóricas.
        
        Args:
            df (pd.DataFrame): DataFrame a analizar
            max_categories (int): Número máximo de categorías a mostrar
            
        Returns:
            dict: Diccionario con DataFrames de análisis para cada columna categórica
        """
        # Seleccionar columnas categóricas
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
        
        if len(cat_cols) == 0:
            logger.warning("No se encontraron columnas categóricas para analizar")
            return {}
        
        results = {}
        
        for col in cat_cols:
            try:
                # Contar frecuencias
                value_counts = df[col].value_counts(dropna=False)
                value_perc = df[col].value_counts(dropna=False, normalize=True) * 100
                
                # Crear DataFrame con el resumen
                summary = pd.DataFrame({
                    'count': value_counts,
                    'percentage': value_perc
                })
                
                # Ordenar por frecuencia
                summary = summary.sort_values('count', ascending=False)
                
                # Limitar el número de categorías mostradas
                if len(summary) > max_categories:
                    others = pd.DataFrame({
                        'count': [summary['count'][max_categories:].sum()],
                        'percentage': [summary['percentage'][max_categories:].sum()]
                    }, index=['OTROS'])
                    
                    summary = pd.concat([summary.iloc[:max_categories], others])
                
                # Agregar información adicional
                summary.loc['TOTAL'] = [summary['count'].sum(), 100.0]
                summary['cumulative_percentage'] = summary['percentage'].cumsum()
                
                results[col] = summary
                
            except Exception as e:
                logger.error(f"Error al analizar la columna {col}: {str(e)}")
        
        return results
    
    def plot_distributions(self, df: pd.DataFrame, output_prefix: str = 'dist'):
        """
        Genera gráficos de distribución para las columnas numéricas.
        
        Args:
            df (pd.DataFrame): DataFrame a analizar
            output_prefix (str): Prefijo para los archivos de salida
        """
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No se encontraron columnas numéricas para graficar")
            return
        
        # Configuración de la figura
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        # Crear figura para histogramas
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            try:
                ax = axes[i]
                sns.histplot(data=df, x=col, kde=True, ax=ax)
                ax.set_title(f'Distribución de {col}')
                ax.set_xlabel('')
                ax.set_ylabel('Frecuencia')
                
                # Rotar etiquetas del eje x si es necesario
                if len(df[col].unique()) < 10:  # Si hay pocos valores únicos
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                
            except Exception as e:
                logger.error(f"Error al graficar la columna {col}: {str(e)}")
        
        # Ocultar ejes vacíos
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        # Guardar la figura
        output_path = self.figures_dir / f"{output_prefix}_distributions.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Gráficos de distribución guardados en: {output_path}")
    
    def plot_correlations(self, df: pd.DataFrame, output_file: str = 'correlation_heatmap.png'):
        """
        Genera un mapa de calor de correlaciones para las columnas numéricas.
        
        Args:
            df (pd.DataFrame): DataFrame a analizar
            output_file (str): Nombre del archivo de salida
        """
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            logger.warning("Se necesitan al menos 2 columnas numéricas para calcular correlaciones")
            return
        
        # Calcular matriz de correlación
        corr = df[numeric_cols].corr()
        
        # Crear máscara para el triángulo superior
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Configurar el tamaño de la figura
        plt.figure(figsize=(12, 10))
        
        # Generar el mapa de calor
        sns.heatmap(
            corr, 
            mask=mask,
            cmap='coolwarm',
            vmin=-1, vmax=1,
            center=0,
            square=True,
            linewidths=.5,
            annot=True,
            fmt=".2f",
            cbar_kws={"shrink": .8}
        )
        
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        
        # Guardar la figura
        output_path = self.figures_dir / output_file
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Mapa de calor de correlaciones guardado en: {output_path}")
    
    def analyze_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analiza los valores faltantes en el DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame a analizar
            
        Returns:
            pd.DataFrame: Resumen de valores faltantes por columna
        """
        # Calcular valores faltantes
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        
        # Crear DataFrame con los resultados
        missing_df = pd.DataFrame({
            'missing_count': missing,
            'missing_percent': missing_percent,
            'data_type': df.dtypes
        })
        
        # Ordenar por porcentaje de valores faltantes
        missing_df = missing_df.sort_values('missing_percent', ascending=False)
        
        return missing_df
    
    def run_analysis(self, output_prefix: str = 'analysis'):
        """
        Ejecuta el análisis exploratorio completo.
        
        Args:
            output_prefix (str): Prefijo para los archivos de salida
        """
        if not self.data:
            logger.warning("No hay datos cargados para analizar")
            return
        
        logger.info("Iniciando análisis exploratorio de datos...")
        
        for name, df in self.data.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"ANÁLISIS: {name.upper()}")
            logger.info(f"{'='*50}")
            
            # Información básica
            logger.info(f"\n[INFORMACIÓN GENERAL]")
            logger.info(f"Número de filas: {df.shape[0]}")
            logger.info(f"Número de columnas: {df.shape[1]}")
            
            # Análisis de valores faltantes
            logger.info("\n[VALORES FALTANTES]")
            missing_df = self.analyze_missing_values(df)
            logger.info("\nResumen de valores faltantes por columna:")
            logger.info("\n" + str(missing_df[missing_df['missing_count'] > 0]))
            
            # Análisis de columnas numéricas
            logger.info("\n[ANÁLISIS DE COLUMNAS NUMÉRICAS]")
            num_stats = self.analyze_numerical_columns(df)
            if not num_stats.empty:
                logger.info("\nEstadísticas descriptivas de columnas numéricas:")
                logger.info("\n" + str(num_stats))
                
                # Generar gráficos de distribución
                self.plot_distributions(df, f"{output_prefix}_{name}_distributions")
                
                # Generar mapa de calor de correlaciones
                self.plot_correlations(df, f"{output_prefix}_{name}_correlation.png")
            else:
                logger.info("No se encontraron columnas numéricas para analizar.")
            
            # Análisis de columnas categóricas
            logger.info("\n[ANÁLISIS DE COLUMNAS CATEGÓRICAS]")
            cat_analysis = self.analyze_categorical_columns(df)
            
            if cat_analysis:
                logger.info("\nResumen de columnas categóricas:")
                for col, summary in cat_analysis.items():
                    logger.info(f"\n{col}:")
                    logger.info("\n" + str(summary.head(10)))  # Mostrar solo las primeras 10 categorías
            else:
                logger.info("No se encontraron columnas categóricas para analizar.")
            
            # Generar reporte de perfil
            logger.info("\n[GENERANDO REPORTE DE PERFIL]")
            self.generate_profile_report(f"{output_prefix}_{name}_report.html")
            
            logger.info(f"\nAnálisis de '{name}' completado. Los resultados se han guardado en {self.reports_dir}")


if __name__ == "__main__":
    # Ejemplo de uso
    eda = ExploratoryDataAnalysis()
    
    # Cargar datos
    eda.load_data()
    
    # Ejecutar análisis completo
    eda.run_analysis()

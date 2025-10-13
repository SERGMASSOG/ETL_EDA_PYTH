# Análisis Exploratorio de Datos

Este proyecto está diseñado para el análisis exploratorio de datos con un enfoque en la minería de datos. Incluye un flujo completo ETL (Extract, Transform, Load) y herramientas para el análisis detallado de los datos.

## Estructura del Proyecto

```
Analisis_exploratorio_1/
├── Data/                   # Datos crudos
│   ├── raw/               # Datos sin procesar
│   ├── processed/         # Datos procesados
│   └── external/          # Fuentes de datos externas
├── notebooks/             # Jupyter notebooks para análisis exploratorio
├── src/                   # Código fuente
│   ├── etl/               # Scripts de ETL
│   ├── features/          # Ingeniería de características
│   ├── models/            # Modelos de machine learning
│   └── utils/             # Utilidades y funciones auxiliares
├── reports/               # Reportes y visualizaciones
│   ├── figures/           # Gráficos y figuras
│   └── results/           # Resultados del análisis
└── requirements.txt       # Dependencias del proyecto
```

## Configuración

1. Crear un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: .\venv\Scripts\activate
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. Colocar los datos en la carpeta `Data/raw/`
2. Ejecutar los scripts de ETL en `src/etl/`
3. Explorar los datos con los notebooks en `notebooks/`
4. Los resultados se guardarán en `reports/`

## Requisitos

- Python 3.8+
- Bibliotecas listadas en `requirements.txt`

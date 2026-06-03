# 🤖 Ecosistema Modular de Inteligencia Artificial & Data Science

Este repositorio contiene una herramienta modular empaquetada en Python diseñada para automatizar flujos de procesamiento de datos, Machine Learning y Deep Learning, acompañada de guías metodológicas y casos prácticos reutilizables.

---

## 📂 Arquitectura del Repositorio

```text
nombre-del-proyecto/
├── core/                       # LA HERRAMIENTA REUTILIZABLE (Código fuente)
│   ├── preprocessing/          # Módulos de Calidad y Transformación
│   │   ├── cleaning.py         # Tratamiento quirúrgico de nulos (MICE) y outliers (IQR)
│   │   └── transformations.py  # Manipulación de fechas, normalizaciones y uniones
│   └── models/                 # Modelos Predictivos (Próximamente...)
├── examples/                   # Entorno de aprendizaje y cuadernos interactivos
├── datasets/                   # Generadores y bases de datos simuladas de prueba
└── requirements.txt            # Dependencias del sistema (Pandas, Numpy, Scikit-Learn)
```

## 🛠️ Guía de Inicio Rápido
### 1. Clonar e instalar dependencias
Active su entorno virtual e instale los requisitos base:

```bash
git clone https://github.com/JoseDanielRomeroSu/AIDataTech.git
cd AIDataTech
python3 -m venv .venv
source .venv/bin/activate  # (o .venv\Scripts\activate en Windows)
pip install -r requirements.txt
```   

### 2. Generar la Base de Datos de Prueba
Para validar que los algoritmos modulares funcionan correctamente, corra el generador automático:

```bash
python datasets/generate_sample_data.py
```

### 3. Ejemplo de Uso Clínico de la Librería
Cree un script o abra un notebook en la raíz y ejecute este pipeline unificado:

```bash
import pandas as pd
from core import DataCleaner, DataTransformer

# Carga de datos de la carpeta de pruebas
df_clientes = pd.read_csv("datasets/sample_data.csv")
df_tx = pd.read_csv("datasets/sample_transacciones.csv")

# Instanciar módulos core
cleaner = DataCleaner()
transformer = DataTransformer()

# Pipeline de ejecución automatizado
df_limpio = cleaner.tratar_valores_perdidos(df_clientes, columnas=["edad"], estrategia="media")
df_limpio = cleaner.tratar_valores_perdidos(df_limpio, columnas=["ingresos"], estrategia="mice")

df_transformado = transformer.normalizar_fechas(df_limpio, columna="fecha_registro")
df_transformado = transformer.estandarizar_texto(df_transformado, columna="ciudad_residencia")

# Enriquecer dataset uniendo las transacciones
df_final = transformer.fusionar_tablas(df_transformado, df_tx, clave="id_cliente", como="left")

print("Dataset procesado con éxito:")
print(df_final.head())
```
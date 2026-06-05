# 🤖 Ecosistema Modular de Inteligencia Artificial & Data Science

Este repositorio contiene una herramienta modular empaquetada en Python diseñada para automatizar flujos de procesamiento de datos, Machine Learning y Deep Learning, acompañada de guías metodológicas y casos prácticos reutilizables.

---

## 📂 Arquitectura del Repositorio

```text
AIDataTech/
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

## ⚡ Guía de Uso del Módulo de Preprocesamiento (core/preprocessing)

<details>
<summary> Cree un script o abra un notebook en la raíz y ejecute este pipeline unificado para limpiar nulos de forma avanzada, normalizar textos, verticalizar datos o fusionar múltiples fuentes:</summary>

```python
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
</details>

## 🧠 Guía de Uso del Módulo de Modelado Predictivo (core/models)
<details>
<summary> La suite de modelos supervisados le permite ejecutar análisis estadísticos rigurosos (como la detección de multicolinealidad mediante el Factor de Inflación de la Varianza - VIF), lanzar regresiones polinómicas de cualquier grado y entrenar los algoritmos basados en árboles más potentes del estado del arte unificando la firma de código.

Utilice el siguiente script integrado para preparar sus datos (incluyendo el casteo automático de variables booleanas), evaluar multicolinealidad, lanzar un pool completo de algoritmos supervisados y extraer un ranking automático de rendimiento basado en R2 y RMSE:</summary>

```python
import pandas as pd
# Importar las suites predictivas unificadas del core
from AIDataTech.core import LinearRegressionSuite, TreeRegressionSuite

# 1. Cargar tu dataset (Ejemplo: Predecir precio de coches)
df_coches = pd.read_csv("datasets/Datos_coches_procesados.csv").dropna()

# 🌟 Tratamiento de tipos: Convertir columnas booleanas (True/False) a bits (1/0)
columnas_bool = df_coches.select_dtypes(include=['bool']).columns
df_coches[columnas_bool] = df_coches[columnas_bool].astype(int)

# 2. Definir variables predictoras (X) y variable objetivo (y)
features = ["Cambio", "Potencia", "Año", "kilometraje", "Diésel", "Eléctrico", "Gasolina"]
target = "Precio"

# Contenedores para almacenar métricas y rankings
records = []
importancias_modelos = {}

# =====================================================================
# PARTE A: Suites de Modelos Lineales y Estadísticos
# =====================================================================
suite_lr = LinearRegressionSuite()

print("--- 📊 ANÁLISIS DE MULTICOLINEALIDAD (VIF) ---")
# El Factor de Inflación de Varianza (VIF) ayuda a diagnosticar redundancias entre variables
print(suite_lr.calcular_vif(df_coches, features))
print("\n" + "="*50 + "\n")

# A.1 Regresión Lineal Multivariable
coef_lr, m_lr = suite_lr.entrenar_regresion_lineal(df_coches, features, target)
records.append({
    "Modelo": "Regresión Lineal", 
    "R2_Train": m_lr["train"]["R2"], 
    "R2_Test": m_lr["test"]["R2"], 
    "RMSE_Test": m_lr["test"]["RMSE"]
})

# Mapear coeficientes del plano lineal a DataFrame para consistencia en el ranking
importancias_modelos["Regresión Lineal"] = pd.DataFrame(
    list(coef_lr["Coeficientes"].items()), 
    columns=["Feature", "Importance"]
).sort_values(by="Importance", key=abs, ascending=False) # Ordenado por impacto absoluto

# A.2 Regresión Polinómica (Grado 2) para capturar relaciones curvas
m_poly = suite_lr.entrenar_regresion_polinomial(df_coches, features, target, grado=2)
records.append({
    "Modelo": "Regresión Polinómica (G2)", 
    "R2_Train": m_poly["train"]["R2"], 
    "R2_Test": m_poly["test"]["R2"], 
    "RMSE_Test": m_poly["test"]["RMSE"]
})

# =====================================================================
# PARTE B: Suites de Modelos Basados en Árboles y Ensembles
# =====================================================================

# B.1 Decision Tree Regressor
suite_dt = TreeRegressionSuite(tipo_modelo="decision_tree", hyperparametros={"max_depth": 5, "random_state": 42})
m_dt, imp_dt = suite_dt.entrenar(df_coches, features, target)
records.append({
    "Modelo": "Árbol de Decisión", 
    "R2_Train": m_dt["train"]["R2"], 
    "R2_Test": m_dt["test"]["R2"], 
    "RMSE_Test": m_dt["test"]["RMSE"]
})
importancias_modelos["Árbol de Decisión"] = imp_dt

# B.2 Random Forest (Ensemble por Bagging)
suite_rf = TreeRegressionSuite(tipo_modelo="random_forest", hyperparametros={"n_estimators": 100, "max_depth": 8, "random_state": 42})
m_rf, imp_rf = suite_rf.entrenar(df_coches, features, target)
records.append({
    "Modelo": "Random Forest", 
    "R2_Train": m_rf["train"]["R2"], 
    "R2_Test": m_rf["test"]["R2"], 
    "RMSE_Test": m_rf["test"]["RMSE"]
})
importancias_modelos["Random Forest"] = imp_rf

# B.3 XGBoost (Ensemble por Gradient Boosting optimizado)
suite_xgb = TreeRegressionSuite(tipo_modelo="xgboost", hyperparametros={"n_estimators": 100, "learning_rate": 0.05, "max_depth": 5, "random_state": 42})
m_xgb, imp_xgb = suite_xgb.entrenar(df_coches, features, target)
records.append({
    "Modelo": "XGBoost", 
    "R2_Train": m_xgb["train"]["R2"], 
    "R2_Test": m_xgb["test"]["R2"], 
    "RMSE_Test": m_xgb["test"]["RMSE"]
})
importancias_modelos["XGBoost"] = imp_xgb

# =====================================================================
# PARTE C: Análisis Comparativo y Diagnosis de Variables
# =====================================================================
# Consolidar métricas ordenando por capacidad de generalización en Test (R²)
df_benchmark = pd.DataFrame(records).sort_values(by="R2_Test", ascending=False).reset_index(drop=True)

print("🏆 RANKING FINAL DE RENDIMIENTO 🏆")
print(df_benchmark)

print("\n" + "="*50 + "\n")
print("🌲 SELECCIÓN DE VARIABLES / FEATURE IMPORTANCE 🌲\n")

# Evaluar qué variables han capturado mayor varianza según el modelo
for modelo, df_imp in importancias_modelos.items():
    print(f"📌 Variables más influyentes para: {modelo}")
    print(df_imp.head(3)) # Muestra el TOP 3
    print("-" * 40)
```
</details>

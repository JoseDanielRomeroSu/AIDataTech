# 🔍 Práctica de Análisis Preliminar y Exploratorio (EDA)

Antes de transformar tus datos con los módulos automáticos de la librería `core`, necesitas inspeccionar la naturaleza de tus archivos. 

Usa este flujo estándar recomendado en tus Jupyter Notebooks para diagnosticar el estado de tus datos:

### 1. Inspección de Dimensiones y Tipos de Datos
```python
import pandas as pd

# Cargar tus datos crudos
df = pd.read_csv("../../datasets/sample_data.csv")

# ¿Qué estructura tienen mis datos?
print(df.shape)
print(df.info())  # Muestra tipos de columnas y conteo de valores no nulos
```
### 2. Auditoría de Valores Faltantes (Nulos)
Identifica qué columnas requieren que invoques posteriormente a nuestro DataCleaner:

```python
# Porcentaje de nulos por columna
total_nulos = df.isnull().sum()
porcentaje_nulos = (df.isnull().sum() / len(df)) * 100
info_nulos = pd.DataFrame({'Nulos Totales': total_nulos, 'Porcentaje (%)': porcentaje_nulos})
print(info_nulos.sort_values(by='Porcentaje (%)', ascending=False))
```

### 3. Estadísticos Descriptivos Rápidos
Detecta anomalías o posibles outliers antes de tratarlos:

```python
# Variables numéricas (¿Hay valores mínimos o máximos sospechosos?)
print(df.describe())

# Variables categóricas (¿Cuántas categorías únicas existen?)
print(df.describe(include=['O']))
```
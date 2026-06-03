import pandas as pd
import numpy as np
from typing import Union, List, Optional
# Importamos el imputer avanzado de scikit-learn
from sklearn.experimental import enable_iterative_imputer  # Obligatorio para habilitarlo
from sklearn.impute import IterativeImputer


class DataCleaner:
    """
    Clase modular encargada de las tareas quirúrgicas de calidad y limpieza de datos:
    Tratamiento de nulos (básico y avanzado), detección/filtrado de outliers y formateo.
    """

    def __init__(self):
        pass

    def tratar_valores_perdidos(
        self, 
        df: pd.DataFrame, 
        columnas: Union[str, List[str]], 
        estrategia: str = "media", 
        valor_fijo: Optional[any] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Trata los valores faltantes (NaN) en las columnas especificadas.
        
        Estrategias disponibles:
        - 'eliminar': Borra las filas que contengan NaN en esas columnas.
        - 'media': Imputa usando la media aritmética (solo numéricas).
        - 'mediana': Imputa usando la mediana (solo numéricas).
        - 'moda': Imputa usando el valor más frecuente.
        - 'fijo': Imputa usando el parámetro 'valor_fijo'.
        - 'mice': Imputación avanzada multivariante (IterativeImputer) basada en Machine Learning.
        """
        df_clean = df.copy()
        
        # Asegurar que columnas sea una lista
        if isinstance(columnas, str):
            columnas = [columnas]
            
        # Validar que todas las columnas existan
        for col in columnas:
            if col not in df_clean.columns:
                raise ValueError(f"La columna '{col}' no existe en el DataFrame.")

        # --- ESTRATEGIA AVANZADA: MICE ---
        if estrategia == "mice":
            # IterativeImputer requiere trabajar sobre variables numéricas
            # Seleccionamos el subset y aplicamos el modelo estimador
            imputer = IterativeImputer(random_state=random_state, max_iter=10)
            
            # Ajustamos y transformamos solo las columnas seleccionadas
            df_clean[columnas] = imputer.fit_transform(df_clean[columnas])
            return df_clean

        # --- ESTRATEGIAS TRADICIONALES ---
        for col in columnas:
            if estrategia == "eliminar":
                df_clean = df_clean.dropna(subset=[col])
            elif estrategia == "media":
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            elif estrategia == "mediana":
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            elif estrategia == "moda":
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            elif estrategia == "fijo":
                if valor_fijo is None:
                    raise ValueError("Si la estrategia es 'fijo', debes proveer un 'valor_fijo'.")
                df_clean[col] = df_clean[col].fillna(valor_fijo)
            else:
                raise ValueError(f"Estrategia '{estrategia}' no reconocida.")
                
        return df_clean

    def detectar_outliers_iqr(
        self, 
        df: pd.DataFrame, 
        columna: str, 
        factor: float = 1.5
    ) -> pd.Series:
        """
        Aplica la regla del Rango Intercuartílico (IQR) para identificar outliers.
        Devuelve una serie booleana (True para los outliers).
        """
        if columna not in df.columns:
            raise ValueError(f"La columna '{columna}' no existe.")
            
        q1 = df[columna].quantile(0.25)
        q3 = df[columna].quantile(0.75)
        iqr = q3 - q1
        
        limite_inferior = q1 - (factor * iqr)
        limite_superior = q3 + (factor * iqr)
        
        return (df[columna] < limite_inferior) | (df[columna] > limite_superior)

    def gestionar_outliers(
        self, 
        df: pd.DataFrame, 
        columna: str, 
        metodo: str = "eliminar", 
        factor: float = 1.5
    ) -> pd.DataFrame:
        """
        Detecta y gestiona los outliers de una columna numérica.
        
        Métodos:
        - 'eliminar': Remueve las filas con outliers.
        - 'cap': Reemplaza los outliers con los valores de los límites (Capping / Winsorización).
        """
        df_clean = df.copy()
        outliers = self.detectar_outliers_iqr(df_clean, columna, factor)
        
        if metodo == "eliminar":
            df_clean = df_clean[~outliers]
        elif metodo == "cap":
            q1 = df_clean[columna].quantile(0.25)
            q3 = df_clean[columna].quantile(0.75)
            iqr = q3 - q1
            limite_inferior = q1 - (factor * iqr)
            limite_superior = q3 + (factor * iqr)
            
            df_clean[columna] = np.where(df_clean[columna] < limite_inferior, limite_inferior, df_clean[columna])
            df_clean[columna] = np.where(df_clean[columna] > limite_superior, limite_superior, df_clean[columna])
        else:
            raise ValueError(f"Método '{metodo}' no reconocido.")
            
        return df_clean
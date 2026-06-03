import pandas as pd
import numpy as np
from typing import Union, List, Optional


class DataCleaner:
    """
    Clase modular encargada de las tareas quirúrgicas de calidad y limpieza de datos:
    Tratamiento de nulos, detección/filtrado de outliers y formateo básico.
    """

    def __init__(self):
        pass

    def tratar_valores_perdidos(
        self, 
        df: pd.DataFrame, 
        columnas: Union[str, List[str]], 
        estrategia: str = "media", 
        valor_fijo: Optional[any] = None
    ) -> pd.DataFrame:
        """
        Trata los valores faltantes (NaN) en las columnas especificadas.
        
        Estrategias disponibles:
        - 'eliminar': Borra las filas que contengan NaN en esas columnas.
        - 'media': Imputa usando la media aritmética (solo numéricas).
        - 'mediana': Imputa usando la mediana (solo numéricas).
        - 'moda': Imputa usando el valor más frecuente.
        - 'fijo': Imputa usando el parámetro 'valor_fijo'.
        """
        df_clean = df.copy()
        
        # Asegurar que columnas sea una lista
        if isinstance(columnas, str):
            columnas = [columnas]
            
        for col in columnas:
            if col not in df_clean.columns:
                raise ValueError(f"La columna '{col}' no existe en el DataFrame.")
                
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
        
        # Filtro booleano: True si está fuera de los límites
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
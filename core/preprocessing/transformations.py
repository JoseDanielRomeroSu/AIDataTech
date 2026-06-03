import pandas as pd
from typing import List, Union

class DataTransformer:
    """
    Clase modular para la transformación, enriquecimiento y formateo avanzado de estructuras de datos.
    Cubre manipulación de fechas, uniones complejas, normalizaciones y agregados (Módulos 2 y 5).
    """
    
    def __init__(self):
        pass

    def normalizar_fechas(self, df: pd.DataFrame, columna: str, formato_destino: str = "%Y-%m-%d") -> pd.DataFrame:
        """
        Convierte una columna de fechas heterogéneas en un formato estandarizado Datetime,
        coaccionando los errores tipográficos fatales a NaT (Not a Time).
        """
        df_trans = df.copy()
        if columna not in df_trans.columns:
            raise ValueError(f"La columna '{columna}' no existe.")
            
        # errors='coerce' transforma registros caóticos como '2025-13-45' en NaT de forma segura
        df_trans[columna] = pd.to_datetime(df_trans[columna], errors='coerce')
        return df_trans

    def expandir_columna_delimitada(self, df: pd.DataFrame, columna: str, delimitador: str = "|") -> pd.DataFrame:
        """
        Toma una columna cuyos elementos contienen strings combinados (ej: 'Moda|Hogar') 
        y los expande en un formato estructurado o dummy. (Modificaciones avanzadas).
        """
        df_trans = df.copy()
        if columna not in df_trans.columns:
            raise ValueError(f"La columna '{columna}' no existe.")
            
        # Crear variables dummy a partir de las listas separadas por el delimitador
        dummies = df_trans[columna].str.get_dummies(sep=delimitador)
        return pd.concat([df_trans, dummies], axis=1)

    def estandarizar_texto(self, df: pd.DataFrame, columna: str) -> pd.DataFrame:
        """
        Elimina variaciones tipográficas básicas (ej: 'madrid ', 'MADRID' -> 'madrid') 
        limpiando espacios vacíos en los extremos y homogenizando a minúsculas.
        """
        df_trans = df.copy()
        if columna not in df_trans.columns:
            raise ValueError(f"La columna '{columna}' no existe.")
            
        df_trans[columna] = df_trans[columna].astype(str).str.strip().str.lower()
        return df_trans

    def fusionar_tablas(self, izquierda: pd.DataFrame, derecha: pd.DataFrame, clave: str, como: str = "inner") -> pd.DataFrame:
        """
        Une dos tablas (Merge/Join) basándose en una clave relacional.
        Soporta: 'inner', 'left', 'right', 'outer'.
        """
        return pd.merge(izquierda, derecha, on=clave, how=como)
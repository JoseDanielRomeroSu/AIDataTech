import pandas as pd
from typing import List, Union, Dict, Any

class DataTransformer:
    """
    Clase modular para la transformación, enriquecimiento, filtrado y 
    reestructuración avanzada de datos (Módulos 2, 5 y 6).
    """
    
    def __init__(self):
        pass

    def normalizar_fechas(self, df: pd.DataFrame, columna: str, formato_destino: str = "%Y-%m-%d") -> pd.DataFrame:
        """
        Convierte una columna de fechas heterogéneas en formato Datetime estandarizado.
        """
        df_trans = df.copy()
        if columna not in df_trans.columns:
            raise ValueError(f"La columna '{columna}' no existe.")
        df_trans[columna] = pd.to_datetime(df_trans[columna], errors='coerce')
        return df_trans

    def expandir_columna_delimitada(self, df: pd.DataFrame, columna: str, delimitador: str = "|") -> pd.DataFrame:
        """
        Expande una columna con strings combinados (ej: 'Moda|Hogar') en variables dummy.
        """
        df_trans = df.copy()
        if columna not in df_trans.columns:
            raise ValueError(f"La columna '{columna}' no existe.")
        dummies = df_trans[columna].str.get_dummies(sep=delimitador)
        return pd.concat([df_trans, dummies], axis=1)

    def estandarizar_texto(self, df: pd.DataFrame, columna: str) -> pd.DataFrame:
        """
        Homogeniza strings eliminando espacios vacíos y convirtiendo a minúsculas.
        """
        df_trans = df.copy()
        if columna not in df_trans.columns:
            raise ValueError(f"La columna '{columna}' no existe.")
        df_trans[columna] = df_trans[columna].astype(str).str.strip().str.lower()
        return df_trans

    def fusionar_tablas(self, izquierda: pd.DataFrame, derecha: pd.DataFrame, clave: str, como: str = "inner") -> pd.DataFrame:
        """
        Une dos tablas (Merge) basándose en una clave relacional.
        """
        return pd.merge(izquierda, derecha, on=clave, how=como)

    # --- NUEVAS FUNCIONES DE FILTRADO (Basadas en tu material) ---
    
    def gestionar_duplicados(self, df: pd.DataFrame, subset: Union[str, List[str]] = None, mantener: str = "first") -> pd.DataFrame:
        """
        Detecta y elimina registros duplicados en el DataFrame basándose en ciertas columnas o en todas.
        """
        df_trans = df.copy()
        return df_trans.drop_duplicates(subset=subset, keep=mantener)

    def filtrar_por_condiciones(self, df: pd.DataFrame, condiciones: Dict[str, Any], operador: str = "AND") -> pd.DataFrame:
        """
        Simplifica el filtrado avanzado de filas permitiendo pasar un diccionario de {columna: valor}.
        Ejemplo: condiciones={'ciudad': 'madrid', 'edad': 25}
        """
        df_trans = df.copy()
        query_parts = []
        
        for col, val in condiciones.items():
            if col not in df_trans.columns:
                raise ValueError(f"La columna '{col}' no existe para aplicar el filtro.")
            
            # Formatear la query interna dinámicamente si es string o numérico
            if isinstance(val, str):
                query_parts.append(f"`{col}` == '{val}'")
            else:
                query_parts.append(f"`{col}` == {val}")
                
        unión = " & " if operador.upper() == "AND" else " | "
        query_final = unión.join(query_parts)
        
        if query_final:
            return df_trans.query(query_final)
        return df_trans

    # --- NUEVAS FUNCIONES DE VERTICALIZACIÓN Y AGREGADOS ---

    def verticalizar_tabla(self, df: pd.DataFrame, id_vars: List[str], value_vars: List[str], var_name: str = "Variable", value_name: str = "Valor") -> pd.DataFrame:
        """
        Transforma una tabla de formato ancho a formato largo (Verticalización / Melt).
        Útil para reorganizar métricas dispersas en columnas hacia una estructura de base de datos normalizada.
        """
        df_trans = df.copy()
        return pd.melt(df_trans, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)

    def resumir_por_grupo(self, df: pd.DataFrame, agrupadores: List[str], metricas: Dict[str, Union[str, List[str]]]) -> pd.DataFrame:
        """
        Agrupa los datos según ciertas columnas clave y calcula resúmenes estadísticos (medias, sumas, conteos).
        Ejemplo: metricas={'monto': ['sum', 'mean']}
        """
        df_trans = df.copy()
        return df_trans.groupby(agrupadores).agg(metricas).reset_index()
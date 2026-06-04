import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb  #


class LinearRegressionSuite:
    """Suite automatizada para modelos de Regresión Lineal y Polinómica."""
    def __init__(self):
        self.model = LinearRegression()
        self.poly_features = None

    def calcular_vif(self, df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
        X = df[variables].dropna()
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data.sort_values(by="VIF", ascending=False)

    def _evaluar_modelo(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_true, y_pred)
        }

    def entrenar_regresion_lineal(self, df: pd.DataFrame, features: List[str], target: str, test_size: float = 0.2, random_state: int = 42):
        X, y = df[features], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        return {"Intercept": self.model.intercept_, "Coeficientes": dict(zip(features, self.model.coef_))}, {"train": self._evaluar_modelo(y_train, self.model.predict(X_train)), "test": self._evaluar_modelo(y_test, self.model.predict(X_test))}

    def entrenar_regresion_polinomial(self, df: pd.DataFrame, features: List[str], target: str, grado: int = 2, test_size: float = 0.2, random_state: int = 42):
        X, y = df[features], df[target]
        self.poly_features = PolynomialFeatures(degree=grado, include_bias=False)
        X_poly = self.poly_features.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        return {"train": self._evaluar_modelo(y_train, self.model.predict(X_train)), "test": self._evaluar_modelo(y_test, self.model.predict(X_test))}


# =====================================================================
# 🌲 NUEVA SUITE DE MODELOS BASADOS EN ÁRBOLES Y ENSEMBLES
# =====================================================================

class TreeRegressionSuite:
    """
    Suite unificada para entrenar, evaluar y extraer importancias en modelos
    basados en Árboles de Decisión, Random Forest y XGBoost.
    """

    def __init__(self, tipo_modelo: str = "random_forest", hyperparametros: Dict[str, Any] = None):
        """
        Modelos disponibles: 'decision_tree', 'random_forest', 'xgboost'.
        """
        params = hyperparametros if hyperparametros else {}
        self.tipo_modelo = tipo_modelo.lower()
        
        if self.tipo_modelo == "decision_tree":
            self.model = DecisionTreeRegressor(**params)
        elif self.tipo_modelo == "random_forest":
            self.model = RandomForestRegressor(**params)
        elif self.tipo_modelo == "xgboost":
            self.model = xgb.XGBRegressor(**params)
        else:
            raise ValueError(f"Modelo '{tipo_modelo}' no soportado. Elige 'decision_tree', 'random_forest' o 'xgboost'.")

    def _evaluar_modelo(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_true, y_pred)
        }

    def entrenar(
        self, 
        df: pd.DataFrame, 
        features: List[str], 
        target: str, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
        """
        Ejecuta el split train-test, entrena el algoritmo seleccionado, genera un 
        reporte de rendimiento completo y extrae la importancia de cada feature.
        """
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Ajustar el modelo predictivo
        self.model.fit(X_train, y_train)

        # Predecir conjuntos
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Computar métricas estructurales
        metricas = {
            "train": self._evaluar_modelo(y_train, y_train_pred),
            "test": self._evaluar_modelo(y_test, y_test_pred)
        }

        # Extraer importancia de las variables (Feature Importance)
        importancias = pd.DataFrame({
            "Feature": features,
            "Importance": self.model.feature_importances_
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        return metricas, importancias
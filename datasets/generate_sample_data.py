import pandas as pd
import numpy as np

def generar_datasets_prueba():
    print("Generando datos de prueba para el proyecto...")
    
    # 1. Dataset Principal: Clientes y Consumos (Para Limpieza, Fechas y Modificaciones)
    np.random.seed(42)
    n_rows = 100
    
    fechas = pd.date_range(start="2025-01-01", periods=n_rows, freq="D").strftime("%d/%m/%Y").tolist()
    # Introducir algunas fechas con formatos caóticos o strings erróneos
    fechas[5] = "2025-13-45" 
    fechas[12] = "05-12-2025"
    
    data_clientes = {
        "id_cliente": [f"CLI_{i:03d}" for i in range(1, n_rows + 1)],
        "fecha_registro": fechas,
        "edad": np.random.choice([np.nan, 25, 34, 45, 52, 61, 19, 88], size=n_rows),
        "ingresos": np.random.choice([np.nan, 1200, 2300, 3100, 4500, 15000], size=n_rows), # 15000 es outlier
        "ciudad_residencia": np.random.choice(["Madrid", "madrid", "MADRID", "Barcelona", "barcelona", "Bilbao", None], size=n_rows),
        "categoria_producto": np.random.choice(["Electrónica|Hogar", "Moda", "Hogar|Cocina", "Mascotas"], size=n_rows)
    }
    
    df_clientes = pd.DataFrame(data_clientes)
    df_clientes.to_csv("datasets/sample_data.csv", index=False)
    print("- Creado: datasets/sample_data.csv (Dataset principal)")

    # 2. Dataset Secundario: Transacciones (Para el módulo de Uniones/Merge)
    data_transacciones = {
        "id_cliente": [f"CLI_{np.random.randint(1, 110):03d}" for _ in range(150)], # Algunos IDs no existirán en clientes
        "id_transaccion": [f_tx for f_tx in range(1000, 1150)],
        "monto": np.round(np.random.uniform(10.5, 500.0, size=150), 2)
    }
    df_tx = pd.DataFrame(data_transacciones)
    df_tx.to_csv("datasets/sample_transacciones.csv", index=False)
    print("- Creado: datasets/sample_transacciones.csv (Para cruces de tablas)")

if __name__ == "__main__":
    generar_datasets_prueba()
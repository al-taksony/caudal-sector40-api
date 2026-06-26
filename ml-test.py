import joblib
import numpy as np
import pandas as pd

# Configuration from notebook
USAR_VARIABLES_TEMPORALES = True
TOLERANCIA_MCA = 2.0

# Define feature columns (from notebook cell 7)
feature_cols = ['presion_salida_mca', 'temperatura_c', 'precipitacion_mm_h', 'humedad_pct', 'viento_m_s']
if USAR_VARIABLES_TEMPORALES:
    feature_cols += ['hora_sin', 'hora_cos', 'dow_sin', 'dow_cos', 'mes']

# Load the best model
model = joblib.load('./model/model.pkl')

# Load the scalers
scaler_X = joblib.load('./model/scaler_X.pkl')
scaler_y = joblib.load('./model/scaler_y.pkl')

print("Model and scalers loaded successfully!")

def preparar_escenarios(nuevos):
    """
    Prepara escenarios para predicción.
    Calcula variables temporales a partir de fecha_hora.
    """
    nuevos = nuevos.copy()
    
    # Convertir fecha_hora a datetime
    if isinstance(nuevos['fecha_hora'].iloc[0], str):
        nuevos['fecha_hora'] = pd.to_datetime(nuevos['fecha_hora'])
    
    # Extraer variables temporales
    nuevos['hora'] = nuevos['fecha_hora'].dt.hour
    nuevos['dia_semana'] = nuevos['fecha_hora'].dt.dayofweek
    nuevos['mes'] = nuevos['fecha_hora'].dt.month
    
    # Crear variables temporales cíclicas
    nuevos['hora_sin'] = np.sin(2 * np.pi * nuevos['hora'] / 24)
    nuevos['hora_cos'] = np.cos(2 * np.pi * nuevos['hora'] / 24)
    nuevos['dow_sin'] = np.sin(2 * np.pi * nuevos['dia_semana'] / 7)
    nuevos['dow_cos'] = np.cos(2 * np.pi * nuevos['dia_semana'] / 7)
    
    return nuevos

def predecir_caudal_api(nuevos):
    """Predice caudal basado en escenarios."""
    nuevos_preparados = preparar_escenarios(nuevos)
    # Using the globally defined feature_cols directly
    X = nuevos_preparados[feature_cols]
    X_scaled = scaler_X.transform(X)
    y_scaled = model.predict(X_scaled)
    y_lps = scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
    nuevos_preparados['caudal_pred_lps_api'] = y_lps
    return nuevos_preparados

# Test data
nuevos_escenarios = pd.DataFrame([
    {
        'fecha_hora': '2026-06-18 10:00:00',
        'presion_salida_mca': 16.0,
        'temperatura_c': 24.0,
        'precipitacion_mm_h': 0.0,
        'humedad_pct': 70.0,
        'viento_m_s': 1.0,
    },
    {
        'fecha_hora': '2026-06-18 02:00:00',
        'presion_salida_mca': 8.0,
        'temperatura_c': 18.0,
        'precipitacion_mm_h': 0.0,
        'humedad_pct': 90.0,
        'viento_m_s': 0.2,
    },
])

predicciones_api = predecir_caudal_api(nuevos_escenarios)
resultado = predicciones_api[['fecha_hora', 'presion_salida_mca', 'caudal_pred_lps_api']].copy()
resultado['caudal_pred_lps_api'] = resultado['caudal_pred_lps_api'].round(3)
resultado['presion_salida_mca'] = resultado['presion_salida_mca'].round(3)
print(resultado)
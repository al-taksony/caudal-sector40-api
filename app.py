from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Caudal Sector 40 API",
    description="API para predicción de caudal en Sector 40",
    version="1.0.0"
)

# Configuration
USAR_VARIABLES_TEMPORALES = True
TOLERANCIA_MCA = 2.0

# Define feature columns
feature_cols = ['presion_salida_mca', 'temperatura_c', 'precipitacion_mm_h', 'humedad_pct', 'viento_m_s']
if USAR_VARIABLES_TEMPORALES:
    feature_cols += ['hora_sin', 'hora_cos', 'dow_sin', 'dow_cos', 'mes']

# Load model and scalers
model = joblib.load('./model/model.pkl')
scaler_X = joblib.load('./model/scaler_X.pkl')
scaler_y = joblib.load('./model/scaler_y.pkl')

print("✓ Modelo cargado exitosamente")


# Request model
class PredictionRequest(BaseModel):
    fecha_hora: str
    presion_salida_mca: float
    temperatura_c: float
    precipitacion_mm_h: float = 0.0
    humedad_pct: float
    viento_m_s: float = 0.0


# Response model
class PredictionResponse(BaseModel):
    fecha_hora: str
    presion_salida_mca: float
    temperatura_c: float
    caudal_pred_lps: float


def preparar_escenarios(nuevos):
    """Prepara escenarios para predicción."""
    nuevos = nuevos.copy()
    
    if isinstance(nuevos['fecha_hora'].iloc[0], str):
        nuevos['fecha_hora'] = pd.to_datetime(nuevos['fecha_hora'])
    
    nuevos['hora'] = nuevos['fecha_hora'].dt.hour
    nuevos['dia_semana'] = nuevos['fecha_hora'].dt.dayofweek
    nuevos['mes'] = nuevos['fecha_hora'].dt.month
    
    nuevos['hora_sin'] = np.sin(2 * np.pi * nuevos['hora'] / 24)
    nuevos['hora_cos'] = np.cos(2 * np.pi * nuevos['hora'] / 24)
    nuevos['dow_sin'] = np.sin(2 * np.pi * nuevos['dia_semana'] / 7)
    nuevos['dow_cos'] = np.cos(2 * np.pi * nuevos['dia_semana'] / 7)
    
    return nuevos


@app.get("/")
def read_root():
    """Endpoint raíz de la API"""
    return {
        "mensaje": "Bienvenido a la API de Caudal Sector 40",
        "version": "1.0.0",
        "endpoints": {
            "prediccion": "/predict",
            "docs": "/docs"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Endpoint para predicción de caudal.
    
    Parámetros requeridos:
    - fecha_hora: Fecha y hora (formato: "YYYY-MM-DD HH:MM:SS")
    - presion_salida_mca: Presión de salida en mca
    - temperatura_c: Temperatura en °C
    - humedad_pct: Humedad relativa en %
    
    Parámetros opcionales:
    - precipitacion_mm_h: Precipitación en mm/h (default: 0.0)
    - viento_m_s: Velocidad del viento en m/s (default: 0.0)
    """
    
    # Crear DataFrame con los datos de entrada
    nuevos_escenarios = pd.DataFrame([{
        'fecha_hora': request.fecha_hora,
        'presion_salida_mca': request.presion_salida_mca,
        'temperatura_c': request.temperatura_c,
        'precipitacion_mm_h': request.precipitacion_mm_h,
        'humedad_pct': request.humedad_pct,
        'viento_m_s': request.viento_m_s,
    }])
    
    # Preparar escenarios
    nuevos_preparados = preparar_escenarios(nuevos_escenarios)
    
    # Hacer predicción
    X = nuevos_preparados[feature_cols]
    X_scaled = scaler_X.transform(X)
    y_scaled = model.predict(X_scaled)
    y_lps = scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
    
    return PredictionResponse(
        fecha_hora=request.fecha_hora,
        presion_salida_mca=request.presion_salida_mca,
        temperatura_c=request.temperatura_c,
        caudal_pred_lps=float(y_lps[0])
    )


@app.post("/predict-batch")
def predict_batch(requests: list[PredictionRequest]):
    """
    Endpoint para predicciones en lote.
    """
    results = []
    for request in requests:
        result = predict(request)
        results.append(result)
    return {"predicciones": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

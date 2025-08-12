from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
import logging
import uvicorn

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

MODEL_PATH = "mobilenetv2_model.h5"
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    logger.exception(f"Error cargando modelo desde {MODEL_PATH}")
    raise RuntimeError(f"No se pudo cargar el modelo: {e}")

class_names = [
    "Potato_early_blight",
    "Potato_healthy",
    "Potato_late_blight",
    "Potato_leafroll_virus",
    "Potato_mosaic_virus"
]

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_k: List[dict]

def preprocess_image(file_bytes: bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Archivo vacío.")

    try:
        img_tensor = preprocess_image(contents)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="No se pudo procesar la imagen. Formato inválido.")
    except Exception as e:
        logger.exception("Error en preprocess_image")
        raise HTTPException(status_code=500, detail=f"Error al procesar imagen: {e}")

    try:
        preds = model.predict(img_tensor)
        probs = preds.flatten()
        top_indices = probs.argsort()[::-1][:3]
        top_k = [
            {"class": class_names[idx], "confidence": round(float(probs[idx]), 4)}
            for idx in top_indices
        ]
        predicted_class = class_names[int(np.argmax(probs))]
        confidence = float(np.max(probs))

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            top_k=top_k
        )
    except Exception as e:
        logger.exception("Error en predicción")
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {e}")




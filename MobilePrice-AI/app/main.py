"""FastAPI application for MobilePrice AI."""
from pathlib import Path
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from app.schemas import PredictionRequest
from app.ml.predict import ModelRegistry, MODEL_FILES

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
registry = ModelRegistry(ROOT / "models")
app = FastAPI(title="MobilePrice AI", description="Mobile price range classification using three ML algorithms.")
app.mount("/static", StaticFiles(directory=APP_DIR/"static"), name="static")
templates = Jinja2Templates(directory=APP_DIR/"templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "models_ready": registry.ready,
        "loaded_models": list(registry.models.keys()),
        "required_models": list(MODEL_FILES.keys()),
        "dataset_available": (ROOT/"data"/"train.csv").is_file(),
    }

@app.get("/api/metrics")
async def metrics():
    path = ROOT/"reports"/"metrics.json"
    if not path.is_file():
        return {"available": False, "message": "Metrics are not available. Add data/train.csv and train the models."}
    return {"available": True, **json.loads(path.read_text(encoding="utf-8"))}

@app.post("/api/predict")
async def predict(payload: PredictionRequest):
    if not registry.models:
        raise HTTPException(status_code=503, detail="No trained models found. Add data/train.csv and run python train_models.py.")
    values = payload.mobile.model_dump()
    names = list(registry.models) if payload.model == "Compare All" else [payload.model]
    missing = [name for name in names if name not in registry.models]
    if missing:
        raise HTTPException(status_code=503, detail=f"Models not available: {', '.join(missing)}. Train the models first.")
    try:
        results = [registry.predict_one(name, values) for name in names]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return {"results": results}

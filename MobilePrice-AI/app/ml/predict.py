"""Load persisted models and produce validated class probabilities."""
from pathlib import Path
import joblib
import pandas as pd
from .preprocessing import FEATURES, CLASS_NAMES

MODEL_FILES = {
    "Random Forest": "random_forest.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "Neural Network": "neural_network.pkl",
}

class ModelRegistry:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.load()

    def load(self):
        self.models = {}
        for name, filename in MODEL_FILES.items():
            path = self.models_dir / filename
            if path.is_file():
                self.models[name] = joblib.load(path)
        return self.models

    @property
    def ready(self):
        return len(self.models) == len(MODEL_FILES)

    def predict_one(self, model_name: str, values: dict):
        if model_name not in self.models:
            raise RuntimeError(f"{model_name} is not trained. Run python train_models.py first.")
        row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
        model = self.models[model_name]
        predicted = int(model.predict(row)[0])
        raw_probabilities = model.predict_proba(row)[0]
        probabilities = {str(i): float(raw_probabilities[list(model.classes_).index(i)])
                         if i in model.classes_ else 0.0 for i in range(4)}
        return {
            "model": model_name,
            "predicted_class": predicted,
            "predicted_category": CLASS_NAMES[predicted],
            "confidence": probabilities[str(predicted)],
            "probabilities": probabilities,
        }

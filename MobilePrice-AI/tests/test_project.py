from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from fastapi.testclient import TestClient
from app.ml.preprocessing import FEATURES, EXPECTED_COLUMNS, load_and_validate_dataset, DatasetValidationError
from app.ml.predict import ModelRegistry
from app.main import app, registry

def sample_frame(n=40):
    rng = np.random.default_rng(7)
    data = {f: rng.integers(1, 100, size=n) for f in FEATURES}
    for f in ["blue","dual_sim","four_g","three_g","touch_screen","wifi"]:
        data[f] = rng.integers(0,2,size=n)
    data["price_range"] = np.tile([0,1,2,3], n//4)
    return pd.DataFrame(data, columns=EXPECTED_COLUMNS)

def test_dataset_validation(tmp_path):
    path = tmp_path/"train.csv"
    sample_frame().to_csv(path,index=False)
    df = load_and_validate_dataset(path)
    assert len(df) == 40
    assert list(df.columns) == EXPECTED_COLUMNS

def test_dataset_rejects_missing_column(tmp_path):
    path = tmp_path/"bad.csv"
    sample_frame().drop(columns=["wifi"]).to_csv(path,index=False)
    try:
        load_and_validate_dataset(path)
        assert False, "Expected validation error"
    except DatasetValidationError as exc:
        assert "wifi" in str(exc)

def test_registry_load_and_predict(tmp_path):
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(
        pd.DataFrame(sample_frame()[FEATURES]), sample_frame()["price_range"]
    )
    joblib.dump(model,tmp_path/"random_forest.pkl")
    registry_obj = ModelRegistry(tmp_path)
    values = {f: float(sample_frame().iloc[0][f]) for f in FEATURES}
    result = registry_obj.predict_one("Random Forest", values)
    assert result["predicted_class"] in range(4)
    assert abs(sum(result["probabilities"].values())-1) < 1e-8

def test_api_health_and_metrics():
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/metrics").status_code == 200

def test_api_input_validation():
    client = TestClient(app)
    response = client.post("/api/predict", json={"model":"Random Forest","mobile":{"battery_power":-1}})
    assert response.status_code == 422

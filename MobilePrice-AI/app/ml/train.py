"""Train exactly three classifiers and persist fitted estimators."""
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .preprocessing import load_and_validate_dataset, prepare_split
from .evaluate import evaluate_model, save_evaluation

def build_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight=None
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000,random_state=42)),
            ]),
        "Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                max_iter=600, random_state=42, early_stopping=True,
                n_iter_no_change=20
            )),
        ]),
    }

def train_all(data_path: Path, models_dir: Path, reports_dir: Path):
    df = load_and_validate_dataset(data_path)
    X_train, X_test, y_train, y_test = prepare_split(df, random_state=42)
    models_dir.mkdir(parents=True, exist_ok=True)
    trained, results = {}, []
    filenames = {
        "Random Forest": "random_forest.pkl",
        "Logistic Regression": "logistic_regression.pkl",
        "Neural Network": "neural_network.pkl",
    }
    for name, estimator in build_models().items():
        print(f"Training {name}...")
        estimator.fit(X_train, y_train)
        trained[name] = estimator
        joblib.dump(estimator, models_dir / filenames[name])
        results.append(evaluate_model(name, estimator, X_test, y_test))
        print(f"{name}: accuracy={results[-1]['accuracy']:.4f}")
    save_evaluation(results, reports_dir)
    metadata = {
        "dataset_rows": int(len(df)),
        "feature_count": 20,
        "class_count": 4,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": 42,
    }
    import json
    (reports_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return trained, results

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    train_all(project_root/"data"/"train.csv", project_root/"models", project_root/"reports")

# MobilePrice AI
**Mobile Price Range Classification Using Machine Learning Algorithms**  
 · Python · Scikit-learn · FastAPI · HTML/Tailwind CSS/JavaScript

## 1. Introduction
MobilePrice AI is a browser-based machine learning application that predicts the dataset price-range class of a mobile phone from 20 technical specifications. It implements exactly three classifiers: Random Forest, Logistic Regression, and a Neural Network using Scikit-learn's MLPClassifier.

The four target labels are dataset categories only: **0 Low Cost, 1 Medium Cost, 2 High Cost, 3 Very High Cost**. They are not actual market prices.

## 2. Objectives
- Validate the supplied Kaggle CSV before model training.
- Train and evaluate three classifiers using one shared stratified 80:20 split.
- Compare accuracy, weighted precision, weighted recall, weighted F1-score and confusion matrices.
- Save fitted models and evaluation charts/reports.
- Provide a responsive web interface and JSON prediction API.

## 3. Algorithms
1. **Random Forest Classifier** — 300 decision trees; no feature scaling.
2. **Logistic Regression** — StandardScaler followed by multiclass LogisticRegression.
3. **Neural Network (MLPClassifier)** — StandardScaler followed by an MLP with hidden layers `(64, 32)`, ReLU activation and Adam solver.

All models use random state 42. Logistic Regression and MLP are saved as complete Scikit-learn pipelines, ensuring the same scaling is applied at prediction time. No model metrics or predictions are pre-filled or fabricated.

## 4. Technologies and libraries
- Python: application and model code.
- Pandas / NumPy: tabular data handling and numerical operations.
- Scikit-learn: splitting, preprocessing, classifiers and metrics.
- Joblib: model persistence.
- FastAPI / Uvicorn: backend API and local web server.
- Pydantic: request validation.
- Jinja2: serving the HTML page.
- Matplotlib: accuracy and confusion-matrix PNG reports.
- HTML / Tailwind CSS CDN / JavaScript: responsive frontend.
- Pytest / HTTPX: automated tests and API test client.

## 5. Dataset setup
Download `train.csv` from the Kaggle **Mobile Price Classification** dataset:
https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

Copy the real file to:
`MobilePrice-AI/data/train.csv`

The expected CSV has the 20 feature columns listed below plus `price_range`:
`battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi`.

The training script checks required and unexpected columns, empty data, missing values, duplicate rows, numeric data types, non-negative features, and the four target classes. It stops with a clear error if validation fails. It does not generate substitute data. `data/train.csv` is intentionally not included in this ZIP because the real Kaggle data could not be bundled reliably; place your downloaded file there manually.

## 6. Windows 11 installation
Install Python 3.10 or 3.11 (64-bit) and ensure **Add Python to PATH** is selected.

Open PowerShell in the extracted `MobilePrice-AI` folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation for this terminal, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 7. Train the models
After putting the real CSV in `data\train.csv`, run:

```powershell
python train_models.py
```

This trains all three algorithms using the same 80:20 stratified split (`random_state=42`) and writes:
- `models\random_forest.pkl`
- `models\logistic_regression.pkl`
- `models\neural_network.pkl`
- `reports\metrics.json`
- `reports\training_metadata.json`
- `reports\model_comparison.csv`
- `reports\model_comparison.png`
- `reports\confusion_matrices.png`

Training can take a few minutes, especially the MLP. Scikit-learn may print a convergence warning if the MLP reaches its iteration limit; inspect the warning and metrics rather than assuming convergence.

## 8. Start the web application
From the project root, with the virtual environment active:

```powershell
python -m uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000** in your browser. API documentation is at **http://127.0.0.1:8000/docs**.

## 9. Use the prediction interface
- Open **Price Prediction**.
- Enter all 20 specifications. Binary fields accept 0 (No) or 1 (Yes).
- Select Random Forest, Logistic Regression, Neural Network, or Compare All.
- Click **Predict price range**.
- The page displays predicted category, class number, confidence and the probability for each of the four classes.

The example values in the form are editable starter values only; they are not labelled training examples and do not imply a known correct class.

## 10. API endpoints
- `GET /` — frontend.
- `GET /api/health` — API, dataset and loaded-model status.
- `GET /api/metrics` — evaluation JSON, or an informative not-trained response.
- `POST /api/predict` — validated input and model prediction.

Example request shape:
```json
{
  "model": "Random Forest",
  "mobile": {
    "battery_power": 1200, "blue": 1, "clock_speed": 2.0, "dual_sim": 1,
    "fc": 5, "four_g": 1, "int_memory": 32, "m_dep": 0.5, "mobile_wt": 150,
    "n_cores": 4, "pc": 12, "px_height": 900, "px_width": 1600, "ram": 2000,
    "sc_h": 12, "sc_w": 7, "talk_time": 12, "three_g": 1,
    "touch_screen": 1, "wifi": 1
  }
}
```

## 11. Tests
Run:
```powershell
python -m pytest -q
```
The included tests verify dataset validation using temporary test fixtures, model-registry loading and prediction using fitted test estimators, and FastAPI health/metrics/validation behavior. These tests do not claim that the Kaggle dataset has been trained. Real training and real prediction require the authentic `data/train.csv`.

## 12. Folder structure
```text
MobilePrice-AI/
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── ml/
│   │   ├── preprocessing.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   ├── templates/index.html
│   └── static/
│       ├── css/style.css
│       └── js/script.js
├── data/                  # Place authentic train.csv here
├── models/                # Created by training
├── reports/               # Metrics and PNG charts created by training
├── tests/
├── train_models.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations / manual steps
- The Kaggle CSV is not included. Download it from the link above and place it in `data/train.csv`.
- No model binaries, accuracy scores, charts, or real predictions are bundled because no authentic dataset training was performed during packaging.
- Tailwind CSS and Chart.js are loaded from public CDNs, so those frontend assets need internet access when viewing the UI. The main layout and CSS are local.
- Prediction inputs are range-validated for sensible numeric bounds, but the dataset itself remains the definitive source for feature distributions.

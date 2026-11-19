"""Dataset validation and shared train/test preparation."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi",
]
TARGET = "price_range"
EXPECTED_COLUMNS = FEATURES + [TARGET]
CLASS_NAMES = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}

class DatasetValidationError(ValueError):
    """Raised when the supplied CSV is not suitable for this project."""

def load_and_validate_dataset(csv_path: Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Download train.csv from the Kaggle "
            "Mobile Price Classification dataset and place it in the data/ folder."
        )
    df = pd.read_csv(csv_path)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))
    if missing_columns:
        raise DatasetValidationError(f"Missing required columns: {missing_columns}")
    if extra_columns:
        raise DatasetValidationError(f"Unexpected columns found: {extra_columns}")
    if df.empty:
        raise DatasetValidationError("Dataset is empty.")
    if df[EXPECTED_COLUMNS].isna().any().any():
        counts = df[EXPECTED_COLUMNS].isna().sum()
        bad = counts[counts > 0].to_dict()
        raise DatasetValidationError(f"Missing values detected: {bad}")
    if df.duplicated().any():
        raise DatasetValidationError(
            f"Dataset contains {int(df.duplicated().sum())} duplicate rows. "
            "Resolve duplicates before training."
        )
    for col in EXPECTED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DatasetValidationError(f"Column '{col}' must be numeric.")
    if not set(df[TARGET].unique()).issubset({0, 1, 2, 3}):
        raise DatasetValidationError("price_range must contain only classes 0, 1, 2, and 3.")
    if set(df[TARGET].unique()) != {0, 1, 2, 3}:
        raise DatasetValidationError("All four target classes must be present.")
    if (df[FEATURES] < 0).any().any():
        raise DatasetValidationError("Feature values must not be negative.")
    return df

def prepare_split(df: pd.DataFrame, random_state: int = 42):
    """Return one shared stratified 80:20 split used by every algorithm."""
    X, y = df[FEATURES].copy(), df[TARGET].astype(int).copy()
    return train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

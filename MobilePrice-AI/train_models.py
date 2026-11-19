"""Convenience entry point: python train_models.py"""
from pathlib import Path
from app.ml.train import train_all

ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    train_all(ROOT/"data"/"train.csv", ROOT/"models", ROOT/"reports")

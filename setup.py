import os
import pathlib

BASE = pathlib.Path("fraud-detection")

dirs = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "reports",
]

files = [
    "notebooks/01_data_understanding.ipynb",
    "notebooks/02_eda.ipynb",
    "notebooks/03_preprocessing.ipynb",
    "notebooks/04_feature_engineering.ipynb",
    "notebooks/05_modeling.ipynb",
    "notebooks/06_evaluation.ipynb",
    "notebooks/07_shap_explainability.ipynb",
    "src/data_loader.py",
    "src/preprocessor.py",
    "src/feature_engineer.py",
    "src/evaluator.py",
    "src/__init__.py",
    "reports/.gitkeep",
    "requirements.txt",
    "README.md",
]

NOTEBOOK_TEMPLATE = """{
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}"""

REQUIREMENTS = """pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.4
seaborn==0.13.2
plotly==5.22.0
scikit-learn==1.4.2
xgboost==2.0.3
lightgbm==4.3.0
imbalanced-learn==0.12.2
shap==0.45.0
scipy==1.13.0
joblib==1.4.2
tqdm==4.66.4
openpyxl==3.1.2
jupyter==1.0.0
notebook==7.1.3
ipykernel==6.29.4
"""


def create_structure():
    # Create directories
    for d in dirs:
        path = BASE / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"  [DIR]  {path}/")

    # Create files
    for f in files:
        path = BASE / f
        if path.exists():
            print(f"  [SKIP] {path} (already exists)")
            continue

        if path.suffix == ".ipynb":
            path.write_text(NOTEBOOK_TEMPLATE, encoding="utf-8")
        elif path.name == "requirements.txt":
            path.write_text(REQUIREMENTS, encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")

        print(f"  [FILE] {path}")

    print(f"\nDone! Project scaffold created at: {BASE.resolve()}")

if __name__ == "__main__":
    create_structure()
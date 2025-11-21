"""Regenerar un GridSearchCV ligero usando los datos de `files/grading`.

Este script carga `x_train.pkl`, `y_train.pkl` y crea un pipeline con
OneHotEncoder + RandomForestClassifier. Realiza un GridSearchCV pequeño
(y rápido) y guarda `files/models/model.pkl.gz`.

Diseñado para ser rápido en CI y producir un modelo compatible con la
versión de scikit-learn del runner.
"""
import os
import gzip
import pickle

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score


def main():
    os.makedirs("files/models", exist_ok=True)

    # Cargar datos de grading
    with open(os.path.join("files", "grading", "x_train.pkl"), "rb") as f:
        x_train = pickle.load(f)
    with open(os.path.join("files", "grading", "y_train.pkl"), "rb") as f:
        y_train = pickle.load(f)

    categorical_columns = ["SEX", "EDUCATION", "MARRIAGE"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("rf", RandomForestClassifier(random_state=42))])

    # Grid pequeño para acelerar
    parameter_grid = {
        "rf__n_estimators": [50, 100],
        "rf__max_depth": [5, None],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring=make_scorer(balanced_accuracy_score),
        cv=3,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    print("Fitting small GridSearchCV on grading train data (rápido)...")
    grid.fit(x_train, y_train)

    model_path = os.path.join("files", "models", "model.pkl.gz")
    print(f"Guardando modelo en {model_path}")
    with gzip.open(model_path, "wb") as f:
        pickle.dump(grid, f)

    print("Modelo regenerado y guardado.")


if __name__ == "__main__":
    main()

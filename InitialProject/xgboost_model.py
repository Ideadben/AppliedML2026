"""Train and save an XGBoost classifier for the InitialProject dataset.

Usage example:
  python InitialProject/xgboost_model.py --train InitialProject/Data/AppML_InitialProject_train.csv

This script will try to auto-detect the target column (common names or last column),
apply simple preprocessing, train an XGBClassifier and save the model with joblib.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


COMMON_TARGET_NAMES = ("target", "label", "class", "y", "Y", "target_label")


def detect_target_column(df: pd.DataFrame, target_name: str | None = None) -> str:
    if target_name and target_name in df.columns:
        return target_name
    for name in COMMON_TARGET_NAMES:
        if name in df.columns:
            return name
    # fallback: assume last column is target
    return df.columns[-1]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Basic preprocessing: drop columns with all-null, fill numeric with median,
    # one-hot encode object/categorical columns with reasonable cardinality.
    df = df.copy()
    df.dropna(axis=1, how="all", inplace=True)

    # Separate numeric and object dtypes
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Fill numeric NaNs with median
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # For object cols, if cardinality is low, one-hot encode; otherwise try label-like cast
    for c in obj_cols:
        n_unique = df[c].nunique(dropna=False)
        if n_unique <= 50:
            dummies = pd.get_dummies(df[c].astype(str), prefix=c, dummy_na=True)
            df = pd.concat([df.drop(columns=[c]), dummies], axis=1)
        else:
            # fallback: convert to category codes (keeps memory small)
            df[c] = df[c].astype("category").cat.codes

    return df


def train_and_save(
    train_csv: str | Path,
    target: str | None = None,
    model_out: str | Path = "InitialProject/models/xgboost_model.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
):
    train_csv = Path(train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"Train file not found: {train_csv}")

    df = pd.read_csv(train_csv)
    target_col = detect_target_column(df, target)

    if target_col not in df.columns:
        raise ValueError(f"Could not find target column '{target_col}' in CSV")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    X = preprocess(X)

    # Align indices
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions and print simple metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost accuracy (test): {acc:.4f}")

    # If probabilistic prediction and binary, print AUC
    if len(np.unique(y_test)) == 2:
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            print(f"XGBoost AUC (test): {auc:.4f}")
        except Exception:
            warnings.warn("Could not compute AUC (predict_proba may be unavailable)")

    # Ensure output directory exists
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "features": X.columns.tolist()}, model_out)
    print(f"Saved model to {model_out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an XGBoost classifier on a CSV dataset")
    p.add_argument("--train", required=True, help="Path to training CSV")
    p.add_argument("--target", required=False, help="Target column name (optional)")
    p.add_argument("--out", default="InitialProject/models/xgboost_model.joblib", help="Output model path")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    p.add_argument("--random-state", type=int, default=42, help="Random state")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_save(args.train, target=args.target, model_out=args.out, test_size=args.test_size, random_state=args.random_state)

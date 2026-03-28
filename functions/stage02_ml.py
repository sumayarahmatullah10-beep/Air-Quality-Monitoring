from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .common import PipelineConfig
from .state import PipelineState

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from xgboost import XGBClassifier
    except Exception:
        XGBClassifier = None


def temporal_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def build_classification_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = [
        c
        for c in ["PM2.5", "PM10", "NO2", "O3", "CO", "Temperature", "RH", "Wind Speed", "BP", "hour"]
        if c in df.columns
    ]
    valid = df.dropna(subset=["AQI_class"]).copy()
    return valid[feature_cols], valid["AQI_class"].astype(str)


def _evaluate_classifier(model, X_train, y_train, X_test, y_test) -> Dict[str, object]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "report": classification_report(y_test, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=sorted(y_test.unique())),
    }


def _run_xgboost_same_time(df_aqi: pd.DataFrame, config: PipelineConfig) -> Dict[str, object]:
    if XGBClassifier is None:
        return {"status": "xgboost_not_installed"}

    feature_cols = [
        c
        for c in ["PM2.5", "PM10", "NO2", "O3", "CO", "Temperature", "RH", "Wind Speed", "BP", "hour"]
        if c in df_aqi.columns
    ]
    valid = df_aqi.dropna(subset=["AQI_class"]).copy()
    cls_df = valid[["datetime"] + feature_cols + ["AQI_class"]].rename(columns={"AQI_class": "target"}).reset_index(drop=True)

    tr, va, te = temporal_split(cls_df, config.train_ratio, config.val_ratio)

    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(tr[feature_cols])
    X_va = imp.transform(va[feature_cols])
    X_te = imp.transform(te[feature_cols])

    le = LabelEncoder()
    y_tr = le.fit_transform(tr["target"].astype(str))
    y_va = le.transform(va["target"].astype(str))
    y_te = le.transform(te["target"].astype(str))

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=config.random_state,
        n_jobs=1,
        eval_metric=["mlogloss", "merror"],
    )
    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], verbose=False)

    pred = model.predict(X_te)
    y_true_lbl = le.inverse_transform(y_te)
    y_pred_lbl = le.inverse_transform(pred)

    return {
        "macro_f1": float(f1_score(y_true_lbl, y_pred_lbl, average="macro")),
        "report": classification_report(y_true_lbl, y_pred_lbl, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_lbl, y_pred_lbl, labels=list(le.classes_)),
        "classes": list(le.classes_),
        "evals_result": model.evals_result(),
        "model": model,
    }


def _run_xgboost_future(df_aqi: pd.DataFrame, config: PipelineConfig) -> Dict[str, object]:
    if XGBClassifier is None:
        return {"status": "xgboost_not_installed"}

    future_h = config.forecast_horizon
    df = df_aqi.copy().sort_values("datetime").reset_index(drop=True)
    df[f"AQI_class_t_plus_{future_h}"] = df["AQI_class"].shift(-future_h)

    base_features = [
        c
        for c in ["PM2.5", "PM10", "NO2", "O3", "CO", "Temperature", "RH", "Wind Speed", "BP", "hour"]
        if c in df.columns
    ]
    for col in [c for c in ["PM2.5", "PM10", "NO2", "O3"] if c in df.columns]:
        df[f"{col}_lag_1"] = df[col].shift(1)
        df[f"{col}_lag_6"] = df[col].shift(6)
        df[f"{col}_lag_24"] = df[col].shift(24)

    feature_cols = [
        c
        for c in df.columns
        if c in base_features or c.endswith("_lag_1") or c.endswith("_lag_6") or c.endswith("_lag_24")
    ]

    target_col = f"AQI_class_t_plus_{future_h}"
    df = df.dropna(subset=[target_col]).copy()

    tr, va, te = temporal_split(df, config.train_ratio, config.val_ratio)

    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(tr[feature_cols])
    X_va = imp.transform(va[feature_cols])
    X_te = imp.transform(te[feature_cols])

    le = LabelEncoder()
    y_tr = le.fit_transform(tr[target_col].astype(str))
    y_va = le.transform(va[target_col].astype(str))
    y_te = le.transform(te[target_col].astype(str))

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        n_estimators=450,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=config.random_state,
        n_jobs=1,
        eval_metric=["mlogloss", "merror"],
    )
    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], verbose=False)

    pred = model.predict(X_te)
    y_true_lbl = le.inverse_transform(y_te)
    y_pred_lbl = le.inverse_transform(pred)

    return {
        "macro_f1": float(f1_score(y_true_lbl, y_pred_lbl, average="macro")),
        "report": classification_report(y_true_lbl, y_pred_lbl, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_lbl, y_pred_lbl, labels=list(le.classes_)),
        "classes": list(le.classes_),
        "evals_result": model.evals_result(),
        "model": model,
    }


def run_stage02(state: PipelineState, config: PipelineConfig) -> PipelineState:
    df_aqi = state.data["df_aqi"]

    X_cls, y_cls = build_classification_dataset(df_aqi)
    cls_df = X_cls.copy()
    cls_df["target"] = y_cls.values

    train_df, _, test_df = temporal_split(cls_df, config.train_ratio, config.val_ratio)

    X_train = train_df.drop(columns="target")
    y_train = train_df["target"]
    X_test = test_df.drop(columns="target")
    y_test = test_df["target"]

    logreg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=config.random_state,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                ),
            ),
        ]
    )

    baseline_results = {
        "logistic_regression": _evaluate_classifier(logreg, X_train, y_train, X_test, y_test),
        "random_forest": _evaluate_classifier(rf, X_train, y_train, X_test, y_test),
    }

    xgb_same = _run_xgboost_same_time(df_aqi, config)
    xgb_future = _run_xgboost_future(df_aqi, config)

    summary_rows = []
    for model_name, res in baseline_results.items():
        summary_rows.append({"task": "AQI same-time", "model": model_name, "metric": "macro_f1", "value": res["macro_f1"]})
    if "macro_f1" in xgb_same:
        summary_rows.append({"task": "AQI same-time", "model": "xgboost", "metric": "macro_f1", "value": xgb_same["macro_f1"]})
    if "macro_f1" in xgb_future:
        summary_rows.append({"task": f"AQI t+{config.forecast_horizon}", "model": "xgboost", "metric": "macro_f1", "value": xgb_future["macro_f1"]})

    state.put_metrics(
        stage02_baselines=baseline_results,
        stage02_xgb_same_time=xgb_same,
        stage02_xgb_future=xgb_future,
        stage02_summary=pd.DataFrame(summary_rows),
    )
    state.put_models(stage02_logreg=logreg, stage02_rf=rf)

    if config.save_artifacts:
        out = config.artifact_dir / "stage02"
        out.mkdir(parents=True, exist_ok=True)
        state.metrics["stage02_summary"].to_csv(out / "stage02_summary.csv", index=False)
        state.put_artifacts(stage02_dir=out)

    return state


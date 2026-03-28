from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from sklearn.metrics import root_mean_squared_error
except Exception:
    def root_mean_squared_error(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from .common import PipelineConfig
from .state import PipelineState

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from xgboost import XGBRegressor
    except Exception:
        XGBRegressor = None


def create_lagged_features(df: pd.DataFrame, cols: List[str], max_lag: int = 24) -> pd.DataFrame:
    temp = df.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        for col in cols:
            for lag in range(1, max_lag + 1):
                temp[f"{col}_lag_{lag}"] = temp[col].shift(lag)
            temp[f"{col}_roll6"] = temp[col].rolling(window=6, min_periods=1).mean().shift(1)
            temp[f"{col}_roll24"] = temp[col].rolling(window=24, min_periods=1).mean().shift(1)

        temp["hour_sin"] = np.sin(2 * np.pi * temp["datetime"].dt.hour / 24)
        temp["hour_cos"] = np.cos(2 * np.pi * temp["datetime"].dt.hour / 24)
        temp["dow_sin"] = np.sin(2 * np.pi * temp["datetime"].dt.dayofweek / 7)
        temp["dow_cos"] = np.cos(2 * np.pi * temp["datetime"].dt.dayofweek / 7)
    return temp


def create_multioutput_targets(df: pd.DataFrame, targets: List[str], horizon: int = 24) -> pd.DataFrame:
    temp = df.copy()
    for target in targets:
        for h in range(1, horizon + 1):
            temp[f"{target}_t_plus_{h}"] = temp[target].shift(-h)
    return temp


def build_forecasting_matrix(df: pd.DataFrame, targets: List[str], horizon: int, max_lag: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    base = create_lagged_features(df, cols=targets, max_lag=max_lag)
    full = create_multioutput_targets(base, targets=targets, horizon=horizon)

    y_cols = [f"{t}_t_plus_{h}" for t in targets for h in range(1, horizon + 1)]
    feature_cols = [c for c in full.columns if c not in ["datetime", "AQI_class"] + y_cols]

    matrix = full.dropna(subset=y_cols).copy()
    return matrix[feature_cols], matrix[y_cols], y_cols


def recursive_multi_step_predict(
    model: Pipeline,
    X_start: pd.DataFrame,
    targets: List[str],
    horizon: int,
    max_lag: int,
) -> np.ndarray:
    col_idx: Dict[str, int] = {c: i for i, c in enumerate(X_start.columns)}
    lag_idx: Dict[str, List[int]] = {}
    roll6_idx: Dict[str, int] = {}
    roll24_idx: Dict[str, int] = {}

    for target in targets:
        lag_idx[target] = [col_idx[f"{target}_lag_{lag}"] for lag in range(1, max_lag + 1) if f"{target}_lag_{lag}" in col_idx]
        if f"{target}_roll6" in col_idx:
            roll6_idx[target] = col_idx[f"{target}_roll6"]
        if f"{target}_roll24" in col_idx:
            roll24_idx[target] = col_idx[f"{target}_roll24"]

    X_arr = X_start.to_numpy(dtype=float).copy()
    n_rows = X_arr.shape[0]
    n_targets = len(targets)
    preds = np.zeros((n_rows, n_targets * horizon), dtype=float)

    for step in range(horizon):
        step_pred = model.predict(pd.DataFrame(X_arr, columns=X_start.columns))

        for t_idx, _ in enumerate(targets):
            preds[:, t_idx * horizon + step] = step_pred[:, t_idx]

        for t_idx, target in enumerate(targets):
            idxs = lag_idx[target]
            if not idxs:
                continue

            if len(idxs) > 1:
                X_arr[:, idxs[1:]] = X_arr[:, idxs[:-1]]
            X_arr[:, idxs[0]] = step_pred[:, t_idx]

            if target in roll6_idx:
                X_arr[:, roll6_idx[target]] = X_arr[:, idxs[: min(6, len(idxs))]].mean(axis=1)
            if target in roll24_idx:
                X_arr[:, roll24_idx[target]] = X_arr[:, idxs[: min(24, len(idxs))]].mean(axis=1)

    return preds


def evaluate_forecast(y_true: pd.DataFrame, y_pred: np.ndarray, y_cols: List[str]) -> pd.DataFrame:
    pred_df = pd.DataFrame(y_pred, columns=y_cols, index=y_true.index)
    out = []
    for col in y_cols:
        out.append(
            {
                "target": col,
                "MAE": mean_absolute_error(y_true[col], pred_df[col]),
                "RMSE": root_mean_squared_error(y_true[col], pred_df[col]),
            }
        )
    return pd.DataFrame(out)


def run_stage04(state: PipelineState, config: PipelineConfig) -> PipelineState:
    df_aqi = state.data["df_aqi"]
    targets = [c for c in ["PM2.5", "PM10", "NO2", "O3"] if c in df_aqi.columns]
    X_fc, y_fc, y_cols = build_forecasting_matrix(
        df_aqi,
        targets,
        horizon=config.forecast_horizon,
        max_lag=config.forecast_max_lag,
    )

    split = int(len(X_fc) * 0.80)
    X_train, X_test = X_fc.iloc[:split], X_fc.iloc[split:]
    y_train, y_test = y_fc.iloc[:split], y_fc.iloc[split:]

    lin = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", MultiOutputRegressor(LinearRegression())),
        ]
    )
    lin.fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)
    lin_metrics = evaluate_forecast(y_test, y_pred_lin, y_cols)
    lin_metrics["model"] = "linear_regression"

    y_train_next = y_train[[f"{t}_t_plus_1" for t in targets]]
    ridge_recursive = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", MultiOutputRegressor(Ridge(alpha=1.0))),
        ]
    )
    ridge_recursive.fit(X_train, y_train_next)
    y_pred_ridge = recursive_multi_step_predict(
        ridge_recursive,
        X_test,
        targets=targets,
        horizon=config.forecast_horizon,
        max_lag=config.forecast_max_lag,
    )
    ridge_metrics = evaluate_forecast(y_test, y_pred_ridge, y_cols)
    ridge_metrics["model"] = "ridge_recursive"

    compare = [lin_metrics, ridge_metrics]

    xgb_metrics = None
    if config.stage04_enable_xgboost and XGBRegressor is not None:
        xgb = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "reg",
                    MultiOutputRegressor(
                        XGBRegressor(
                            n_estimators=config.stage04_xgb_estimators,
                            learning_rate=0.05,
                            max_depth=config.stage04_xgb_max_depth,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            random_state=config.random_state,
                            n_jobs=1,
                            objective="reg:squarederror",
                        )
                    ),
                ),
            ]
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        xgb_metrics = evaluate_forecast(y_test, y_pred_xgb, y_cols)
        xgb_metrics["model"] = "xgboost"
        compare.append(xgb_metrics)
        state.put_models(stage04_xgb=xgb)

    compare_df = pd.concat(compare, ignore_index=True)

    state.put_data(stage04_X_test=X_test, stage04_y_test=y_test)
    state.put_models(stage04_linear=lin, stage04_ridge_recursive=ridge_recursive)
    state.put_metrics(
        stage04_linear_metrics=lin_metrics,
        stage04_ridge_recursive_metrics=ridge_metrics,
        stage04_xgb_metrics=xgb_metrics,
        stage04_compare_metrics=compare_df,
    )

    if config.save_artifacts:
        out = config.artifact_dir / "stage04"
        out.mkdir(parents=True, exist_ok=True)
        compare_df.to_csv(out / "forecast_metrics.csv", index=False)
        state.put_artifacts(stage04_dir=out)

    return state

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .common import PipelineConfig
from .state import PipelineState

try:
    from tensorflow.keras import Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import (
        Bidirectional,
        Conv1D,
        Dense,
        Dropout,
        GRU,
        GlobalAveragePooling1D,
        Input,
        LSTM,
    )
except Exception:
    try:
        from keras import Model
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from keras.layers import (
            Bidirectional,
            Conv1D,
            Dense,
            Dropout,
            GRU,
            GlobalAveragePooling1D,
            Input,
            LSTM,
        )
    except Exception:
        Model = None
        EarlyStopping = None
        ReduceLROnPlateau = None
        Bidirectional = Conv1D = Dense = Dropout = GRU = GlobalAveragePooling1D = Input = LSTM = None


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["datetime"]
    out["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)
    return out


def make_sequences_same_time(X_arr: np.ndarray, y_arr: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X_seq, y_seq = [], []
    max_i = len(X_arr) - seq_len + 1
    for i in range(max_i):
        X_seq.append(X_arr[i : i + seq_len])
        y_seq.append(y_arr[i + seq_len - 1])
    return np.array(X_seq), np.array(y_seq)


def _build_cnn_classifier(seq_len: int, n_features: int, n_classes: int) -> Model:
    inp = Input(shape=(seq_len, n_features))
    x = Conv1D(64, kernel_size=3, padding="causal", activation="relu")(inp)
    x = Conv1D(128, kernel_size=3, padding="causal", activation="relu")(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(96, activation="relu")(x)
    x = Dropout(0.25)(x)
    out = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out, name="cnn_aqi_classifier")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def _build_bilstm_classifier(seq_len: int, n_features: int, n_classes: int) -> Model:
    inp = Input(shape=(seq_len, n_features))
    x = Bidirectional(LSTM(64, return_sequences=False))(inp)
    x = Dropout(0.25)(x)
    x = Dense(96, activation="relu")(x)
    x = Dropout(0.25)(x)
    out = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out, name="bilstm_aqi_classifier")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def _build_gru_classifier(seq_len: int, n_features: int, n_classes: int) -> Model:
    inp = Input(shape=(seq_len, n_features))
    x = GRU(64, return_sequences=False)(inp)
    x = Dropout(0.25)(x)
    x = Dense(96, activation="relu")(x)
    x = Dropout(0.25)(x)
    out = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out, name="gru_aqi_classifier")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def _fit_model(model: Model, X_tr, y_tr, X_va, y_va, config: PipelineConfig):
    es = EarlyStopping(monitor="val_loss", patience=config.dl_patience, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, config.dl_patience - 1), min_lr=1e-5)
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=config.dl_epochs,
        batch_size=config.dl_batch_size,
        callbacks=[es, rlrop],
        verbose=0,
    )
    return history


def _evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, class_names: np.ndarray) -> Dict[str, object]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def run_stage03(state: PipelineState, config: PipelineConfig) -> PipelineState:
    if Model is None:
        state.put_metrics(stage03_status="keras_or_tensorflow_not_installed")
        return state

    df_aqi = state.data["df_aqi"].copy().sort_values("datetime")

    base_features = [
        c
        for c in ["PM2.5", "PM10", "NO2", "O3", "CO", "Temperature", "RH", "Wind Speed", "BP"]
        if c in df_aqi.columns
    ]
    use_cols = ["datetime", "AQI_class"] + base_features
    dl_df = df_aqi[use_cols].tail(config.dl_tail_rows).reset_index(drop=True)
    dl_df = dl_df.dropna(subset=["AQI_class"]).copy()
    dl_df = _add_time_features(dl_df)

    feature_cols = [c for c in dl_df.columns if c not in ["datetime", "AQI_class"]]
    dl_df[feature_cols] = dl_df[feature_cols].interpolate(limit_direction="both").ffill().bfill()

    X_raw = dl_df[feature_cols].values.astype(float)
    y_raw = dl_df["AQI_class"].astype(str).values

    X_seq, y_seq_str = make_sequences_same_time(X_raw, y_raw, config.dl_seq_len)
    if len(X_seq) == 0:
        state.put_metrics(stage03_status="insufficient_sequences_for_split")
        return state

    # Fixed split requested: 70% train, 15% validation, 15% test.
    n_seq = len(X_seq)
    tr_end = int(n_seq * 0.70)
    va_end = int(n_seq * 0.85)

    X_tr, X_va, X_te = X_seq[:tr_end], X_seq[tr_end:va_end], X_seq[va_end:]
    y_tr_str, y_va_str, y_te_str = y_seq_str[:tr_end], y_seq_str[tr_end:va_end], y_seq_str[va_end:]

    if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
        state.put_metrics(stage03_status="insufficient_sequences_for_split")
        return state

    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, X_tr.shape[-1]))

    def _scale(X):
        flat = X.reshape(-1, X.shape[-1])
        return scaler.transform(flat).reshape(X.shape)

    X_tr = _scale(X_tr)
    X_va = _scale(X_va)
    X_te = _scale(X_te)

    le = LabelEncoder()
    le.fit(y_tr_str)

    known_val_mask = np.isin(y_va_str, le.classes_)
    known_test_mask = np.isin(y_te_str, le.classes_)
    if known_val_mask.sum() == 0 or known_test_mask.sum() == 0:
        state.put_metrics(stage03_status="no_known_labels_in_val_or_test")
        return state

    X_va = X_va[known_val_mask]
    X_te = X_te[known_test_mask]
    y_va_str = y_va_str[known_val_mask]
    y_te_str = y_te_str[known_test_mask]

    y_tr = le.transform(y_tr_str)
    y_va = le.transform(y_va_str)
    y_te = le.transform(y_te_str)

    n_classes = len(le.classes_)
    class_ids = np.arange(n_classes)

    models: Dict[str, object] = {}
    histories: Dict[str, object] = {}
    results: Dict[str, Dict[str, object]] = {}
    summary_rows = []

    model_builders = {
        "cnn_cls": _build_cnn_classifier,
        "bilstm_cls": _build_bilstm_classifier,
        "gru_cls": _build_gru_classifier,
    }

    for name, builder in model_builders.items():
        model = builder(config.dl_seq_len, len(feature_cols), n_classes)
        history = _fit_model(model, X_tr, y_tr, X_va, y_va, config)
        pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
        res = _evaluate_classifier(y_te, pred, class_ids, le.classes_)

        models[name] = model
        histories[name] = history.history
        results[name] = res
        summary_rows.append(
            {
                "task": "AQI same-time",
                "model": name,
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
                "weighted_f1": res["weighted_f1"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    best_model_name = summary_df.iloc[0]["model"]

    state.put_models(**{f"stage03_{k}": v for k, v in models.items()})
    state.put_data(
        stage03_label_classes=list(le.classes_),
        stage03_best_model=best_model_name,
        stage03_split={"train": len(X_tr), "val": len(X_va), "test": len(X_te)},
    )
    state.put_metrics(stage03_histories=histories, stage03_results=results, stage03_summary=summary_df)

    if config.save_artifacts:
        out = config.artifact_dir / "stage03"
        out.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out / "dl_summary.csv", index=False)
        state.put_artifacts(stage03_dir=out)

    return state

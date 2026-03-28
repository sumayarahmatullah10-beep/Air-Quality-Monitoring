from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@dataclass
class PipelineConfig:
    base_dir: Path = field(default_factory=Path.cwd)
    csv_name: str = "gazipur_air_quality_data.csv"
    xlsx_name: str = "air_quality_index_dataset.xlsx"
    random_state: int = 42
    save_artifacts: bool = False
    artifact_dir: Path = field(default_factory=lambda: Path.cwd() / "artifacts")

    missing_threshold: float = 0.60
    winsor_k: float = 3.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    forecast_horizon: int = 24
    forecast_max_lag: int = 24
    stage04_enable_xgboost: bool = False
    stage04_xgb_estimators: int = 120
    stage04_xgb_max_depth: int = 4

    dl_seq_len: int = 48
    dl_horizon: int = 24
    dl_tail_rows: int = 12000
    dl_epochs: int = 15
    dl_batch_size: int = 64
    dl_patience: int = 3

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.artifact_dir = Path(self.artifact_dir)

    @property
    def csv_path(self) -> Path:
        return self.base_dir / self.csv_name

    @property
    def xlsx_path(self) -> Path:
        return self.base_dir / self.xlsx_name

    @property
    def key_pollutants(self) -> List[str]:
        return ["PM2.5", "PM10", "O3", "NO2", "SO2", "CO"]

    @property
    def meteo_cols(self) -> List[str]:
        return ["Temperature", "RH", "Wind Speed", "Wind Dir", "BP", "Rain", "Solar Rad"]


def setup_environment(config: PipelineConfig) -> None:
    np.random.seed(config.random_state)
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (12, 5)
    if config.save_artifacts:
        config.artifact_dir.mkdir(parents=True, exist_ok=True)

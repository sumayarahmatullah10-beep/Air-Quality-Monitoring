from __future__ import annotations

import pandas as pd

from .common import PipelineConfig
from .state import PipelineState


def run_stage05(state: PipelineState, config: PipelineConfig) -> PipelineState:
    df_aqi = state.data["df_aqi"]

    try:
        from scipy.stats import kruskal, spearmanr
        from statsmodels.tsa.stattools import adfuller, kpss
    except Exception as exc:
        state.put_metrics(stage05_status=f"missing_dependency: {exc}")
        return state

    rows = []
    series = df_aqi["PM2.5"].dropna()

    adf_stat, adf_p, *_ = adfuller(series)
    rows.append({"test": "ADF (PM2.5)", "statistic": float(adf_stat), "p_value": float(adf_p)})

    kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
    rows.append({"test": "KPSS (PM2.5)", "statistic": float(kpss_stat), "p_value": float(kpss_p)})

    groups = [grp["PM2.5"].dropna().values for _, grp in df_aqi.groupby(df_aqi["datetime"].dt.hour)]
    k_stat, k_p = kruskal(*groups)
    rows.append({"test": "Kruskal-Wallis PM2.5 by hour", "statistic": float(k_stat), "p_value": float(k_p)})

    if "RH" in df_aqi.columns:
        rho, rho_p = spearmanr(df_aqi["PM2.5"], df_aqi["RH"], nan_policy="omit")
        rows.append({"test": "Spearman PM2.5 vs RH", "statistic": float(rho), "p_value": float(rho_p)})

    stats_df = pd.DataFrame(rows)
    state.put_metrics(stage05_results=stats_df)

    if config.save_artifacts:
        out = config.artifact_dir / "stage05"
        out.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(out / "statistical_tests.csv", index=False)
        state.put_artifacts(stage05_dir=out)

    return state



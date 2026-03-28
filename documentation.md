# Air Quality Monitoring Analysis Pipeline Documentation

## Overview

This project implements a comprehensive air quality monitoring analysis pipeline using Python. The pipeline is structured as a modular, stage-wise workflow that processes air quality data from Gazipur, Bangladesh, to perform data cleaning, machine learning classification, deep learning modeling, time-series forecasting, and statistical analysis.

The pipeline is orchestrated through a Jupyter notebook (`main.ipynb`) that imports and runs functions from the `functions/` directory. Each stage builds upon the previous one, passing data and metrics through a shared `PipelineState` object.

## Project Structure

```
d:\Air Quality Monitoring/
├── main.ipynb                    # Main orchestrator notebook
├── requirements.txt              # Python dependencies
├── gazipur_air_quality_data.csv  # Raw air quality dataset
├── functions/
│   ├── __init__.py              # Module imports
│   ├── common.py                # Shared configuration and utilities
│   ├── state.py                 # Pipeline state management
│   ├── stage01_data.py          # Data loading and preprocessing
│   ├── stage02_ml.py            # Machine learning classification
│   ├── stage03_dl.py            # Deep learning classification
│   ├── stage04_timeseries.py    # Time-series forecasting
│   └── stage05_stats.py         # Statistical analysis
└── artifacts/                   # (Created if save_artifacts=True)
```

## Dependencies

The project requires the following Python packages (listed in `requirements.txt`):

- Core data science: `pandas`, `numpy`, `matplotlib`, `seaborn`
- Machine learning: `scikit-learn`, `xgboost` (optional)
- Deep learning: `tensorflow` or `keras`
- Statistics: `scipy`, `statsmodels`

Install dependencies with:
```bash
pip install -r requirements.txt
# For GPU support with TensorFlow:
pip install tensorflow[and-cuda]
```

## Configuration

The pipeline is configured through the `PipelineConfig` dataclass in `functions/common.py`. Key parameters include:

- **Data paths**: CSV file location, artifact directory
- **Data processing**: Missing threshold (60%), winsorization k-factor (3.0)
- **Train/validation/test splits**: 70%/15%/15%
- **Forecasting**: Horizon (24 hours), max lag (24 hours)
- **Deep learning**: Sequence length (48), epochs (15), batch size (64)
- **XGBoost settings**: Estimators, max depth for forecasting

## Data Description

The dataset (`gazipur_air_quality_data.csv`) contains hourly air quality measurements from Gazipur, Bangladesh, with the following key columns:

- **Date/Time**: Date and Time columns (normalized to datetime)
- **Pollutants**: PM2.5, PM10, NO2, O3, CO, SO2
- **Meteorological**: Temperature, RH (Relative Humidity), Wind Speed, Wind Direction, BP (Barometric Pressure), Rain, Solar Radiation

The pipeline derives Air Quality Index (AQI) values and classes based on EPA breakpoints for each pollutant.

## Workflow Stages

### Stage 1: Data Loading, Filtering, Preprocessing, and Description

**Purpose**: Load raw data, clean it, derive AQI, and generate quality reports.

**Key Functions** (`stage01_data.py`):
- `load_air_quality_data()`: Reads CSV, normalizes datetime from Date/Time columns
- `describe_data_quality()`: Generates missing value and uniqueness statistics
- `drop_sparse_initial_period()`: Removes early data with high missing rates
- `impute_time_series()`: Time-based interpolation and forward/backward fill
- `winsorize_iqr()`: Outlier treatment using IQR method
- `derive_aqi()`: Calculates AQI sub-indices and overall AQI class

**Outputs**:
- Clean dataset with AQI values and classes
- Data quality reports (raw and clean)
- Overview metrics: shapes, date ranges, AQI class distribution

### Stage 2: Machine Learning Classification

**Purpose**: Train ML models to classify AQI categories using current pollutant measurements.

**Models**:
- **Logistic Regression**: Baseline linear classifier
- **Random Forest**: Ensemble tree-based classifier  
- **XGBoost**: Gradient boosting for same-time and future (t+24) AQI prediction

**Key Features**:
- Temporal split (70% train, 15% val, 15% test)
- Feature engineering: pollutant concentrations, hour of day
- Evaluation: Macro F1-score, classification reports, confusion matrices

**Outputs**:
- Trained models and evaluation metrics
- Summary table comparing model performance

### Stage 3: Deep Learning Training and Results

**Purpose**: Train neural network models for sequence-based AQI classification.

**Models**:
- **CNN Classifier**: Convolutional neural network with causal convolutions
- **BiLSTM Classifier**: Bidirectional LSTM for sequence modeling
- **GRU Classifier**: Gated Recurrent Unit classifier

**Key Features**:
- Sequence length: 48 hours of historical data
- Feature engineering: pollutant lags, cyclical time features (sin/cos hour/day)
- Training: Early stopping, learning rate reduction, validation monitoring
- Evaluation: Accuracy, macro/weighted F1, confusion matrices

**Outputs**:
- Trained DL models and training histories
- Performance comparison across models
- Best model selection based on macro F1

### Stage 4: Time-Series Forecasting

**Purpose**: Multi-step ahead forecasting of pollutant concentrations.

**Models**:
- **Linear Regression**: Direct multi-output prediction
- **Ridge Regression**: Recursive multi-step forecasting with regularization
- **XGBoost Regressor**: Tree-based forecasting (optional)

**Key Features**:
- Feature engineering: Lagged values (1-24 hours), rolling means (6h, 24h), cyclical time features
- Recursive prediction: Uses own predictions as input for future steps
- Horizon: 24-hour ahead forecasts
- Evaluation: MAE, RMSE per target and horizon

**Outputs**:
- Forecasted values and error metrics
- Model comparison across forecasting horizons

### Stage 5: Statistical Analysis

**Purpose**: Perform statistical tests on the time series data.

**Tests**:
- **Stationarity Tests**: ADF (Augmented Dickey-Fuller) and KPSS tests on PM2.5
- **Non-parametric Tests**: Kruskal-Wallis test for hourly PM2.5 differences
- **Correlation**: Spearman rank correlation between PM2.5 and relative humidity

**Outputs**:
- Statistical test results with p-values
- Summary table of hypothesis tests

## Unified Summary

The final section of `main.ipynb` aggregates results from all stages into a unified dashboard, focusing on AQI same-time classification performance across ML and DL models.

## Running the Pipeline

1. **Setup**: Install dependencies and ensure data file is present
2. **Configuration**: Modify `PipelineConfig` in the notebook as needed
3. **Execution**: Run cells in `main.ipynb` sequentially
4. **Artifacts**: Set `save_artifacts=True` to export intermediate results

## Key Design Patterns

- **Modular Architecture**: Each stage is self-contained with clear inputs/outputs
- **State Management**: `PipelineState` passes data, models, and metrics between stages
- **Temporal Validation**: Time-ordered splits prevent data leakage
- **Error Handling**: Graceful degradation when dependencies are missing
- **Reproducibility**: Fixed random seeds and configuration-driven parameters

## Performance Considerations

- Deep learning uses GPU acceleration if TensorFlow with CUDA is available
- XGBoost can be disabled for faster execution
- Data subsetting (tail rows) for DL to manage memory
- Parallel processing where possible (model training)

This pipeline provides a complete framework for air quality analysis, from raw data processing to advanced predictive modeling, suitable for environmental monitoring and research applications.
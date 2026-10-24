# Flight Delay Risk Predictor — Backend & ML Engine

Production Machine Learning and FastAPI inference engine for schedule-only pre-departure flight delay risk stratification.

## Architecture & Subsystems
- **`src/`**: Operational ML pipeline (data cleaning, temporal splitting, feature engineering, monthly calibration, and inference predictor).
- **`src/research/`**: Offline experiments, LightGBM/CatBoost training baselines, hyperparameter tuning, and ablation studies.
- **`api/`**: FastAPI REST server providing `/health`, `/predict`, and `/predict/batch` endpoints.
- **`artifacts/`**: Serialized production artifacts (Champion CatBoost v2 model, Platt calibrator, and serving lookups).
- **`tests/`**: Pytest unit and integration test suite.
- **`scripts/`**: Verification, drift monitoring, benchmark audits, and build scripts.
- **`reports/`**: Pre-registered benchmark metrics, evaluation manifests, and figures.

## Quickstart
```bash
# 1. Install dependencies and editable package
pip install -r requirements.txt
pip install -e .

# 2. Run API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 3. Run test suite
pytest tests/ -v
```

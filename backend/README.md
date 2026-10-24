# Flight Delay Risk Predictor — Backend & ML Engine

Production Machine Learning, pipeline verification, and FastAPI inference engine for schedule-only pre-departure flight delay risk stratification.

---

## 1. System Overview & Working Directory

The `backend/` directory encapsulates the offline research pipeline, model distillation tools, artifact compilation, and the reference containerized FastAPI server.

> [!IMPORTANT]
> **Working Directory Notice**: All commands documented below should be executed from within the `backend/` directory:
> ```bash
> cd backend
> ```
> Alternatively, from the repository root, prefix commands with `backend/` (e.g., `pytest backend/tests/ -v`).

---

## 2. Monorepo Role & Subsystems

- **`src/`**: Operational ML pipeline core.
  - `src/cleaning.py`: BTS raw data cleaning and anomaly filtering.
  - `src/split.py`: Strict zero-leakage temporal dataset partition (Train: 2019/2021, Val: 2022, Test: Jan–Aug 2023).
  - `src/features.py`: 33-feature leakage-free schedule feature engineering pipeline.
  - `src/baseline.py`: Baseline dummy, logistic regression, and evaluation models.
  - `src/serving/`: Reference containerized predictor (`FlightPredictor`), Bayesian prior lookups, and geodesic Haversine distance calculator.
- **`src/research/`**: Offline research experiments, LightGBM/CatBoost training scripts, Optuna Bayesian hyperparameter studies, and ablation benchmarks.
- **`api/`**: Full containerized FastAPI application (`main.py`) providing `/health`, `/options`, `/metadata`, `/predict`, and `/predict/batch` endpoints.
- **`artifacts/`**: Serialized production artifacts:
  - `models/champion_v2.joblib`: Frozen Champion CatBoost v2 pipeline (178 trees, depth 6).
  - `models/model_metadata_v2.json`: Model hyperparameters, feature lists, and training hashes.
  - `serving/`: Serialized lookup tables (`vocab.json`, `route_lookup.json`, `airport_coords.json`, `priors/`).
- **`tests/`**: Comprehensive Pytest suite comprising **34 unit, integration, and distillation parity tests**.
- **`scripts/`**: Model distillation scripts (`export_model_to_python.py`), claims verification (`verify_docs_numbers.py`), serving approximation audits, and benchmark evaluation.
- **`reports/`**: Empirical evaluation manifests, pre-registered benchmarks, and day-clustered bootstrap metrics.
- **`notebooks/`**: Exploratory data analysis, calibration curves, and feature ablation notebooks.

---

## 3. Dual API Parity & Pydantic Input Hardening

The repository implements a **dual API architecture**:
1. **Full-Stack Container API** (`backend/api/main.py`): Powers local Docker deployments and batch evaluation using the Joblib pipeline.
2. **Serverless Edge API** (`frontend/api/index.py`): Powers the production Vercel deployment using the distilled pure-Python engine.

Both endpoints maintain **100% functional and validation parity**:
- **Semantic Route Validation (`ORIGIN != DEST`)**: Both APIs implement a Pydantic `@model_validator(mode="after")` enforcing that origin and destination airport codes are distinct. Submitting identical airports (e.g., `ORD` $\to$ `ORD`) returns a structured **HTTP 422 Unprocessable Entity** error with specific field mapping.
- **Strict Military Time Validation**: Scheduled departure and arrival times are validated to lie within $[0, 2359]$ with minutes strictly $< 60$.
- **ISO Date Parsing**: Flight dates must match the standard ISO `YYYY-MM-DD` pattern.
- **Batch Ceiling**: Batch scoring endpoints enforce a hard ceiling of $\le 500$ flights per request.

---

## 4. Comprehensive Test Suite (34/34 Passing Tests)

The test suite covers unit, integration, validation, and automated distillation parity testing across 4 test modules:

| Test Module | Tests | Scope & Coverage |
| :--- | :---: | :--- |
| **`tests/test_api.py`** | 12 | Health probes, metadata endpoints, single/batch prediction schemas, Pydantic 422 validation (`ORIGIN != DEST`, military times), CORS headers, and request logging privacy. |
| **`tests/test_model_distillation_parity.py`** | 3 | Automated bit-level parity verification across 200 diverse nationwide flight itineraries, analytical Platt scaling verification against Scikit-Learn, and unseen cold-start route Haversine fallback parity. |
| **`tests/test_pipeline.py`** | 6 | BTS raw data cleaning, strict temporal split integrity (no 2020 contamination), cyclical feature encoding, and leakage-free feature transformers. |
| **`tests/test_predictor.py`** | 13 | Predictor initialization, feature alignment, empirical Bayes prior lookups, geodesic distance approximation, directional log-odds attribution, and decile assignment. |

### Running the Test Suite

From the `backend/` directory:

```bash
cd backend
pytest tests/ -v
```

All 34 tests execute in approximately 30 seconds.

---

## 5. Local Setup & Quickstart

### 5.1 Python Virtual Environment

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### 5.2 Launch Local API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- API Base URL: `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 5.3 Docker Containerized Execution

```bash
# Build and run with Docker Compose
docker compose build
docker compose up -d

# Run tests inside container
docker compose run --rm api pytest tests/ -v
```

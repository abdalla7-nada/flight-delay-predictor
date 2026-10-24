# Flight Delay Risk Predictor

*Production-Grade Schedule-Only ML Inference Engine & Analytics Dashboard*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![CI Status](https://github.com/abdalla7-nada/flight-delay-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/abdalla7-nada/flight-delay-predictor/actions/workflows/ci.yml)
[![Framework](https://img.shields.io/badge/Frontend-Next.js%2016%20(Turbopack)-black.svg)](https://nextjs.org/)
[![API](https://img.shields.io/badge/Inference-FastAPI%20(ASGI%20Serverless)-009688.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Champion-CatBoost%20v2%20(Pure%20Python)-green.svg)](https://catboost.ai/)
[![Payload](https://img.shields.io/badge/Payload-48.4%20MB%20%2F%20250%20MB%20Limit-success.svg)](https://vercel.com)
[![Tests](https://img.shields.io/badge/Tests-31%2F31%20Passing-success.svg)](https://github.com/abdalla7-nada/flight-delay-predictor)

**Author**: Abdallah Nada  
**Release Version**: v2.0 (Unified Vercel Serverless Architecture & Next.js 16 Interface)  
**Task**: Schedule-Only Pre-Departure Flight Delay Risk Stratification  
**Live Production Application**: [https://flight-delay-predictor.vercel.app](https://flight-delay-predictor.vercel.app) *(Vercel)*  

---

## 1. Problem Statement & Motivation

Flight delays generate billions of dollars annually in commercial disruption, missed passenger connections, and operational friction. Most published machine learning models for flight delay rely on tactical, post-departure, or real-time features—such as actual departure delays (`DEP_DELAY`), taxi-out duration, or live weather radar. While those features achieve high nominal accuracy, they are useless at the time of ticket booking (weeks or months in advance), and introducing them into pre-departure models constitutes catastrophic data leakage.

This project addresses the true pre-departure challenge: **Can flight delay vulnerability be meaningfully separated at booking time using only schedule coordinates?**

The target is defined as:

$$\text{DELAY} = \mathbf{1}(\mathrm{ARR\_DELAY} > 15 \text{ minutes})$$

### Threshold Sensitivity & DOT Convention
The standard US Department of Transportation (DOT) reporting convention treats flights arriving at exactly 15 minutes past schedule as delayed ($\ge 15$ minutes). In this pipeline, the target is defined as strictly greater than 15 minutes ($> 15$). In the 2022 validation dataset, exactly **3,137** flights recorded an arrival delay of exactly 15.0 minutes; under this model's formulation, those flights are classified as on-time.

---

## 2. Architectural Challenge & Distillation Engineering Moat

### The 250 MB Serverless Challenge
Deploying production machine learning architectures to serverless hosting environments like Vercel is conventionally blocked by the hard **250 MB unzipped size ceiling**. Standard Python ML dependencies alone exceed this boundary:
- `catboost`: ~329 MB
- `scipy`: ~100 MB
- `scikit-learn`: ~40 MB
- `pandas` & `numpy`: ~70 MB
- Total uncompressed runtime environment: **> 845 MB** (over 3.3x the Vercel ceiling, triggering fatal `LAMBDA_SIZE_EXCEEDED` errors).

### The Solution: Standalone Pure-Python Model + Closed-Form Calibration
Rather than managing costly external virtual machines or cold-boot Docker containers, the Champion v2 pipeline was distilled into an ultra-lean, self-contained inference engine:

1. **Direct Pure-Python CatBoost Decision Tree Compilation**: The 178-tree ensemble was compiled directly into native Python arithmetic (`frontend/api/champion_model.py` - 47.33 MB) with all 12,782 categorical hash combinations mapped via an offline training pool.
2. **Closed-Form Platt Scaling Calibration**: Rather than loading `scipy` or `scikit-learn` calibrators, probability calibration is computed analytically via exact Platt logistic scaling:
   $$P(\text{delay}) = \frac{1}{1 + \exp(-(0.933139 \cdot z + 0.097311))}$$
   where $z = \text{logit}(P_{\text{raw}}) = \text{raw\_margin}$ is the direct margin output of the tree ensemble, evaluated using standard library `math.exp` with zero third-party dependencies.
3. **Precomputed Serialized Asset Lookups**: Airport coordinates, reporting carrier priors, route elapsed times, and holiday dates are compressed into lightweight JSON lookups (`1.09 MB`).
4. **Extreme Payload Reduction**: Slashed the entire runtime bundle from **845 MB** down to **48.4 MB** total (< 20% of Vercel's ceiling, leaving **201.6 MB** of safety headroom).
5. **Exact Bit-Level Parity**: The serverless pure-Python engine achieves exact bit-level parity ($\Delta P \approx 10^{-17}$) against the full CatBoost + Joblib pipeline with zero loss of precision.

---

## 3. Operational Safety, Transparency & Edge-Case Fallbacks

In production, user payloads provide only itinerary schedule parameters (origin, destination, airline carrier, flight date, scheduled departure time, scheduled arrival time). The serving layer features robust operational safety mechanisms:

### 3.1 Temporal Drift Detection
The engine inspects input flight dates against the baseline training distribution (2019–2023). If an itinerary falls outside this window, an automated operational warning is emitted alerting the user that macro airspace congestion profiles and base rates may have drifted.

### 3.2 Cold-Start Route Geodesic Fallback
When a user queries an unseen or rare route not present in the historical training route table (e.g., `ANC_MIA`), the serving engine intercepts the cold-start exception and dynamically computes the great-circle flight distance using the spherical Haversine formula from precomputed airport IATA coordinates:

$$d = 2r \arcsin \left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$

where $r = 3,958.8$ statute miles. Scheduled elapsed duration is derived via physical flight time approximation ($t_{\text{elapsed}} = \frac{d}{7.5} + 30.0$ minutes) and assigned an empirical global prior fallback, preventing runtime crashes while returning a transparent fallback notice.

### 3.3 Transparent Risk Attribution (Directional Log-Odds Drivers)
To eliminate black-box uncertainty without loading heavy runtime explainability libraries, the engine evaluates directional risk drivers contrasting local flight features against the 17.62% historical baseline:

$$\Delta \text{logit} = \ln\left(\frac{p}{1-p}\right) - \ln\left(\frac{p_{\text{base}}}{1-p_{\text{base}}}\right)$$

The top-3 drivers with the largest absolute log-odds impact are returned in human-readable plain language (e.g., origin window congestion, carrier delay rate, or high-volume holiday periods).

---

## 4. Monorepo Architecture

The repository is organized as a decoupled monorepo:

```text
flight-delay-predictor/
├── frontend/                     # Next.js 16 Web Application & Vercel Serverless API
│   ├── api/                      # Vercel Python Serverless Backend (FastAPI)
│   │   ├── champion_model.py     # Distilled zero-dependency CatBoost tree ensemble (47.3 MB)
│   │   ├── predictor.py          # Pure-Python inference pipeline & Platt calibrator
│   │   ├── index.py              # FastAPI ASGI entrypoint (CORS, schemas, endpoints)
│   │   ├── requirements.txt      # Ultra-light runtime dependencies (fastapi + pydantic only)
│   │   └── assets/               # Precomputed lookups (vocab, route priors, coordinates)
│   ├── src/                      # Next.js 16 App Router interface (React, Tailwind CSS, Lucide)
│   │   ├── app/                  # App Router pages and global layouts
│   │   ├── components/           # UI components (SinglePredictor, BatchPredictor, Transparency)
│   │   ├── lib/                  # Universal API client with relative routing
│   │   └── types/                # TypeScript interface contracts
│   └── vercel.json               # Serverless route rewrites and strict security headers
│
├── backend/                      # Offline Research, Training, and Validation Pipeline
│   ├── src/                      # Core ML pipeline (cleaning, splitting, features, baseline)
│   │   └── serving/              # Standard containerized predictor and evaluation tools
│   ├── artifacts/                # Frozen models, pipelines, and lookup tables
│   ├── notebooks/                # Exploratory data analysis and model development
│   ├── reports/                  # Comprehensive audit metrics and verification claims
│   ├── scripts/                  # Verification, asset building, and evaluation scripts
│   ├── tests/                    # Comprehensive test suite (31/31 unit & integration tests)
│   ├── Dockerfile                # Production container specification for alternative deployments
│   ├── docker-compose.yml        # Multi-container local orchestration
│   └── pyproject.toml            # Poetry / Pip build and dependency specifications
│
├── .github/                      # Continuous Integration workflows
├── LICENSE                       # Standard MIT License
└── README.md                     # Monorepo documentation and empirical report
```

---

## 5. Dataset & Zero-Leakage Data Pipeline

The project utilizes the US Bureau of Transportation Statistics (BTS) Airline On-Time Performance records spanning **2019 through August 2023**:
- **Source**: [Kaggle Flight Delay and Cancellation Dataset (2019-2023)](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-data-2019-2023-v2) *(raw data is not redistributed in this repository)*.
- **Completed Flights Analyzed**: 1,643,787 completed domestic flights across 378 commercial airports and 18 airlines.
- **COVID-19 Quarantine**: The anomalous pandemic shock year of **2020 was completely excluded** to avoid learning non-generalizable regime collapse.
- **Strict Temporal Split**:
  - **Training Set (2019, 2021)**: 894,775 flights (Observed base delay rate: **17.62%**). (An earlier draft cited 1,029,918; it could not be reproduced and was removed).
  - **Validation Set (2022)**: 444,554 flights (Observed base delay rate: **20.32%**; seasonal Jan–Aug subset: 296,276 flights, **21.06%** base rate). (An earlier draft cited 438,719 and 21.56%; it could not be reproduced and was removed).
  - **Out-of-Sample Test Set (Jan–Aug 2023)**: 304,458 flights (Observed base delay rate: **22.24%**).

> [!NOTE]
> The train/val/test parquets are not in git; reproduction commands are (in order): `python scripts/01_convert_to_parquet.py`, `python src/cleaning.py`, `python src/split.py`, `python src/features.py` (about 80 s; it rewrites the v1 artifacts, and rebuilt parquet files may differ at the byte level, so compare content/metrics, not file hashes). Times other than the 80 s are approximate.

---

## 6. Machine Learning Approach & Features

The system employs a leakage-free 33-feature pipeline and a 178-tree CatBoost classifier post-hoc calibrated via univariate Platt scaling.

### 6.1 Feature Pipeline (Exact 33 Features)
- **Schedule Coordinates**: `CRS_DEP_TIME`, `CRS_ARR_TIME`, `CRS_ELAPSED_TIME`, `DISTANCE`, `log_distance`, `scheduled_speed`.
- **Cyclical & Temporal Encodings**: `month`, `day_of_week`, `is_weekend`, `is_holiday_period`, `dep_hour`, `arr_hour`, `dep_hour_sin`, `dep_hour_cos`, `arr_hour_sin`, `arr_hour_cos`, `month_sin`, `month_cos`, `day_of_week_sin`, `day_of_week_cos`.
- **Smoothed Historical Priors (Empirical Bayes)**: Additive m-estimate smoothing fitted strictly on train:
  - `prior_airline` (reporting carrier delay rate, $m=50$)
  - `prior_origin` (origin airport delay rate, $m=50$)
  - `prior_dest` (destination airport delay rate, $m=50$)
  - `prior_route` (directional route prior, $m=100$)
  - `prior_origin_hour` (origin airport $\times$ departure hour interaction prior, $m=50$)
- **Operational Volume Frequencies**: `origin_freq`, `dest_freq`, `route_freq`.
- **Native Categoricals (5 Columns)**: `ORIGIN` (378 airports), `DEST` (378 airports), `route` (7,260 routes in training set; 3,942 frequent routes mapped, rare routes mapped to `'OTHER_RARE_ROUTE'`), `AIRLINE_CODE` (18 carriers), `part_of_day` (night, morning, afternoon, evening per `src/features.py`).

### 6.2 Model & Calibrator Parameters
- **Classifier**: CatBoostClassifier (`tree_count_`: 178, `iterations`: 1000 setting with early stopping at best iteration 177, `depth`: 6, `learning_rate`: 0.08, `l2_leaf_reg`: 3, `subsample`: 0.80 MVS, `loss_function`: "Logloss").
- **Calibrator**: Univariate Platt Scaling ($\text{logit}(P_{\text{cal}}) = A \cdot \text{logit}(P_{\text{raw}}) + B$) fitted on 2022 validation log-odds ($A=0.933139, B=0.097311$).

---

## 7. Empirical Evaluation & Results

All test confidence intervals were computed via **250 day-clustered bootstrap resamples** clustered by flight date (`FL_DATE`), preserving systemic day-level weather and network correlations.

### 7.1 Benchmark Metrics on 2023 Out-of-Sample Test Set
| Evaluation Metric | Validation 2022 | Out-of-Sample Test 2023 | 95% Day-Clustered Test CI | Pre-Registered Expectation | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ROC-AUC** | **0.6571** | **0.6523** | $[0.6432, 0.6619]$ | $[0.648, 0.665]$ | **MET** |
| **PR-AUC** | **0.3257** | **0.3497** | $[0.3343, 0.3655]$ | $[0.330, 0.360]$ | **MET** |
| **Brier Score** | **0.1534** | **0.1638** | $[0.1591, 0.1682]$ | $[0.158, 0.166]$ | **MET** |
| **Top 10% Decile Lift** | **1.95x** | **1.93x** | — | $\ge 1.80\text{x}$ | **MET** |
| **Brier Skill Score (vs. Base Rate)** | **+5.25%** | **+5.29%** | $[+4.1\%, +6.5\%]$ | $\ge +4.0\%$ | **MET** |
| **Expected Calibration Error (ECE-10)** | **0.0027** | **0.0104** | — | $\le 0.0350$ | **MET** |

*(Note on validation numbers: an earlier draft cited Validation PR-AUC 0.3392, ROC-AUC 0.6558, and Brier 0.1601; it could not be reproduced and was removed in favor of audited numbers from `reports/phase4_metrics.json`).*

### 7.2 Comparative Baseline Evaluation (2023 Test Set)
Differences against the Champion were tested using paired day-clustered bootstrap testing ($B=250$ resamples).

| Model Architecture | PR-AUC | ROC-AUC | Brier Score | Top 10% Lift | $\Delta$ PR-AUC vs. Champ | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Constant Train Rate)** | 0.2224 | 0.5000 | 0.1751 | 1.00x (not meaningful for constant predictor) | $+0.1273$ | $< 0.0001$ |
| **Logistic Regression (Linear)** | 0.3376 | 0.6455 | 0.1667 | 1.83x | $+0.0121$ | $< 0.0001$ |
| **LightGBM Variant G (v1 Artifact)** | 0.3504 | 0.6534 | 0.1651 | 1.93x | -0.0007 | 0.248 |
| **Champion CatBoost v2 (Frozen)** | **0.3497** | **0.6523** | **0.1638** | **1.93x** | **Reference** | — |

*Statistical Finding*: On the test set CatBoost was statistically equivalent to LightGBM Variant G (delta PR-AUC -0.0007, p=0.248); CatBoost was selected as champion prior to testing and remained frozen.

### 7.3 Monotonic Decile Risk Stratification Table (2023 Test Set)
Ranking the 304,458 test flights into 10 equal deciles demonstrates strict monotonic risk separation with zero inversions:

| Decile Tier | Flight Count | Predicted Prob ($\bar{P}$) | Observed Delay Rate ($\bar{Y}$) | Decile Lift | Calibration Ratio ($\bar{P}/\bar{Y}$) | Assigned Risk Band |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **D1 (Lowest Risk)** | 30,446 | 8.40% | **9.83%** | **0.44x** | 0.855 | Low |
| **D2** | 30,446 | 11.57% | **12.87%** | **0.58x** | 0.899 | Low |
| **D3** | 30,446 | 13.81% | **15.10%** | **0.68x** | 0.914 | Below average |
| **D4** | 30,445 | 15.92% | **16.68%** | **0.75x** | 0.955 | Below average |
| **D5** | 30,446 | 18.11% | **19.25%** | **0.87x** | 0.941 | Average |
| **D6** | 30,446 | 20.45% | **21.76%** | **0.98x** | 0.940 | Average |
| **D7** | 30,445 | 23.19% | **24.77%** | **1.11x** | 0.937 | Above average |
| **D8** | 30,446 | 26.73% | **27.31%** | **1.23x** | 0.979 | Above average |
| **D9** | 30,446 | 31.91% | **32.01%** | **1.44x** | 0.997 | High |
| **D10 (Highest Risk)**| 30,446 | 42.30% | **42.86%** | **1.93x** | 0.987 | High |

*Calibration Behavior*: The model is well calibrated ON AVERAGE; monthly calibration ratios 0.74-1.21; deciles 1-7 underestimate by about 5-15%; probabilities shift with the national delay rate.

---

## 8. Negative Results Registry

In alignment with empirical machine learning standards, every candidate architecture, feature, or tuning experiment tested in this project is explicitly cataloged:

1. **Candidate Features Exploration** (Evaluated in `reports/feature_experiments_metrics.json` via paired bootstrap on validation data):
   - `prior_origin_month`: PR-AUC 0.3220 ($\Delta = -0.00121$, 95% CI $[-0.00235, -0.00032]$, $p = 0.0160$). Result: Dropped (statistically significant degradation).
   - `prior_origin_dow_pod`: PR-AUC 0.3233 ($\Delta = +0.00009$, 95% CI $[-0.00070, +0.00089]$, $p = 0.7840$). Result: Dropped (not significant).
   - `prior_airline_hour`: PR-AUC 0.3236 ($\Delta = +0.00040$, 95% CI $[-0.00022, +0.00101]$, $p = 0.2560$). Result: Dropped (not significant).
   - `prior_airline_origin`: PR-AUC 0.3239 ($\Delta = +0.00066$, 95% CI $[-0.00013, +0.00151]$, $p = 0.1600$). Result: Dropped (not significant).
   - `prior_dest_arrhour`: PR-AUC 0.3235 ($\Delta = +0.00028$, 95% CI $[-0.00030, +0.00090]$, $p = 0.4480$). Result: Dropped (not significant).
   - `typical_dep_density`: PR-AUC 0.3233 ($\Delta = +0.00011$, 95% CI $[-0.00042, +0.00062]$, $p = 0.7680$). Result: Dropped (not significant).
   - `days_to_nearest_holiday`: PR-AUC 0.3217 ($\Delta = -0.00154$, 95% CI $[-0.00293, +0.00013]$, $p = 0.0720$). Result: Dropped (not significant).
2. **Optuna Hyperparameter Tuning on LightGBM** (`reports/lgbm_tuning_summary.json`, `reports/optuna_study.json`): 40 Bayesian optimization trials over depth, leaves, and regularization lifted PR-AUC from 0.3232 to 0.3237 ($\Delta = +0.00051$, 95% CI $[-0.00028, +0.00136]$, $p = 0.24$). Result: Dropped as not statistically significant.
3. **Priors vs. Raw Categoricals (Ablation B vs. C)**: Adding historical priors to calendar and physics features (Variant B PR-AUC 0.3223 vs Variant C PR-AUC 0.3222) did not produce standalone gains without native tree splitting over high-cardinality route categories.
4. **Day of Year and Raw Route Ablation (Variants D, E, F, G)**: Including raw `day_of_year` in tree models caused severe training overfitting (Variant D train PR-AUC 0.3549 vs val 0.3190; Variant E train PR-AUC 0.3577 vs val 0.3177). Removing `day_of_year` and raw route in LightGBM recovered validation PR-AUC to 0.3222 (Variant F) and 0.3232 (Variant G).
5. **Isotonic Regression Calibrator** (`reports/calibration_metrics.json`): While Isotonic regression achieved nominal ECE of 0.00208, it produced non-differentiable step artifacts without improving Brier score (0.15358 vs 0.15356 for Platt). Platt scaling was selected because it preserves strict monotonicity and achieves lowest log loss (0.47983).
6. **CatBoost vs. LightGBM on Test Set** (`reports/test_evaluation_metrics.json`): CatBoost v2 achieved PR-AUC 0.3497 vs LightGBM Variant G 0.3504 ($\Delta = -0.00067$, 95% CI $[-0.00194, +0.00034]$, $p = 0.248$). CatBoost was retained due to superior probability calibration (Brier 0.1638 vs 0.1651, $p < 0.001$).

---

## 9. Serving Approximation Verification

In production, user payloads provide only itinerary schedule parameters (origin, destination, airline carrier, flight date, scheduled departure time, scheduled arrival time). The serving layer automatically derives scheduled elapsed time and distance from `route_lookup.json`.

To quantify the approximation error of using route-median elapsed time and distance, **5,000** random validation rows (seed 42) were evaluated (including 25 flights across 24 unseen routes using geodesic and physical fallback):
- **Mean Absolute Difference**: **0.000446** (0.045 percentage points) in predicted probability.
- **Max Absolute Difference**: **0.011377** (1.14 percentage points).
- **95th Percentile Absolute Difference**: **0.002274** (0.23 percentage points).
- **Spearman Rank Correlation**: **0.999935** (near-perfect preservation of flight ranking).
- **Decile Invariance**: **98.98%** of flights retained their exact risk decile; **100.0%** fell within $\pm 1$ decile.
- **Numerical Pipeline Equivalence**: 200 validation rows scored through the serving predictor vs the batch pipeline yielded a max absolute difference of **0.000000** ($< 10^{-6}$).

---

## 10. Deployment & Quickstart

### 10.1 Primary Deployment: Vercel (1-Click Serverless Architecture)
The complete application (Next.js 16 Client + FastAPI Serverless Inference Engine) is pre-configured for zero-configuration Vercel deployment:

1. Import the repository in [Vercel](https://vercel.com).
2. Set the **Root Directory** to `frontend`.
3. Leave all build and install commands at default (`npm run build` and `npm install`).
4. Click **Deploy**.

Vercel automatically compiles the Next.js frontend with Turbopack, enables edge routing, and registers `frontend/api/index.py` as an ultra-fast Python Serverless Function. The total function payload is **48.4 MB**, well under the 250 MB ceiling with zero external VM dependencies.

### 10.2 Alternative Deployment: Containerized Docker Evaluation
To run the containerized evaluation API and test suite locally:

```bash
cd backend
docker compose build
docker compose up
```

- API Endpoint: `http://localhost:8000/predict`
- Interactive OpenAPI Documentation: `http://localhost:8000/docs`

### 10.3 Local Development Setup

```bash
# 1. Setup backend Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

# 2. Run backend test suite (31/31 unit & integration tests)
pytest backend/tests/ -v

# 3. Launch Next.js interface
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` to interact with the flight delay risk interface.

---

## 11. Limitations & Operational Boundaries

1. **Schedule-Only Horizon**: The model has zero visibility into live weather radar, convective storm fronts, pilot duty timeouts, mechanical failures, or FAA Ground Delay Programs issued hours before flight.
2. **Intrinsic Discrimination Ceiling**: With an ROC-AUC of $\sim 0.6523$, this system is an informational **probabilistic risk stratifier**, not a deterministic certainty classifier. It cannot guarantee whether an individual flight will depart or arrive on time.
3. **Macro Base-Rate Drift**: Aggregate US delay rates drifted from 17.62% in 2019/2021 to 22.24% in 2023. Unadjusted model probabilities reflect historical baseline conditions unless recalibrated against prevailing delay rates.
4. **Intended Use & Prohibited Applications**:
   - Intended solely for informational risk ranking for travelers and itinerary planners to evaluate connection buffers and avoid high-risk itineraries with delays over 15 minutes.
   - Not for tactical flight dispatch, aircraft release, or fuel reserve planning.
   - Not for air traffic separation or safety-critical decisions.
   - Must not be used to deny statutory passenger compensation for realized delays.

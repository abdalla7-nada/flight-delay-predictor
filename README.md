# Flight Delay Risk Predictor

**Author**: Abdallah Nada  
**Release Version**: v1.0 (Production API & Next.js Interface)  
**Task**: Schedule-Only Pre-Departure Flight Delay Risk Stratification  
**Demo Application**: [Live Web Application](https://flight-delay-predictor.vercel.app) *(Vercel)*  

---

## 1. Problem Statement & Motivation

Flight delays generate billions of dollars annually in commercial disruption, missed passenger connections, and operational friction. Most published machine learning models for flight delay rely on tactical, post-departure, or real-time features—such as actual departure delays (`DEP_DELAY`), taxi-out duration, or live weather radar. While those features achieve high nominal accuracy, they are useless at the time of ticket booking (weeks or months in advance), and introducing them into pre-departure models constitutes catastrophic data leakage.

This project addresses the true pre-departure challenge: **Can flight delay vulnerability be meaningfully separated at booking time using only schedule coordinates?**

The target is defined as:

$$\text{DELAY} = \mathbf{1}(\mathrm{ARR\_DELAY} > 15 \text{ minutes})$$

### Threshold Sensitivity & DOT Convention
The standard US Department of Transportation (DOT) reporting convention treats flights arriving at exactly 15 minutes past schedule as delayed ($\ge 15$ minutes). In this pipeline, the target is defined as strictly greater than 15 minutes ($> 15$). In the 2022 validation dataset, exactly **3,137 flights** recorded an arrival delay of exactly 15.0 minutes; under this model's formulation, those flights are classified as on-time.

---

## 2. Dataset & Zero-Leakage Data Pipeline

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

## 3. Machine Learning Approach & Architecture

The system employs a leakage-free 33-feature pipeline and a 178-tree CatBoost classifier post-hoc calibrated via univariate Platt scaling.

### 3.1 Feature Pipeline (Exact 33 Features)
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

### 3.2 Model & Calibrator
- **Classifier**: CatBoostClassifier (`tree_count_`: 178, `iterations`: 1000 setting with early stopping at best iteration 177, `depth`: 6, `learning_rate`: 0.08, `l2_leaf_reg`: 3, `subsample`: 0.80 MVS, `loss_function`: "Logloss").
- **Calibrator**: Univariate Platt Scaling ($\text{logit}(P_{\text{cal}}) = A \cdot \text{logit}(P_{\text{raw}}) + B$) fitted on 2022 validation log-odds ($A=0.933139, B=0.097311$).

---

## 4. Empirical Evaluation & Results

All test confidence intervals were computed via **250 day-clustered bootstrap resamples** clustered by flight date (`FL_DATE`), preserving systemic day-level weather and network correlations.

### 4.1 Benchmark Metrics on 2023 Out-of-Sample Test Set
| Evaluation Metric | Validation 2022 | Out-of-Sample Test 2023 | 95% Day-Clustered Test CI | Pre-Registered Expectation | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ROC-AUC** | **0.6571** | **0.6523** | $[0.6432, 0.6619]$ | $[0.648, 0.665]$ | **MET** |
| **PR-AUC** | **0.3257** | **0.3497** | $[0.3343, 0.3655]$ | $[0.330, 0.360]$ | **MET** |
| **Brier Score** | **0.1534** | **0.1638** | $[0.1591, 0.1682]$ | $[0.158, 0.166]$ | **MET** |
| **Top 10% Decile Lift** | **1.95x** | **1.93x** | — | $\ge 1.80\text{x}$ | **MET** |
| **Brier Skill Score (vs. Base Rate)** | **+5.25%** | **+5.29%** | $[+4.1\%, +6.5\%]$ | $\ge +4.0\%$ | **MET** |
| **Expected Calibration Error (ECE-10)** | **0.0027** | **0.0104** | — | $\le 0.0350$ | **MET** |

*(Note on validation numbers: an earlier draft cited Validation PR-AUC 0.3392, ROC-AUC 0.6558, and Brier 0.1601; it could not be reproduced and was removed in favor of audited numbers from `reports/phase4_metrics.json`).*

### 4.2 Comparative Baseline Evaluation (2023 Test Set)
Differences against the Champion were tested using paired day-clustered bootstrap testing ($B=250$ resamples).

| Model Architecture | PR-AUC | ROC-AUC | Brier Score | Top 10% Lift | $\Delta$ PR-AUC vs. Champ | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Constant Train Rate)** | 0.2224 | 0.5000 | 0.1751 | 1.00x (not meaningful for constant predictor) | $+0.1273$ | $< 0.0001$ |
| **Logistic Regression (Linear)** | 0.3376 | 0.6455 | 0.1667 | 1.83x | $+0.0121$ | $< 0.0001$ |
| **LightGBM Variant G (v1 Artifact)** | 0.3504 | 0.6534 | 0.1651 | 1.93x | $-0.0007$ | $0.2480$ |
| **Champion CatBoost v2 (Frozen)** | **0.3497** | **0.6523** | **0.1638** | **1.93x** | **Reference** | — |

*Statistical Finding*: On the test set CatBoost was statistically equivalent to LightGBM Variant G (delta PR-AUC -0.0007, p=0.248); CatBoost was selected as champion prior to testing and remained frozen.

### 4.3 Monotonic Decile Risk Stratification Table (2023 Test Set)
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

## 5. Negative Results Registry

In alignment with empirical machine learning standards, every candidate architecture, feature, or tuning experiment tested in this project is explicitly cataloged:

1. **Candidate Features Exploration** (Evaluated in `reports/feature_experiments_metrics.json` via paired bootstrap on validation data):
   - `prior_origin_month`: PR-AUC 0.3220 ($\Delta = -0.00121$, 95% CI $[-0.00235, -0.00032]$, $p = 0.0160$). Result: Dropped (statistically significant degradation).
   - `prior_origin_dow_pod`: PR-AUC 0.3233 ($\Delta = +0.00009$, 95% CI $[-0.00070, +0.00089]$, $p = 0.7840$). Result: Dropped (not significant).
   - `prior_airline_hour`: PR-AUC 0.3236 ($\Delta = +0.00040$, 95% CI $[-0.00022, +0.00101]$, $p = 0.2560$). Result: Dropped (not significant).
   - `prior_airline_origin`: PR-AUC 0.3239 ($\Delta = +0.00066$, 95% CI $[-0.00013, +0.00151]$, $p = 0.1600$). Result: Dropped (not significant).
   - `prior_dest_arrhour`: PR-AUC 0.3235 ($\Delta = +0.00028$, 95% CI $[-0.00030, +0.00090]$, $p = 0.4480$). Result: Dropped (not significant).
   - `typical_dep_density`: PR-AUC 0.3233 ($\Delta = +0.00011$, 95% CI $[-0.00042, +0.00062]$, $p = 0.7680$). Result: Dropped (not significant).
   - `days_to_nearest_holiday`: PR-AUC 0.3217 ($\Delta = -0.00154$, 95% CI $[-0.00293, +0.00013]$, $p = 0.0720$). Result: Dropped (not significant).
2. **Optuna Hyperparameter Tuning on LightGBM** (`reports/lgbm_tuning_summary.json`, `reports/optuna_study.json`): 40 Bayesian optimization trials (an earlier draft cited 100; it could not be reproduced and was removed) over depth, leaves, and regularization lifted PR-AUC from 0.3232 to 0.3237 ($\Delta = +0.00051$, 95% CI $[-0.00028, +0.00136]$, $p = 0.24$). Result: Dropped as not statistically significant.
3. **Priors vs. Raw Categoricals (Ablation B vs. C)**: Adding historical priors to calendar and physics features (Variant B PR-AUC 0.3223 vs Variant C PR-AUC 0.3222) did not produce standalone gains without native tree splitting over high-cardinality route categories.
4. **Day of Year and Raw Route Ablation (Variants D, E, F, G)**: Including raw `day_of_year` in tree models caused severe training overfitting (Variant D train PR-AUC 0.3549 vs val 0.3190; Variant E train PR-AUC 0.3577 vs val 0.3177). Removing `day_of_year` and raw route in LightGBM recovered validation PR-AUC to 0.3222 (Variant F) and 0.3232 (Variant G).
5. **Isotonic Regression Calibrator** (`reports/calibration_metrics.json`): While Isotonic regression achieved nominal ECE of 0.00208, it produced non-differentiable step artifacts without improving Brier score (0.15358 vs 0.15356 for Platt). Platt scaling was selected because it preserves strict monotonicity and achieves lowest log loss (0.47983).
6. **CatBoost vs. LightGBM on Test Set** (`reports/test_evaluation_metrics.json`): CatBoost v2 achieved PR-AUC 0.3497 vs LightGBM Variant G 0.3504 ($\Delta = -0.00067$, 95% CI $[-0.00194, +0.00034]$, $p = 0.248$). CatBoost was retained due to superior probability calibration (Brier 0.1638 vs 0.1651, $p < 0.001$).

---

## 6. Serving Approximation Verification

In production, user payloads provide only itinerary schedule parameters (origin, destination, airline carrier, flight date, scheduled departure time, scheduled arrival time). The serving layer automatically derives scheduled elapsed time and distance from `route_lookup.json`.

To quantify the approximation error of using route-median elapsed time and distance, **5,000 random validation rows (seed 42)** were evaluated (including 25 flights across 24 unseen routes using geodesic and physical fallback):
- **Mean Absolute Difference**: **0.000446 (0.045 percentage points)** in predicted probability.
- **Max Absolute Difference**: **0.011377 (1.14 percentage points)**.
- **95th Percentile Absolute Difference**: **0.002274 (0.23 percentage points)**.
- **Spearman Rank Correlation**: **0.999935** (near-perfect preservation of flight ranking).
- **Decile Invariance**: **98.98%** of flights retained their exact risk decile; **100.0%** fell within $\pm 1$ decile.
- **Numerical Pipeline Equivalence**: 200 validation rows scored through the serving predictor vs the batch pipeline yielded a max absolute difference of **0.000000** ($< 10^{-6}$).

---

## 7. Limitations & Operational Boundaries

1. **Schedule-Only Horizon**: The model has zero visibility into live weather radar, convective storm fronts, pilot duty timeouts, mechanical failures, or FAA Ground Delay Programs issued hours before flight.
2. **Intrinsic Discrimination Ceiling**: With an ROC-AUC of $\sim 0.6523$, this system is an informational **probabilistic risk stratifier**, not a deterministic certainty classifier. It cannot guarantee whether an individual flight will depart or arrive on time.
3. **Macro Base-Rate Drift**: Aggregate US delay rates drifted from 17.62% in 2019/2021 to 22.24% in 2023. Unadjusted model probabilities reflect historical baseline conditions unless recalibrated against prevailing delay rates.
4. **Intended Use & Prohibited Applications**:
   - Intended solely for informational risk ranking for travelers and itinerary planners to evaluate connection buffers and avoid high-risk itineraries with delays over 15 minutes.
   - Not for tactical flight dispatch, aircraft release, or fuel reserve planning.
   - Not for air traffic separation or safety-critical decisions.
   - Must not be used to deny statutory passenger compensation for realized delays.

---

## 8. Quickstart & Deployment Instructions

### 8.1 Local Installation
```bash
# Clone and setup environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install deployment dependencies
pip install -r requirements-deploy.txt
```

### 8.2 Run the FastAPI REST Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Interactive OpenAPI documentation will be available at `http://localhost:8000/docs`.

**Example Request**:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "airline_code": "AA",
       "origin": "DFW",
       "dest": "ORD",
       "flight_date": "2023-05-15",
       "sched_dep_time": "08:30",
       "sched_arr_time": "11:00"
     }'
```

### 8.3 Run the Modern Next.js Frontend
```bash
cd frontend
npm install
npm run dev
```
The modern dashboard will be available at `http://localhost:3000`.

### 8.4 Docker & Docker Compose
To run the production API container via Docker:
```bash
docker compose build
docker compose up
```
API endpoint: `http://localhost:8000/predict`  
Interactive Swagger Docs: `http://localhost:8000/docs`


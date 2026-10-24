# Model Card: Flight Delay Risk Predictor (Champion v2)

**Author**: Abdallah Nada  
**Model Architecture**: CatBoost Classifier with Native Categorical Feature Handling + Post-Hoc Platt Scaling Calibrator  
**Model Version**: v2.0 (Champion)  
**Release Date**: October 2026 (Evaluated on 2023 US Domestic Flight Test Data)  
**Artifact Dependencies**:  
- Classifier: [`artifacts/models/champion_v2.joblib`](artifacts/models/champion_v2.joblib)
- Calibrator: [`artifacts/models/calibrator_v2.joblib`](artifacts/models/calibrator_v2.joblib)
- Feature Pipeline: [`artifacts/feature_pipeline_v2.joblib`](artifacts/feature_pipeline_v2.joblib)
- Metadata: [`artifacts/models/model_metadata_v2.json`](artifacts/models/model_metadata_v2.json)

---

## 1. Model Overview & Intended Use

### 1.1 Summary
The **Flight Delay Risk Predictor (Champion v2)** is a machine learning model designed to estimate the probability that a commercial domestic US passenger flight will arrive **more than 15 minutes past its scheduled arrival time** ($\text{ARR\_DELAY} > 15$ minutes).

The system operates under a strict **schedule-only constraint**: every feature is computable at the time of ticket booking, months or days in advance, relying solely on scheduled timetable coordinates (origin, destination, airline carrier, scheduled departure time, scheduled arrival time, and date). The model employs 33 features processed through a leakage-free feature pipeline, modeled via a 178-tree CatBoost classifier, and calibrated via Platt logistic scaling.

### 1.2 Target Definition & Threshold Convention
- **Target Formulation**:
  $$\text{DELAY} = \mathbf{1}(\text{ARR\_DELAY} > 15)$$
- **Threshold Sensitivity**: The standard US Department of Transportation (DOT) reporting convention classifies flights arriving at exactly 15 minutes past schedule as delayed ($\ge 15$ minutes). In this pipeline, the target is defined as strictly greater than 15 minutes ($> 15$). In the 2022 validation dataset, exactly **3,137 flights** recorded an arrival delay of exactly 15.0 minutes; under this model's formulation, those flights are classified as on-time.

### 1.3 Intended Use Cases
- **Passenger Booking Decision Support**: Comparing the structural delay vulnerability of competing flight options at booking time.
- **Travel Schedule Planning**: Informational risk ranking for travelers and itinerary planners to evaluate connection buffers based on historical schedule performance and avoid high-risk itineraries with delays over 15 minutes.
- **Airline Timetable Robustness Auditing**: Analyzing schedule vulnerability across bank hours, airport slot congestion, and route profiles.

### 1.4 Out-of-Scope & Prohibited Uses
- **Operational Flight Dispatch**: This model must **NOT** be used for tactical aircraft dispatch, flight release, or mandatory fuel reserve calculation.
- **Safety-Critical Operations**: Under no circumstances should this model inform air traffic control separation, safety abort decisions, or emergency procedures.
- **Passenger Compensation Denial**: Airlines must not cite schedule-level risk predictions to deny legally mandated passenger compensation for realized delays.
- **Real-Time Ground Delay Execution**: The model is not designed to replace real-time FAA Ground Delay Programs (GDP) or tactical slot allocations.

---

## 2. Training, Validation, and Test Datasets

The model was developed, tuned, and evaluated on a multi-year temporal split of US Bureau of Transportation Statistics (BTS) Airline On-Time Performance data.

> [!NOTE]
> The train/val/test parquets are not in git; reproduction commands are (in order): `python scripts/01_convert_to_parquet.py`, `python src/cleaning.py`, `python src/split.py`, `python src/features.py` (about 80 s; it rewrites the v1 artifacts, and rebuilt parquet files may differ at the byte level, so compare content/metrics, not file hashes). Times other than the 80 s are approximate.

| Dataset Split | Temporal Range | Flight Count | Calendar Days | Observed Base Delay Rate | Role in Lifecycle |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Training Set** | Jan 1, 2019 – Dec 31, 2019<br>Jan 1, 2021 – Dec 31, 2021 | **894,775** | 730 | **17.62%** | Feature engineering, target encoding priors, and CatBoost tree construction. (An earlier draft cited 1,029,918; it could not be reproduced and was removed). |
| **Validation Set** | Jan 1, 2022 – Dec 31, 2022 | **444,554** | 365 | **20.32%** | Model selection, hyperparameter tuning, and Platt calibrator fitting. (An earlier draft cited 438,719 and 21.56%; it could not be reproduced and was removed). |
| *(Val Seasonal Matched)* | Jan 1, 2022 – Aug 31, 2022 | 296,276 | 243 | **21.06%** | Pre-registration baseline matching test calendar months. |
| **Test Set** | Jan 1, 2023 – Aug 31, 2023 | **304,458** | 243 | **22.24%** | Final one-time empirical evaluation. Strictly unblinded once in Phase 5. |
| **Total Corpus** | Jan 1, 2019 – Aug 31, 2023 | **1,643,787** | 1,338 | **19.21%** | Total completed non-cancelled flights analyzed across all 3 splits. |

### Sampling & Exclusion Rules:
- **COVID-19 Exclusion**: The year 2020 was completely excluded from training, validation, and testing because pandemic flight cancellations, government lockdowns, and anomalous network collapse represent non-generalizable regime shifts.
- **Domestic Commercial Flights**: Only scheduled passenger flights operated by reporting carriers to the US Department of Transportation are included.
- **Cancelled and Diverted Flights**: In alignment with FAA/BTS convention, cancelled and diverted flights are excluded prior to delay modeling.

---

## 3. Feature Pipeline & Model Architecture

### 3.1 Input Features (Exact 33 Features)
All features comply with the strict schedule-only inference contract. As defined in `model_metadata_v2.json` and `artifacts/feature_pipeline_v2.joblib`, the model uses exactly 33 features:

1. **Calendar & Date Signals**: `month`, `day_of_week`, `is_weekend`, `is_holiday_period`.
2. **Scheduled Clock Times**: `dep_hour`, `arr_hour`, `CRS_DEP_TIME`, `CRS_ARR_TIME`.
3. **Cyclical Transforms**: `dep_hour_sin`, `dep_hour_cos`, `arr_hour_sin`, `arr_hour_cos`, `month_sin`, `month_cos`, `day_of_week_sin`, `day_of_week_cos`.
4. **Physical Schedule Coordinates**: `CRS_ELAPSED_TIME`, `DISTANCE`, `log_distance`, `scheduled_speed`.
5. **Historical Target Encodings (Out-of-Fold Empirical Bayes Priors)**:
   - `prior_airline` (smoothed reporting carrier delay rate, $m=50$)
   - `prior_origin` (marginal origin airport delay rate, $m=50$)
   - `prior_dest` (marginal destination airport delay rate, $m=50$)
   - `prior_route` (directional origin-destination route prior, $m=100$)
   - `prior_origin_hour` (origin airport $\times$ departure hour interaction prior, $m=50$)
6. **Operational Volume Frequencies**: `origin_freq`, `dest_freq`, `route_freq`.
7. **Native Categorical Features** (5 columns passed to CatBoost):
   - `part_of_day` (night: 00:00–05:59, morning: 06:00–11:59, afternoon: 12:00–17:59, evening: 18:00–23:59)
   - `AIRLINE_CODE` (18 reporting carriers)
   - `ORIGIN` (378 commercial airports)
   - `DEST` (378 commercial airports)
   - `route` (7,260 routes in training set; 3,942 frequent directional routes tracked, rare routes mapped to `'OTHER_RARE_ROUTE'`)

### 3.2 Hyperparameters & Calibration
Extracted directly from `artifacts/models/champion_v2.joblib` and `artifacts/models/calibrator_v2.joblib`:
- **Classifier**: CatBoostClassifier
  - `iterations`: 1000 setting (`early_stopping_rounds`: 40, `best_iteration`: 177, `tree_count_`: 178)
  - `depth`: 6
  - `learning_rate`: 0.08
  - `l2_leaf_reg`: 3
  - `subsample`: 0.80 (MVS bootstrap)
  - `loss_function`: "Logloss"
  - `eval_metric`: "PRAUC"
- **Calibrator**: Univariate Platt Scaling logistic mapping:
  $$P_{\text{cal}} = \sigma(A \cdot \text{logit}(P_{\text{raw}}) + B)$$
  fitted on 2022 validation log-odds ($A=0.933139, B=0.097311$).

---

## 4. Empirical Evaluation & Performance Metrics

All test confidence intervals are estimated via **250 day-clustered bootstrap resamples** clustered by flight date (`FL_DATE`) to preserve real-world daily weather and airspace correlations.

### 4.1 Benchmark Metrics: Validation vs. Test

| Metric | Full Validation (2022) | Seasonal Validation (Jan–Aug 2022) | **Final Test Set (Jan–Aug 2023)** | **95% Day-Clustered Test CI** |
| :--- | :---: | :---: | :---: | :---: |
| **Sample Size ($N$)** | **444,554** | 296,276 | **304,458** | 243 temporal clusters |
| **Observed Delay Prevalence** | **20.32%** | 21.06% | **22.24%** | $[21.1\%, 23.4\%]$ |
| **PR-AUC (Average Precision)** | **0.3257** | 0.3355 | **0.3497** | $[0.3343, 0.3655]$ |
| **ROC-AUC** | **0.6571** | 0.6576 | **0.6523** | $[0.6432, 0.6619]$ |
| **Brier Score** | **0.1534** | 0.1572 | **0.1638** | $[0.1591, 0.1682]$ |
| **Binary Log Loss** | **0.4793** | 0.4876 | **0.5045** | — |
| **Brier Skill Score (vs. Base Rate)** | **+5.25%** | +5.44% | **+5.29%** | $[+4.1\%, +6.5\%]$ |
| **Expected Calibration Error (ECE-10)**| **0.0027** | 0.0031 | **0.0104** | — |
| **Mean Predicted Probability** | 0.2032 | 0.2109 | **0.2124** | Bias: $-0.0100$ ($-1.0$ pp) |
| **Top 10% Decile Precision** | 39.72% | 40.59% | **42.86%** | $[40.9\%, 44.8\%]$ |
| **Top 10% Decile Lift** | 1.95x | 1.93x | **1.93x** | Invariant across years |

*(Note on historical validation figures: an earlier draft cited Validation PR-AUC 0.3392, ROC-AUC 0.6558, and Brier 0.1601; it could not be reproduced and was removed in favor of the audited metrics from `reports/phase4_metrics.json`).*

---

### 4.2 Comparative Baselines (Evaluated on Same 304,458 Test Flights)

Differences against the Champion were evaluated via paired day-clustered bootstrap testing ($B=250$ resamples) on the 2023 test set:

| Architecture | PR-AUC | ROC-AUC | Brier Score | Top 10% Lift | $\Delta$ PR-AUC vs. Champion | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Constant Train Rate)** | 0.2224 | 0.5000 | 0.1751 | 1.00x (not meaningful for constant predictor) | $+0.1273$ | $< 0.0001$ |
| **Logistic Regression (Phase 3)** | 0.3376 | 0.6455 | 0.1667 | 1.83x | $+0.0121$ | $< 0.0001$ |
| **LightGBM Variant G (v1 Artifact)** | 0.3504 | 0.6534 | 0.1651 | 1.93x | $-0.0007$ | $0.2480$ (Not Sig) |
| **Champion v2 Raw (Uncalibrated)** | 0.3497 | 0.6523 | 0.1652 | 1.93x | $0.0000$ | $1.0000$ |
| **Champion v2 (Calibrated Platt)** | **0.3497** | **0.6523** | **0.1638** | **1.93x** | **Reference** | — |

---

### 4.3 Risk Decile Monotonicity (2023 Test Set)

| Decile | Flight Count | Mean Pred Prob ($\bar{P}$) | Observed Delay Rate ($\bar{Y}$) | Decile Lift | Calibration Ratio ($\bar{P}/\bar{Y}$) | Assigned Risk Band |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **D1** (Lowest Risk) | 30,446 | 0.0840 | 0.0983 (9.83%) | **0.44x** | 0.855 | Low |
| **D2** | 30,446 | 0.1157 | 0.1287 (12.87%) | **0.58x** | 0.899 | Low |
| **D3** | 30,446 | 0.1381 | 0.1510 (15.10%) | **0.68x** | 0.914 | Below average |
| **D4** | 30,445 | 0.1592 | 0.1668 (16.68%) | **0.75x** | 0.955 | Below average |
| **D5** | 30,446 | 0.1811 | 0.1925 (19.25%) | **0.87x** | 0.941 | Average |
| **D6** | 30,446 | 0.2045 | 0.2176 (21.76%) | **0.98x** | 0.940 | Average |
| **D7** | 30,445 | 0.2319 | 0.2477 (24.77%) | **1.11x** | 0.937 | Above average |
| **D8** | 30,446 | 0.2673 | 0.2731 (27.31%) | **1.23x** | 0.979 | Above average |
| **D9** | 30,446 | 0.3191 | 0.3201 (32.01%) | **1.44x** | 0.997 | High |
| **D10** (Highest Risk) | 30,446 | 0.4230 | 0.4286 (42.86%) | **1.93x** | 0.987 | High |

---

### 4.4 Serving Input Approximation Error (Validation Set Audit)

In deployment, flight payloads provide schedule coordinates without elapsed duration or distance, which are derived from `route_lookup.json`. To quantify the impact of this operational approximation, **5,000 random validation flights (seed 42)** were scored comparing true schedule values against route-lookup values (including 25 flights across 24 unseen routes handled via geodesic fallback):
- **Mean Absolute Difference**: **0.000446 (0.045 percentage points)** in predicted probability.
- **Max Absolute Difference**: **0.011377 (1.14 percentage points)**.
- **95th Percentile Absolute Difference**: **0.002274 (0.23 percentage points)**.
- **Spearman Rank Correlation**: **0.999935** (near-perfect preservation of flight risk ranking).
- **Decile Invariance**: **98.98%** of flights retained their exact risk decile; **100.0%** fell within $\pm 1$ decile.
- **Numerical Pipeline Equivalence**: 200 validation rows scored through the serving predictor vs the batch pipeline yielded a max absolute difference of **0.000000** ($< 10^{-6}$).

---

## 5. Subgroup Slices & Operational Stability

The model was audited across major airlines, top airports, and departure time windows on the 2023 test set (subgroups with at least 3,000 flights):

- **Carrier Variance** (Table 5.A in Phase 5 Report):
  - Highest discrimination: Southwest Airlines (`WN`, $N=63,519$, $\text{ROC-AUC} = 0.7003$, $\text{Lift} = 2.08\text{x}$).
  - Lowest discrimination: Regional partners (SkyWest `OO` $\text{ROC}=0.5940$, Envoy `MQ` $\text{ROC}=0.5763$).
  - *Operational Hypothesis*: Point-to-point carriers accumulate schedule delays monotonically across the day; regional carriers suffer unpredictable mainline ground stops and slot rationing that schedule coordinates cannot capture.
- **Airport Variance** (Table 5.B in Phase 5 Report):
  - Highest lift: New York LaGuardia (`LGA`, $\text{Lift} = 2.16\text{x}$) and Las Vegas (`LAS`, $\text{ROC} = 0.6793$).
  - Weather-volatile hubs: Chicago O'Hare (`ORD`, $\text{ROC} = 0.5963$, $\text{Lift} = 1.50\text{x}$).
  - *Operational Hypothesis*: Constrained slot airports exhibit strong structural delay patterns; midwestern hubs are dominated by acute, unforecasted weather disruptions.
- **Time of Day Variance** (Table 5.C in Phase 5 Report):
  - Delay rates increase from **10.05% at Night** and **14.80% in Morning** to **25.53% in Afternoon** and **31.29% in Evening**.
  - Top 10% lift is highest at Night ($2.21\text{x}$) and Morning ($1.55\text{x}$), demonstrating strong ability to isolate late-operating or turnaround-delayed flights.

---

## 6. Model Limitations & Caveats

1. **Schedule-Only Information Horizon**: The model has zero access to actual day-of-departure conditions (real-time METAR/TAF weather forecasts, pilot duty-time timeouts, mechanical faults, or FAA Ground Delay Programs).
2. **Intrinsic Discrimination Ceiling**: With an ROC-AUC of $\sim 0.6523$, the model is an effective **probabilistic risk stratifier**, not a deterministic binary predictor. It cannot definitively state whether a specific flight will arrive on time.
3. **Macro Base-Rate Drift**: Base delay prevalence rose from 17.62% in 2019/2021 to 22.24% in 2023. This causes a residual probability underestimation of approximately $1.0$ percentage point unless recalibrated to prevailing operational conditions.
4. **No Inbound Aircraft Tracking (Tail Chains)**: Because aircraft tail numbers were not linked to preceding flight legs, upstream arrival delays cannot be explicitly traced to downstream departures.
5. **Geographic Coverage**: US domestic flights only. International flights, general aviation, military flights, and non-reporting regional carriers are excluded.
6. **Subsample Scope**: The dataset represents 1,643,787 non-cancelled domestic commercial flights across the four years (2019, 2021, 2022, Jan–Aug 2023).

---

## 7. Ethical, Fairness, and Responsible AI Considerations

- **Non-Discrimination**: Carrier and airport variations reflect genuine physical infrastructure constraints, airspace density, and operating models. They must not be misconstrued as inherent airline malice or used to unfairly penalize budget carriers in consumer search rankings without clear context.
- **Consumer Protection**: Probabilities should empower travelers with transparent expectations, not be used to justify algorithmic price increases on flights deemed "low-risk".
- **Accountability**: Realized flight delays remain the legal responsibility of airlines and air navigation service providers. Predictions generated by this system carry no legal standing in regulatory compliance or dispute arbitration.

---

## 8. Maintenance & Monitoring Guidelines

- **Recalibration Frequency**: Platt scaling intercept should be evaluated quarterly against recent 60-day delay rates to eliminate base-rate drift bias.
- **Retraining Schedule**: Feature lookups and CatBoost tree weights should be retrained annually, incorporating the preceding calendar year's completed flight records.
- **Data Drift Alarms**: Alert triggers should be configured if:
  - Nationwide monthly base delay rate drifts outside $[15\%, 30\%]$.
  - Route vocabulary miss rate exceeds $2.0\%$ (indicating major airline network restructuring).
  - Out-of-sample Brier Skill Score drops below $+3.0\%$.

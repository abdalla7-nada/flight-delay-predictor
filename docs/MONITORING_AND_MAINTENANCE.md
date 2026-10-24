# Production Monitoring and Maintenance Protocol

**System**: Flight Delay Risk Predictor (Champion v2)  
**Author**: Abdallah Nada  
**Status**: Active Production Standard  
**Last Updated**: October 2026  

---

## 1. Overview & Purpose

The Flight Delay Risk Predictor operates under a strict schedule-only pre-departure contract. Because airline networks, slot controls, and seasonal weather patterns evolve, maintaining long-term probabilistic calibration and ranking skill requires structured telemetry, drift detection, automated alerts, and scheduled model refreshes.

This document outlines the monitoring metrics, drift detection protocols (PSI), alerting thresholds, intervention triggers, and the retraining procedure for incorporating new operational flight data.

---

## 2. Key Monitoring Metrics & Observability

Production monitoring operates on two distinct cadences:
1. **Real-Time / Daily Inference Ingestion (Labels Absent)**: Evaluates incoming requests for distribution shifts, missing routes, and input anomalies.
2. **Monthly Retrospective Audit (Labels Realized)**: Evaluates performance once Bureau of Transportation Statistics (BTS) or airline ground truth delay records become available (~45 days post-flight).

### 2.1 Daily & Weekly Metrics (Pre-Label)
- **Daily Request Volume**: Number of inference calls per carrier, origin airport, and destination airport.
- **Mean Daily Predicted Probability**: Tracks whether model predictions are drifting due to shifting time-of-day or seasonal query mixes.
- **Route Miss Rate**: Proportion of inference requests where the route is not present in `route_lookup.json` (triggering geodesic distance fallback).
- **Latency & Error Rates**: P50, P95, and P99 inference latencies (target P95 < 100 ms) and HTTP 422/500 error rates.

### 2.2 Monthly Retrospective Metrics (Post-Label)
- **Observed Delay Rate ($\bar{Y}$)**: Percentage of flights arriving $\ge 15$ minutes late.
- **Mean Predicted Probability ($\bar{P}$)**: Expected delay probability across the same cohort.
- **Calibration Ratio**: Defined as $\bar{P} / \bar{Y}$.
- **Discrimination (ROC-AUC & PR-AUC)**: Evaluated monthly to ensure ranking power is preserved.
- **Brier Score & Brier Skill Score (BSS)**: Mean squared error of probabilities compared against the baseline climatology rate.
- **Top 10% Decile Lift**: Delay prevalence in the highest risk decile relative to the monthly base rate (benchmark: $\ge 1.80\text{x}$).

---

## 3. Input Feature Drift & Population Stability Index (PSI)

Input drift is tracked using the **Population Stability Index (PSI)**, comparing current monthly inference inputs ($P$) against the baseline training distribution ($B$):

$$\text{PSI} = \sum_{k=1}^{K} (P_k - B_k) \cdot \ln\left(\frac{P_k}{B_k}\right)$$

### 3.1 PSI Bins and Feature Coverage
- **Continuous Features (10 Quantile Bins)**: `CRS_DEP_TIME`, `CRS_ARR_TIME`, `DISTANCE`, `CRS_ELAPSED_TIME`, `scheduled_speed`, `prior_route`, `prior_origin`.
- **Categorical Features**: `AIRLINE_CODE` (18 bins), top-20 `ORIGIN` airports, top-20 `DEST` airports, `part_of_day` (4 bins).

### 3.2 PSI Operational Thresholds
| PSI Value | Assessment | Required Action |
| :--- | :--- | :--- |
| $\text{PSI} < 0.10$ | **Stable** | No action required; normal production operation. |
| $0.10 \le \text{PSI} < 0.25$ | **Moderate Drift** | Heightened surveillance; inspect carrier volume or seasonal shift in requests. |
| $\text{PSI} \ge 0.25$ | **Significant Drift** | Root-cause analysis; trigger feature priority audit and potential model recalibration. |

---

## 4. Operational Alert Triggers & Remediation Workflows

| Condition / Trigger | Threshold / Duration | Severity | Root Cause | Remediation Action |
| :--- | :--- | :---: | :--- | :--- |
| **Calibration Drift** | Calibration ratio $\bar{P}/\bar{Y} < 0.90$ or $> 1.10$ for **2 consecutive months** | High | Macroeconomic travel surge, air traffic control congestion, or seasonal baseline shift. | **Recalibrate Intercept**: Refit Platt scaling intercept $B$ on recent 60-day flight window (fast, zero tree retraining). |
| **Discrimination Collapse** | Monthly ROC-AUC $< 0.62$ for **2 consecutive months** | Critical | Structural change in flight scheduling, new hub operations, or widespread timetable revisions. | **Full Model Retraining**: Retrain CatBoost tree ensemble incorporating latest annual data. |
| **Unseen Route Spike** | Route lookup miss rate $> 2.0\%$ over a **7-day rolling window** | Medium | Airline carrier route restructuring or new regional network additions. | **Update Route Lookup Table**: Recompute `route_lookup.json` medians and add new frequent routes. |
| **P95 Latency Spike** | Single prediction latency $> 100$ ms for $> 1.0\%$ of requests | Medium | Server resource contention, cache misses, or excessive SHAP computation overhead. | Scale container replicas or disable live SHAP computation on high-volume traffic. |

---

## 5. Recalibration vs. Retraining Protocols

### 5.1 Protocol A: Intercept Recalibration (Lightweight, ~5 minutes)
When discrimination remains stable ($\text{ROC-AUC} \ge 0.64$) but base delay rates shift, full tree retraining is unnecessary. Only the Platt logistic calibrator intercept requires adaptation:
1. Extract the most recent 60 days of completed flights ($N \approx 50,000$ rows).
2. Compute raw uncalibrated probabilities $P_{\text{raw}}$ from the frozen CatBoost model.
3. Fit a 1-parameter logistic shift:
   $$\text{logit}(P_{\text{cal}}) = \text{logit}(P_{\text{raw}}) + \Delta_B$$
   where $\Delta_B = \text{logit}(\bar{Y}_{\text{recent}}) - \text{logit}(\bar{P}_{\text{raw}})$.
4. Validate that monthly calibration ratio returns to $[0.95, 1.05]$.
5. Update `artifacts/models/calibrator_v2.joblib` and increment patch version (`v2.0.1`).

### 5.2 Protocol B: Annual Retraining Plan (Full Pipeline)
A full retraining cycle should be conducted annually or when discrimination falls below $0.62$:
1. **Dataset Extension**: Incorporate the most recent complete calendar year (e.g., add 2023 completed flights to the 2019/2021 training corpus).
2. **Leakage-Free Feature Re-Fit**:
   - Recompute global mean, frequent routes ($\ge 50$ flights), and route elapsed/distance medians.
   - Recompute historical smoothed priors using 5-fold out-of-fold (OOF) cross-validation on the expanded training set.
3. **CatBoost Training**:
   - Fit CatBoost with native categoricals (`ORIGIN`, `DEST`, `route`, `part_of_day`, `AIRLINE_CODE`).
   - Early stopping on the preceding 6 months as validation split.
4. **Platt Calibrator Fitting**:
   - Fit post-hoc Platt scaling on out-of-fold validation log-odds.
5. **Shadow Evaluation & Pre-Registration**:
   - Run the new candidate model in shadow mode alongside the active champion for 30 days.
   - Verify that candidate PR-AUC is non-inferior to champion ($p \ge 0.05$) and that ECE $\le 0.015$.
6. **Artifact Hashing & Sign-Off**:
   - Compute SHA256 of new artifacts, update `freeze_manifest.json`, and deploy.

---

## 6. Audit Trail & Governance Log

All retraining events, intercept recalibrations, and drift investigations must be logged in `reports/` with:
- Timestamp (UTC) and operator name.
- Old vs new artifact SHA256 digests.
- Validation dataset temporal window and sample size.
- Pre- and post-calibration Brier scores and ROC-AUC metrics.
- Explicit confirmation that test partitions were not used during training or tuning.

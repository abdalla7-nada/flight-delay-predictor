# Pre-Registered Expectations for 2023 Test Set Evaluation

**Author**: Abdallah Nada  
**Project**: US Flight Arrival Delay Prediction (`DELAY = ARR_DELAY > 15`)  
**Evaluation Set**: 2023 Test Set (`data/processed/test.parquet`, 304,458 flights, Jan–Aug 2023).  
**Protocol Status**: Pre-registered and cryptographically locked **BEFORE** scoring or extracting features from the test set.  
**Fixed Random Seed**: `42`  

---

## 1. Cryptographic Freeze Manifest

To enforce experimental integrity and verify that no model refitting, tuning, or feature alterations occurred after observing test data, the SHA256 checksums of all modeling assets and data files were recorded prior to scoring:

| File Path | SHA256 Checksum (First 16 Characters) | File Size (Bytes) | Role in Evaluation |
| :--- | :--- | :---: | :--- |
| `artifacts/feature_pipeline_v2.joblib` | `fdaca2ceb7f84ca0` | 559,397 | Feature engineering pipeline (schedule-only lookups) |
| `artifacts/models/champion_v2.joblib` | `e106c35c0800631c` | 14,885,501 | Champion CatBoost model (33 features) |
| `artifacts/models/calibrator_v2.joblib` | `445f4fc419504ee6` | 949 | Platt probability calibrator (fitted on 2022 validation) |
| `data/processed/train.parquet` | `6b0f6483cbae3339` | 6,906,667 | Historical training dataset (2019 + 2021) |
| `data/processed/val.parquet` | `2894054b137722b1` | 3,473,645 | Historical validation dataset (2022) |
| `data/processed/test.parquet` | `ca6c8ca99d98f3ad` | 2,343,501 | Out-of-sample test dataset (2023 Jan–Aug) |

*Full 64-character hashes are preserved in [reports/freeze_manifest.json](freeze_manifest.json).*

---

## 2. Temporal & Seasonal Alignment: Jan–Aug 2022 Benchmark

The 2023 test set spans **January 1, 2023 through August 31, 2023** (8 months, 243 calendar days).  
To establish a direct apple-to-apple comparison free of seasonal bias (e.g. low-delay autumn vs. summer surge), the champion model was benchmarked on the identical seasonal window of the validation set (**Jan–Aug 2022**, 296,276 flights across 243 days):

- **Jan–Aug 2022 Delay Rate**: **21.06%** (0.2106)
- **Validation PR-AUC (Jan–Aug)**: **0.3355** (95% Day-Clustered Bootstrap CI: **[0.3208, 0.3508]**)
- **Validation ROC-AUC (Jan–Aug)**: **0.6576** (95% Day-Clustered Bootstrap CI: **[0.6481, 0.6649]**)
- **Validation Brier Score (Jan–Aug)**: **0.1572** (95% Day-Clustered Bootstrap CI: **[0.1528, 0.1618]**)
- **Brier Skill Score (BSS vs. base rate)**: **+0.0544 (+5.44%)**
- **Top 10% Precision / Lift**: **40.63% (1.93x lift)**

---

## 3. Ground-Truth Test Base Rate & Prevalence Shift

Examining the ground-truth binary labels (`DELAY`) of the test set—strictly without accessing any feature columns or model predictions—reveals continued secular air travel congestion:

$$\text{Base Rate}_{\text{train (2019+2021)}} = 0.1762 \quad \longrightarrow \quad \text{Base Rate}_{\text{val Jan–Aug (2022)}} = 0.2106 \quad \longrightarrow \quad \mathbf{\text{Base Rate}_{\text{test Jan–Aug (2023)}} = 0.2224}$$

The 2023 test base rate is **22.24%** (+1.18 percentage points higher than Jan–Aug 2022, and +4.62 percentage points higher than the historical training period).

### Prevalence Shift Implications
1. **PR-AUC Prevalence Sensitivity**:  
   Unlike ROC-AUC, the baseline for PR-AUC is equal to the class prevalence $P(Y=1)$. Because the 2023 test prevalence rose from $0.2106$ to $0.2224$, the minimum naive baseline PR-AUC automatically shifts upward by $+0.0118$. Thus, an effective model should achieve a higher nominal PR-AUC on test simply due to prevalence inflation.
2. **Reference Brier Score Shift**:  
   The variance of a Bernoulli random variable is $p(1-p)$. The uninformative reference Brier score on test is:
   $$\text{Brier}_{\text{ref}} = p_{\text{test}}(1 - p_{\text{test}}) = 0.2224 \times (1 - 0.2224) = \mathbf{0.1729}$$
   (compared to $0.1662$ in Jan–Aug 2022 and $0.1451$ in train). Higher base delay rates inherently produce larger uncalibrated squared error losses.

---

## 4. Formal Pre-Registered Metric Expectations

Before running `src/final_test_eval.py`, the following formal target bounds are pre-registered:

### A. Discrimination Expectations
- **ROC-AUC Expected Range**: **[0.648, 0.665]**  
  *Rationale*: ROC-AUC is mathematically invariant to positive class prevalence shifts. The 95% day-clustered CI on Jan–Aug 2022 validation was $[0.6481, 0.6649]$. If the underlying ranking mechanism holds, test ROC-AUC must land squarely within this interval.
  - *Pass Condition*: $0.645 \le \text{ROC-AUC} \le 0.668$.
  - *Anomaly Alert*: $\text{ROC-AUC} < 0.620$ (indicates pipeline degradation/schema mismatch) or $\text{ROC-AUC} > 0.750$ (indicates severe post-departure leakage).

- **PR-AUC Expected Range**: **[0.330, 0.360]**  
  *Rationale*: Accounting for the $+0.0118$ prevalence increase from Jan–Aug 2022 ($0.3355 \pm 0.015$), expected PR-AUC should center near $0.345 \pm 0.015$.
  - *Pass Condition*: $\text{PR-AUC} \ge 0.325$.

### B. Probability Calibration & Error Expectations
- **Brier Score Expected Range**: **[0.158, 0.166]**  
  *Rationale*: With higher base rate variance ($p(1-p) = 0.1729$), squared error will slightly elevate from 2022 ($0.1572$). However, it must remain substantially below the reference baseline ($0.1729$).
- **Brier Skill Score (BSS)**: **$\ge +4.0\%$** vs. the constant test base rate ($0.1729$), and **$\ge +5.0\%$** vs. the historical train constant ($0.1762$).
- **Expected Calibration Error (ECE)**: **$\le 0.035$** (10 bins).  
  *Rationale*: The calibrator was fitted on 2022 validation data ($20.32\%$ base rate). Because 2023 experienced a $+1.92\%$ macroeconomic shift to $22.24\%$, a modest residual underestimation gap ($\sim 1.5\% - 2.0\%$) is expected, slightly elevating ECE compared to in-validation folds.

### C. Operational Risk Stratification Expectations
- **Top 10% Decile Lift**: **$\ge 1.80\text{x}$** (expected range $1.85\text{x} - 2.05\text{x}$).
- **Decile Monotonicity**: Observed delay rates across the 10 risk deciles must be monotonically increasing.

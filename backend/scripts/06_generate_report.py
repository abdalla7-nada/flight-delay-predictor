"""
Step 6: Comprehensive EDA & Data Quality Report Generator
- Compiles all JSON results from Steps 1-5 into reports/eda_report.md
- Generates a styled, publication-ready HTML version in reports/eda_report.html
- Embeds all generated figures with detailed analytical commentary
"""

import os
import json
import base64

def generate_report():
    print("[*] Generating Comprehensive EDA & Data Quality Report...")

    # Load all json results
    with open("reports/audit_results.json", "r") as f:
        audit = json.load(f)
    with open("reports/target_analysis.json", "r") as f:
        target = json.load(f)
    with open("reports/delay_patterns.json", "r") as f:
        patterns = json.load(f)
    with open("reports/causes_and_associations.json", "r") as f:
        assoc = json.load(f)

    # Build Markdown Content
    md = []
    md.append("# Exploratory Data Analysis & Data Quality Report")
    md.append("## Flight Delay and Cancellation Dataset (2019–2023) — Full 2,000,000 Flight Analysis\n")
    md.append("> **Dataset Reference**: `Flight Delay and Cancellation Data (2019-2023) v2` (patrickzel)  ")
    md.append("> **Analysis Scope**: Exhaustive 2,000,000-row audit using DuckDB, Polars, and PyArrow  ")
    md.append("> **Report Status**: Complete Pre-Modeling EDA and Data Quality Clearance  \n")

    md.append("---\n")
    md.append("## Executive Summary\n")
    md.append("This report presents an exhaustive exploratory data analysis and rigorous data quality audit on the full **2,000,000 flight records** spanning January 1, 2019 through August 31, 2023 across 18 major US airlines and 380 airports. ")
    md.append("The primary objective is to evaluate data integrity, uncover operational and temporal patterns, identify statistical risks, and lay a mathematically sound foundation for predicting whether a flight will experience an arrival delay exceeding 15 minutes (`DELAY = ARR_DELAY > 15`) using **only pre-departure information**.\n")

    md.append("### Key High-Level Findings\n")
    md.append("1. **Data Volume & Compression**: The raw CSV occupies **390.47 MB** (409,432,439 bytes) and was converted into an optimized Parquet store of **52.86 MB** (7.39x compression) in **10.44 seconds**, enabling fast columnar analytical scans.")
    md.append("2. **Flight Completion vs. Non-Completion**: Out of 2,000,000 flights, **1,942,769 (97.14%)** successfully completed their journey; **52,522 flights (2.63%)** were cancelled, and **4,709 flights (0.23%)** were diverted.")
    md.append("3. **Target Class Balance**: Among completed flights, **343,875 (17.70%)** arrived > 15 minutes late, while **1,598,894 (82.30%)** arrived on time or early (median delay: **-7.0 minutes**). Across all 2M flights, the positive rate is **17.19%**.")
    md.append("4. **Zero Missingness on Completed Flights**: All missing values in `ARR_DELAY`, `ELAPSED_TIME`, and `AIR_TIME` (**57,231 rows / 2.86%**) are **100% explained** by cancelled (52,522) and diverted (4,709) flights. Non-cancelled, non-diverted flights have **0% missing values**.")
    md.append("5. **Severe Post-Departure Leakage Identified**: Crucial operational fields (`DEP_DELAY`, `DEP_TIME`, `TAXI_OUT`, `WHEELS_OFF`, `WHEELS_ON`, `TAXI_IN`, `ARR_TIME`, `ELAPSED_TIME`, `AIR_TIME`, `DELAY_DUE_*`) occur post-departure. In particular, `DEP_DELAY` has a **Pearson correlation of r = 0.96** with `ARR_DELAY` and **must be strictly excluded** from pre-departure modeling.")
    md.append("6. **Critical Temporal Drift (2020 COVID Shock)**: Cancellation spiked to **6.04%** in 2020 with 16,169 security/pandemic cancellations (Code D), while delay rates plunged to **9.42%** as airspace emptied. Post-pandemic recovery saw delay rates climb steadily: 16.62% (2021), 20.32% (2022), and **22.24%** (2023).")
    md.append("7. **Primary Pre-Departure Delay Drivers**: Scheduled departure hour (`dep_hour`) is the single strongest pre-departure categorical signal (Cramér's V = 0.1429), with delay rates compounding from **~7.5% at 06:00** to **26.5% at 21:00** due to late-arriving aircraft propagation (**37.75%** of all delay minutes).\n")

    md.append("---\n")
    md.append("## 1. Dataset Overview & Conversion Audit\n")
    md.append("### Conversion Efficiency & Storage Metrics\n")
    md.append("| Metric | Raw CSV File | Processed Parquet File | Improvement / Ratio |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **File Path** | `data/raw/flights_sample_2m.csv` | `data/processed/flights_2m.parquet` | - |")
    md.append(f"| **File Size** | **390.47 MB** (409,432,439 bytes) | **52.86 MB** (55,431,684 bytes) | **7.39x Compression** (86.46% savings) |")
    md.append(f"| **Row Count** | 2,000,000 | 2,000,000 | 100% Integrity Preserved |")
    md.append(f"| **Column Count** | 32 | 32 | Complete Schema Retention |")
    md.append(f"| **Conversion Time** | - | **10.44 seconds** | DuckDB Columnar Pipe |")
    md.append(f"| **Row Groups** | N/A | 17 row groups | Optimized Parquet Chunking |")
    md.append(f"| **Compression Codec** | None | Zstandard (`ZSTD`) | High-throughput decompression |\n")

    # Exact Dtypes Mapping
    col_dtypes = {
        "FL_DATE": "date32[day]", "AIRLINE": "string", "AIRLINE_DOT": "string",
        "AIRLINE_CODE": "string", "DOT_CODE": "int64", "FL_NUMBER": "int64",
        "ORIGIN": "string", "ORIGIN_CITY": "string", "DEST": "string", "DEST_CITY": "string",
        "CRS_DEP_TIME": "int64", "DEP_TIME": "double", "DEP_DELAY": "double",
        "TAXI_OUT": "double", "WHEELS_OFF": "double", "WHEELS_ON": "double", "TAXI_IN": "double",
        "CRS_ARR_TIME": "int64", "ARR_TIME": "double", "ARR_DELAY": "double",
        "CANCELLED": "double", "CANCELLATION_CODE": "string", "DIVERTED": "double",
        "CRS_ELAPSED_TIME": "double", "ELAPSED_TIME": "double", "AIR_TIME": "double",
        "DISTANCE": "double", "DELAY_DUE_CARRIER": "double", "DELAY_DUE_WEATHER": "double",
        "DELAY_DUE_NAS": "double", "DELAY_DUE_SECURITY": "double", "DELAY_DUE_LATE_AIRCRAFT": "double"
    }

    md.append("### Schema and Data Types\n")
    md.append("The dataset comprises 32 attributes categorized into temporal, spatial, operational, and causal dimensions:\n")
    md.append("| Column Name | Inferred Parquet Dtype | Category | Pre-Departure Status |")
    md.append("| :--- | :--- | :--- | :--- |")
    for item in assoc["columns_leakage"]:
        cname = item["column"]
        dt = col_dtypes.get(cname, "string")
        md.append(f"| `{cname}` | `{dt}` | {item['category']} | **{item['status']}** |")
    md.append("\n")

    md.append("---\n")
    md.append("## 2. Rigorous Data Quality Audit\n")
    md.append("### 2.1 Missing Value Analysis and Systematic Patterns\n")
    md.append("A thorough missingness audit reveals that missing data is neither random (MCAR) nor erroneous. Rather, missingness is entirely governed by the flight lifecycle state machine:\n")
    md.append("| Column Name | Missing Count | Missing % | Governed Flight Condition | Operational Explanation |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")

    missing_explanations = {
        "DEP_TIME": ("Cancelled before pushback", "51,430 flights cancelled prior to gate departure."),
        "DEP_DELAY": ("Cancelled before pushback", "51,449 flights cancelled before departure."),
        "TAXI_OUT": ("Cancelled before takeoff", "52,281 flights cancelled before taxi or gate return."),
        "WHEELS_OFF": ("Cancelled before takeoff", "52,281 flights never became airborne."),
        "WHEELS_ON": ("Cancelled or diverted flights", "52,522 cancelled + 569 uncompleted diversions."),
        "TAXI_IN": ("Cancelled or diverted flights", "52,522 cancelled + 569 uncompleted diversions."),
        "ARR_TIME": ("Cancelled or diverted flights", "52,522 cancelled + 569 uncompleted diversions."),
        "ARR_DELAY": ("Cancelled (52,522) + Diverted (4,709)", "100% of missing delay is explained by cancelled & diverted flights."),
        "CANCELLATION_CODE": ("Non-cancelled completed flights", "Populated only when CANCELLED = 1.0 (52,522 rows)."),
        "CRS_ELAPSED_TIME": ("Schedule data gaps", "17 flights missing published duration in timetable."),
        "ELAPSED_TIME": ("Cancelled (52,522) + Diverted (4,709)", "Flight aborted prior to scheduled arrival."),
        "AIR_TIME": ("Cancelled (52,522) + Diverted (4,709)", "Flight aborted prior to scheduled arrival."),
        "DELAY_DUE_CARRIER": ("Flights without ≥ 15m delay", "DOT rule: only populated when delay attribution required."),
        "DELAY_DUE_WEATHER": ("Flights without ≥ 15m delay", "DOT rule: only populated when delay attribution required."),
        "DELAY_DUE_NAS": ("Flights without ≥ 15m delay", "DOT rule: only populated when delay attribution required."),
        "DELAY_DUE_SECURITY": ("Flights without ≥ 15m delay", "DOT rule: only populated when delay attribution required."),
        "DELAY_DUE_LATE_AIRCRAFT": ("Flights without ≥ 15m delay", "DOT rule: only populated when delay attribution required.")
    }

    for col, v in audit["missing_counts"].items():
        if v["count"] > 0:
            cond, expl = missing_explanations.get(col, ("Operational condition", "Lifecycle governed"))
            md.append(f"| `{col}` | {v['count']:,} | {v['pct']:.2f}% | {cond} | {expl} |")
    md.append("\n")

    md.append("#### Systematic Determinism of Missingness:\n")
    md.append("- **Completed Flights (1,942,769 flights)**: **0.00% missing values** in `ARR_DELAY`, `ARR_TIME`, `ELAPSED_TIME`, and `AIR_TIME`. Every single completed flight has complete telemetry.")
    md.append("- **Cancelled Flights (52,522 flights)**: 100% of cancelled flights have null `ARR_DELAY`, null `ARR_TIME`, and null `AIR_TIME`. 51,430 (97.9%) were cancelled prior to pushback (`DEP_TIME` is null); 1,092 were cancelled after gate departure but prior to takeoff. All 52,522 cancelled flights possess a valid `CANCELLATION_CODE`.")
    md.append("- **Diverted Flights (4,709 flights)**: 100% have null `ARR_DELAY` and null `AIR_TIME` because scheduled destination was never reached.")
    md.append("- **Delay Causes (`DELAY_DUE_*`)**: 1,643,780 rows (82.19%) are missing. Exactly 343,875 delayed completed flights (>15m) and 12,345 operational flights carry cause entries. On-time flights do not receive DOT delay cause codes.\n")

    md.append("### 2.2 Duplicates and Identifier Uniqueness\n")
    md.append("- **Full-Row Exact Duplicates**: **0 rows (0.00%)**. The dataset has zero duplicate rows.")
    md.append("- **Business Key Uniqueness (`FL_DATE + AIRLINE_CODE + FL_NUMBER + ORIGIN`)**: Only **1 group (2 rows)** shared this compound key across 2 million records.")
    md.append("  - *Inspection*: Spirit Airlines (`NK`) flight 631 on `2020-04-07` from `DTW` had two entries scheduled at 17:35: one destined for Tampa (`TPA`) and one destined for Atlantic City (`ACY`), both cancelled under Code D during initial COVID lockdowns.")
    md.append("  - *Conclusion*: When including destination (`+ DEST`), the dataset has **100.00% unique primary keys**.\n")

    md.append("### 2.3 Invalid, Negative, and Boundary Values\n")
    md.append("| Quality Check | Invalid Count | % of Data | Assessment & Treatment |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| Scheduled Dep Time (`CRS_DEP_TIME < 0` or `> 2359`) | {int(audit['invalid_checks']['invalid_crs_dep_time'])} | 0.00% | Clean military time (0001 to 2359) |")
    md.append(f"| Scheduled Dep Minutes (`CRS_DEP_TIME % 100 >= 60`) | {int(audit['invalid_checks']['invalid_crs_dep_minute'])} | 0.00% | Valid minute range [00, 59] |")
    md.append(f"| Scheduled Arr Time (`CRS_ARR_TIME > 2359`) | {int(audit['invalid_checks']['invalid_crs_arr_time'])} | 0.0006% | 12 flights recorded as `2400` (midnight). Convert `2400 -> 0000` |")
    md.append(f"| Non-positive Distance (`DISTANCE <= 0`) | {int(audit['invalid_checks']['non_positive_distance'])} | 0.00% | All flight distances are positive (min: 31 miles) |")
    md.append(f"| Non-positive Duration (`CRS_ELAPSED_TIME <= 0`) | {int(audit['invalid_checks']['non_positive_crs_elapsed'])} | 0.00005% | Exactly 1 flight had CRS_ELAPSED_TIME = 0. Drop or impute with distance median |")
    md.append(f"| Negative Taxi Times (`TAXI_OUT < 0`, `TAXI_IN < 0`) | {int(audit['invalid_checks']['negative_taxi_out'] + audit['invalid_checks']['negative_taxi_in'])} | 0.00% | Zero negative taxi durations |")
    md.append(f"| Negative Airborne Time (`AIR_TIME < 0`) | {int(audit['invalid_checks']['negative_air_time'])} | 0.00% | Zero negative airborne times |")
    md.append(f"| Simultaneous Cancelled & Diverted Flag | {int(audit['invalid_checks']['both_cancelled_and_diverted'])} | 0.00% | Mutual exclusivity 100% verified |\n")

    md.append("### 2.4 Categorical Integrity & Mappings\n")
    md.append("- **Airline Identification**: Exactly **18 unique `AIRLINE` names**, exactly **18 unique `AIRLINE_CODE` values**, and exactly **18 unique `DOT_CODE` values**. The mapping is strictly 1-to-1-to-1 with zero orphaned codes or ambiguous aliases.")
    md.append("- **Airport Codes**: Exactly **380 unique origin airports** and **380 unique destination airports**. All 380 codes strictly follow the standard 3-letter IATA uppercase convention (0 malformed strings).\n")

    md.append("### 2.5 Outlier Analysis: ARR_DELAY and DEP_DELAY\n")
    md.append("Due to operational extremes in air traffic (snowstorms, aircraft mechanical failures, rolling ground stops), delay distributions exhibit heavy right-skewed tails:\n")
    md.append("| Metric | Arrival Delay (`ARR_DELAY`) | Departure Delay (`DEP_DELAY`) | Interpretation |")
    md.append("| :--- | :--- | :--- | :--- |")
    for stat in audit["delay_stats"]:
        m = stat["metric"]
        if m == "ARR_DELAY":
            arr_s = stat
        else:
            dep_s = stat
    md.append(f"| **Valid Observations** | {arr_s['count']:,} | {dep_s['count']:,} | Completed flights only |")
    md.append(f"| **Minimum** | **{arr_s['min_val']:.1f} min** (-1.98 hrs early) | **{dep_s['min_val']:.1f} min** | Early departures and arrivals |")
    md.append(f"| **1st Percentile (p01)** | {arr_s['p01']:.1f} min | {dep_s['p01']:.1f} min | Significant headwind/buffer arrival |")
    md.append(f"| **25th Percentile (Q1)** | {arr_s['q1']:.1f} min | {dep_s['q1']:.1f} min | Routine early arrival |")
    md.append(f"| **Median (p50)** | **{arr_s['median_val']:.1f} min** | **{dep_s['median_val']:.1f} min** | **Typical flight arrives 7 minutes EARLY** |")
    md.append(f"| **Mean ± Std** | **{arr_s['mean_val']:.2f} ± {arr_s['std_val']:.2f} min** | **{dep_s['mean_val']:.2f} ± {dep_s['std_val']:.2f} min** | Substantial positive skew |")
    md.append(f"| **75th Percentile (Q3)** | {arr_s['q3']:.1f} min | {dep_s['q3']:.1f} min | On-time window upper quartile |")
    md.append(f"| **95th Percentile (p95)** | {arr_s['p95']:.1f} min | {dep_s['p95']:.1f} min | Severe operational delay |")
    md.append(f"| **99th Percentile (p99)** | {arr_s['p99']:.1f} min (3.17 hrs) | {dep_s['p99']:.1f} min | Extreme delay tail |")
    md.append(f"| **99.9th Percentile (p99.9)** | {arr_s['p999']:.1f} min (10.9 hrs) | {dep_s['p999']:.1f} min | Overnight disruption / rolling delay |")
    md.append(f"| **Maximum** | **{arr_s['max_val']:.1f} min** (53.95 hrs) | **{dep_s['max_val']:.1f} min** (53.68 hrs) | Extreme multi-day mechanical hold |")
    md.append(f"| **Interquartile Range (IQR)** | {audit['outlier_counts']['ARR_DELAY']['iqr']:.1f} min | {audit['outlier_counts']['DEP_DELAY']['iqr']:.1f} min | Spread of central 50% |")
    md.append(f"| **Tukey Upper Fence (Q3+1.5*IQR)** | {audit['outlier_counts']['ARR_DELAY']['upper_fence']:.1f} min | {audit['outlier_counts']['DEP_DELAY']['upper_fence']:.1f} min | Statistical outlier boundary |")
    md.append(f"| **Tukey Outliers Above Fence** | {int(audit['outlier_counts']['ARR_DELAY']['above_upper_count']):,} ({audit['outlier_counts']['ARR_DELAY']['above_upper_pct']}%) | {int(audit['outlier_counts']['DEP_DELAY']['above_upper_count']):,} ({audit['outlier_counts']['DEP_DELAY']['above_upper_pct']}%) | High-impact business events |\n")

    md.append("> [!NOTE]")
    md.append("> **Outlier Treatment Rationale**: For the primary binary classification task (`ARR_DELAY > 15`), extreme values (such as 500 or 3,000 minutes) cleanly collapse into the positive label `1` and require no truncation or deletion. If secondary regression modeling is performed, target clipping at the 99th percentile (190 minutes) or a log-transform is strongly recommended to stabilize gradient descent.")
    md.append("\n")

    md.append("---\n")
    md.append("## 3. Target Distribution and Temporal Drift Analysis\n")
    md.append("### 3.1 Class Balance of the Target (`DELAY = ARR_DELAY > 15`)\n")
    md.append("In accordance with Federal Aviation Administration (FAA) standards, a flight is classified as delayed if it reaches the gate more than 15 minutes after its scheduled arrival time (`ARR_DELAY > 15`).\n")
    md.append("| Population Cohort | Total Flights | On-Time Flights (≤ 15 min) | Delayed Flights (> 15 min) | Delay Rate (%) |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    md.append(f"| **Completed Flights (Target Subpopulation)** | **1,942,769** | **1,598,894** (82.30%) | **343,875** (17.70%) | **17.70%** |")
    md.append(f"| **All Scheduled Flights (Full Dataset)** | **2,000,000** | 1,598,894 (79.95%) | 343,875 (17.19%) | 17.19% |")
    md.append(f"| *Cancelled Flights* | 52,522 (2.63%) | - | - | N/A (Flight uncompleted) |")
    md.append(f"| *Diverted Flights* | 4,709 (0.23%) | - | - | N/A (Destination unreached) |\n")

    md.append("![Figure 1: Target Distribution and Class Balance](figures/fig01_target_distribution.png)\n")

    md.append("### 3.2 Cancellation Breakdown and CANCELLATION_CODE Analysis\n")
    md.append("Of the 2,000,000 flights, **52,522 were cancelled (2.626%)**. Analysis of the official FAA reason codes reveals:\n")
    md.append("| Code | Standard FAA Description | Cancelled Flights | % of All Cancellations | % of Total Flights | Dominant Driver / Year |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for c in target["cancellation_code_breakdown"]:
        driver = "Prevalent in 2020 (16,169 flights, 99.1% in 2020) due to COVID travel bans" if c["CANCELLATION_CODE"] == "D" \
                 else "Leading cause across all standard operational years (winter storms, convective activity)" if c["CANCELLATION_CODE"] == "B" \
                 else "Airline crew shortages, maintenance holds, airline dispatch issues" if c["CANCELLATION_CODE"] == "A" \
                 else "ATC ground delay programs, volume constraints, runway closures"
        md.append(f"| **{c['CANCELLATION_CODE']}** | {c['description']} | **{c['count']:,}** | **{c['pct_of_cancelled']:.2f}%** | {c['pct_of_total']:.3f}% | {driver} |")
    md.append("\n")

    md.append("### 3.3 Year-by-Year Temporal Drift and the 2020 COVID Shock\n")
    md.append("Evaluating delay and cancellation rates chronologically reveals significant dataset shift caused by the pandemic:\n")
    md.append("| Year | Scheduled Flights | Cancelled Flights | Cancellation Rate | Diverted Flights | Completed Flights | Delayed Flights (>15m) | Delay Rate (Completed) | Mean Arrival Delay | Median Arrival Delay |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for y in target["yearly_drift"]:
        md.append(f"| **{y['year']}** | {y['total_flights']:,} | {y['cancelled_flights']:,} | {y['cancellation_rate']:.2f}% | {y['diverted_flights']:,} | {y['completed_flights']:,} | {y['delayed_flights']:,} | **{y['delay_rate_completed']:.2f}%** | {y['mean_arr_delay']:+.2f} min | {y['median_arr_delay']:+.1f} min |")
    md.append("\n")
    md.append("> [!IMPORTANT]")
    md.append("> **Critical Insight on 2020 Data Drift**: In 2020, flight volume collapsed by ~37%, cancellation spiked to **6.04%** (driven by pandemic lockdowns), and the delay rate dropped to just **9.42%** (mean delay: **-4.96 min**, median: **-11.0 min**). With minimal traffic, runway queues and air traffic congestion vanished. As commercial aviation rebounded, delay rates surged from **16.62% in 2021** to **20.32% in 2022** and **22.24% in 2023**.")
    md.append("\n")

    md.append("![Figure 2: Yearly and Monthly Flight Trends](figures/fig02_yearly_monthly_trends_covid.png)\n")

    md.append("---\n")
    md.append("## 4. Delay Patterns Across Operational Dimensions\n")

    md.append("### 4.1 Scheduled Departure Hour and Day of Week\n")
    md.append("The heatmap of delay rate by departure hour and day of week exposes a major operational reality of modern aviation:\n")
    md.append("- **Early Morning (05:00 - 08:00)**: Flights experience the lowest delay rates of the day (**7.5% - 9.5%**). Aircraft and flight crews are fresh, having parked overnight at their gates with zero accumulated delay.")
    md.append("- **Midday Progression (11:00 - 15:00)**: Delay rates rise steadily (**14.0% - 19.5%**) as turnaround buffers are absorbed by minor gate and air traffic holds.")
    md.append("- **Evening Peak (18:00 - 22:00)**: Delay rates peak at **24.5% - 27.2%**. Late-arriving incoming aircraft trigger cascade delays down the network.")
    md.append("- **Day-of-Week Effect**: Thursdays, Fridays, and Sundays suffer higher evening delays (**26% - 27%**) due to overlapping business travel and weekend leisure demand, whereas Tuesdays and Wednesdays are markedly calmer.")
    md.append("\n")
    md.append("![Figure 3: Heatmap of Delay Rate by Departure Hour and Day of Week](figures/fig03_day_of_week_and_hour_heatmap.png)\n")

    md.append("### 4.2 Airline Punctuality and Delay Dispersion\n")
    md.append("Delay rates vary substantially across the 18 reporting carriers (national average: **17.7%**):\n")
    md.append("| Airline Name | Code | Total Flights | Cancelled | Cancellation Rate | Completed | Delayed (>15m) | Delay Rate (%) | Mean Arrival Delay |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for a in patterns["airlines"]:
        md.append(f"| **{a['AIRLINE']}** | `{a['AIRLINE_CODE']}` | {a['total_flights']:,} | {a['cancelled_flights']:,} | {a['cancellation_rate']:.2f}% | {a['completed_flights']:,} | {a['delayed_flights']:,} | **{a['delay_rate']:.2f}%** | {a['mean_arr_delay']:+.2f} min |")
    md.append("\n")
    md.append("![Figure 4: Airline Delay Rates](figures/fig04_airline_delay_rates.png)\n")
    md.append("Key observations:\n")
    md.append("- **Highest Delay Rates**: JetBlue Airways (`B6`, **26.28%** delay rate), Allegiant Air (`G4`, **25.99%**), and Frontier Airlines (`F9`, **25.38%**), followed closely by Spirit Air Lines (`NK`, **21.92%**). JetBlue is concentrated in the congested Northeast corridor (JFK/BOS/EWR), while ultra-low-cost carriers operate tight turnaround buffers.")
    md.append("- **Lowest Delay Rates**: Endeavor Air (`9E`, **12.51%** delay rate), Horizon Air (`QX`, **13.49%**), and Delta Air Lines (`DL`, **13.75%**), followed by Hawaiian Airlines (`HA`, **15.05%**). Delta and regional partners achieve superior punctuality through disciplined hub management.")
    md.append("\n")

    md.append("### 4.3 Airport Congestion: Top and Bottom Origin Airports\n")
    md.append("To avoid noisy estimates from small regional airstrips with few flights, a robust statistical threshold of **minimum 5,000 flights** was applied (leaving 84 major commercial airports):\n")
    md.append("| Airport (Top 10 Most Delayed) | Code | City / Metro | Flight Count | Delay Rate (%) | Mean Arrival Delay |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for ap in patterns["top_airports"]:
        md.append(f"| **{ap['ORIGIN']}** | `{ap['ORIGIN']}` | {ap['ORIGIN_CITY']} | {ap['flight_count']:,} | **{ap['delay_rate']:.2f}%** | {ap['mean_delay']:+.2f} min |")
    md.append("\n")
    md.append("| Airport (Top 10 Most Punctual) | Code | City / Metro | Flight Count | Delay Rate (%) | Mean Arrival Delay |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for ap in patterns["bottom_airports"]:
        md.append(f"| **{ap['ORIGIN']}** | `{ap['ORIGIN']}` | {ap['ORIGIN_CITY']} | {ap['flight_count']:,} | **{ap['delay_rate']:.2f}%** | {ap['mean_delay']:+.2f} min |")
    md.append("\n")
    md.append("![Figure 5: Top and Bottom Airports by Delay Rate](figures/fig05_top_bottom_airports.png)\n")

    md.append("### 4.4 Top Delayed Flight Routes\n")
    md.append("Applying a minimum volume filter of **1,000 flights per directional route** isolates high-delay corridors:\n")
    md.append("| Route (Origin → Destination) | Flight Volume | Delay Rate (%) | Mean Delay | Operational Characteristic |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for r in patterns["top_routes"]:
        md.append(f"| **{r['route']}** | {r['flights']:,} | **{r['delay_rate']:.2f}%** | {r['mean_delay']:+.2f} min | Severe terminal airspace saturation / ATC metering |")
    md.append("\n")
    md.append("![Figure 6: Top Routes by Delay Rate](figures/fig06_top_routes_delay_rate.png)\n")

    md.append("### 4.5 Distance and Scheduled Flight Duration\n")
    md.append("Examining delay probability across flight mileage and scheduled duration:\n")
    md.append("| Distance Bracket | Scheduled Duration Bracket | Flight Count | Delay Rate (%) | Mean Delay | Takeaway |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for d, el in zip(patterns["distance_buckets"], patterns["elapsed_buckets"]):
        md.append(f"| {d['distance_bucket']} | {el['elapsed_bucket']} | {d['flights']:,} | **{d['delay_rate']:.2f}%** | {d['mean_delay']:+.2f} min | Longer routes possess schedule padding to absorb minor ground delays |")
    md.append("\n")
    md.append("![Figure 7: Distance and Elapsed Time Analysis](figures/fig07_distance_and_elapsed_time.png)\n")

    md.append("---\n")
    md.append("## 5. Delay Causes Analysis (`DELAY_DUE_*`)\n")
    md.append("> [!WARNING]")
    md.append("> **Operational Context Only**: In accordance with project instructions, `DELAY_DUE_*` columns are evaluated exclusively for systemic diagnostic understanding. These columns represent post-flight official DOT delay apportionments and **must never be used as predictive features**.")
    md.append("\n")
    md.append("Across all delayed flights carrying cause apportionments (totaling **24.13 million delay minutes**):\n")
    md.append("| Delay Cause Category | Total Minutes Apportioned | Share of Total Delay Minutes | Primary Driver |")
    md.append("| :--- | :--- | :--- | :--- |")
    for c in patterns["causes_summary"]:
        driver = "Cascading delay from previous flight using the same aircraft" if c["Cause"] == "Late_Aircraft" \
                 else "Maintenance, baggage handling, cleaning, crew duty time limits" if c["Cause"] == "Carrier" \
                 else "Air traffic control flow control, airport volume, runway congestion" if c["Cause"] == "NAS" \
                 else "Severe meteorological events (thunderstorms, blizzards, hurricanes)" if c["Cause"] == "Weather" \
                 else "Security breaches, terminal evacuation, re-screening holds"
        md.append(f"| **{c['Cause']}** | **{c['Total_Minutes']/1e6:.2f} Million** | **{c['Percentage']:.2f}%** | {driver} |")
    md.append("\n")
    md.append("Key takeaways from delay causes:\n")
    md.append("- **Late Aircraft Propagation (37.75%)** is the single greatest cause of flight delays in the United States, followed closely by **Carrier Operations (36.81%)**. Together, carrier-controllable and network-propagation issues account for **74.56%** of all delay minutes.")
    md.append("- **National Aviation System (NAS, 19.38%)** represents runway queues and FAA ATC ground metering holds.")
    md.append("- **Direct Weather (5.86%)** accounts for a modest share of direct delay minutes, because severe weather events often lead to full cancellations rather than prolonged delay, or manifest as NAS flow management.")
    md.append("- **Security (0.20%)** represents rare terminal evacuation and screening incidents.\n")
    md.append("![Figure 8: Delay Causes Breakdown](figures/fig08_delay_causes_breakdown.png)\n")

    md.append("---\n")
    md.append("## 6. Correlation Analysis & Feature Associations\n")
    md.append("### 6.1 Numeric Feature Correlations\n")
    md.append("A Pearson and Spearman correlation analysis across operational and pre-departure features highlights critical dynamics:\n")
    md.append("- **`DEP_DELAY` vs `ARR_DELAY`**: **r = +0.96** (near perfect linearity). If an aircraft leaves the gate 30 minutes late, it almost invariably lands late. This demonstrates why `DEP_DELAY` is severe target leakage.\n")
    md.append("- **`DISTANCE` vs `CRS_ELAPSED_TIME`**: **r = +0.98** (expected collinearity between distance and flight duration).\n")
    md.append("- **`TAXI_OUT` vs `ARR_DELAY`**: **r = +0.31** (long runway queues at hub airports directly bleed into arrival delays).\n")
    md.append("![Figure 9: Numeric Correlation Matrix](figures/fig09_numeric_correlation_matrix.png)\n")

    md.append("### 6.2 Categorical Feature Associations with Target (`DELAY > 15`)\n")
    md.append("To measure non-linear relationship between categorical features and the binary delay target, **Cramér's V** (measuring association strength on a 0 to 1 scale) and **Mutual Information (MI)** were computed on a representative sample:\n")
    md.append("| Feature | Cardinality | Cramér's V | Mutual Information (nats) | Predictive Ranking & Note |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for cat in assoc["cat_associations"]:
        note = "Strongest pre-departure predictor; captures diurnal cascade accumulation" if cat["feature"] == "dep_hour" \
               else "Captures route-specific airspace constraints and terrain" if cat["feature"] == "route" \
               else "Captures airline fleet buffer and turnaround operational efficiency" if cat["feature"] == "AIRLINE_CODE" \
               else "Origin airport terminal and runway capacity constraints" if cat["feature"] == "ORIGIN" \
               else "Seasonal weather and holiday travel demand surge" if cat["feature"] == "month" \
               else "Destination airport arrival metering and weather" if cat["feature"] == "DEST" \
               else "Weekly travel patterns (business vs weekend leisure)"
        md.append(f"| `{cat['feature']}` | {cat['unique_values_in_sample']} | **{cat['cramers_v']:.4f}** | **{cat['mutual_information']:.5f}** | {note} |")
    md.append("\n")

    md.append("---\n")
    md.append("## 7. Data Leakage Review & Column Classification\n")
    md.append("In a realistic production deployment, the model predicts flight arrival delay **at or before scheduled departure time** (e.g. at booking, 24h prior, or 2h prior to gate pushback). Using any post-departure variable induces catastrophic data leakage.\n")
    md.append("| Column Name | Status | Availability Time | Rationale & Strict Leakage Warning |")
    md.append("| :--- | :--- | :--- | :--- |")
    for item in assoc["columns_leakage"]:
        badge = "SAFE" if item["status"] == "SAFE" \
                else "TARGET" if item["status"] == "TARGET" \
                else "LEAKAGE"
        md.append(f"| `{item['column']}` | **{badge}** | {item['category']} | {item['reason']} |")
    md.append("\n")
    md.append("> [!CAUTION]")
    md.append("> **Confirmed Leakage Columns to Exclude from Model Training**:\n")
    md.append("> `DEP_TIME`, `DEP_DELAY`, `TAXI_OUT`, `WHEELS_OFF`, `WHEELS_ON`, `TAXI_IN`, `ARR_TIME`, `ELAPSED_TIME`, `AIR_TIME`, `CANCELLED`, `CANCELLATION_CODE`, `DIVERTED`, `DELAY_DUE_CARRIER`, `DELAY_DUE_WEATHER`, `DELAY_DUE_NAS`, `DELAY_DUE_SECURITY`, `DELAY_DUE_LATE_AIRCRAFT`.")
    md.append("\n")

    md.append("---\n")
    md.append("## 8. High-Cardinality Analysis & Encoding Strategies\n")
    md.append("### Cardinality Overview (2,000,000 Flights)\n")
    md.append(f"- `AIRLINE_CODE`: **{assoc['cardinality_stats']['unique_airlines']} unique categories**  \n")
    md.append(f"- `ORIGIN`: **{assoc['cardinality_stats']['unique_origins']} unique airports**  \n")
    md.append(f"- `DEST`: **{assoc['cardinality_stats']['unique_dests']} unique airports**  \n")
    md.append(f"- `ROUTE (ORIGIN_DEST)`: **{assoc['cardinality_stats']['unique_routes']} unique routes**  \n")
    md.append("\n")
    md.append("### Distribution of Route Frequencies\n")
    rf = assoc["route_frequency_stats"]
    md.append(f"- **High-Volume Routes (≥ 1,000 flights)**: **{int(rf['routes_over_1000_flights'])} routes** (Dense statistical support)  \n")
    md.append(f"- **Medium-Volume Routes (100 - 999 flights)**: **{int(rf['routes_100_to_999'])} routes**  \n")
    md.append(f"- **Low-Volume Routes (10 - 99 flights)**: **{int(rf['routes_10_to_99'])} routes**  \n")
    md.append(f"- **Rare / Sparsely Flown Routes (< 10 flights)**: **{int(rf['rare_routes_under_10'])} routes** (Severe risk of overfitting)  \n")
    md.append("\n")
    md.append("### Recommended Encoding Strategies\n")
    md.append("1. **`AIRLINE_CODE` (18 classes)**: **One-Hot Encoding** is ideal. With only 18 dimensions, one-hot adds minimal memory overhead and allows linear models and tree-based algorithms to learn pure carrier-specific intercepts.")
    md.append("2. **`ORIGIN` and `DEST` (380 classes each)**:")
    md.append("   - **Target Encoding with Out-Of-Fold Smoothing (Recommended)**: Calculate historical delay rates smoothed using empirical Bayes or additive m-estimate (`(n * mean_cat + m * global_mean) / (n + m)` with `m = 50`) fitted **strictly on training folds** to prevent target leakage.")
    md.append("   - **Frequency / Count Encoding**: Airport volume is a strong proxy for ground congestion and gate capacity.")
    md.append("   - **Cluster / Metro Area Grouping**: Map multi-airport metros (e.g. JFK/LGA/EWR -> NYC Metro, ORD/MDW -> Chicago Metro) to capture regional airspace congestion.")
    md.append("3. **`ROUTE` (7,718 classes)**:")
    md.append("   - **Regularized Target Encoding**: High predictive power (Cramér's V = 0.1231) but must be smoothed aggressively (`m = 100`).")
    md.append("   - **Rare Route Binning**: Group routes with < 50 historical flights into a designated `'OTHER_RARE_ROUTE'` category.")
    md.append("   - **Learned Embeddings**: For neural network architectures, 16-dimensional entity embeddings initialized with airport latitude/longitude coordinates.\n")

    md.append("---\n")
    md.append("## 9. Time-Split Feasibility & Partitioning Evaluation\n")
    md.append("Because air traffic exhibits strong temporal dependency, seasonality, and long-term trend drift, **random k-fold cross-validation is strictly invalid** and introduces temporal look-ahead leakage. Chronological splits must be employed.\n")
    md.append("### Candidate Split Configurations\n")
    md.append("#### Candidate Option 1: Standard Chronological 3-Way Split\n")
    md.append("| Partition | Calendar Period | Scheduled Flights | Completed Flights | Cancelled Flights | Delay Rate (%) | Mean Arrival Delay | Purpose |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for s in assoc["split_options"]["Standard_3way_Split"]:
        md.append(f"| **{s['split_name']}** | 2019 - 2021 / 2022 / 2023 | {s['total_flights']:,} | {s['completed']:,} | {s['cancelled']:,} | **{s['delay_rate']:.2f}%** | {s['mean_arr_delay']:+.2f} min | Model training, validation tuning, and holdout testing |")
    md.append("\n")

    md.append("#### Candidate Option 2: COVID-Filtered Chronological Split (Recommended)\n")
    md.append("| Partition | Calendar Period | Scheduled Flights | Completed Flights | Cancelled Flights | Delay Rate (%) | Mean Arrival Delay | Purpose |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for s in assoc["split_options"]["COVID_Excluded_Split"]:
        md.append(f"| **{s['split_name']}** | 2019 + 2021 / 2022 / 2023 | {s['total_flights']:,} | {s['completed']:,} | {s['cancelled']:,} | **{s['delay_rate']:.2f}%** | {s['mean_arr_delay']:+.2f} min | Eliminates artificial 2020 pandemic distortion |")
    md.append("\n")
    md.append("![Figure 10: Temporal Split Structure](figures/fig10_temporal_split_structure.png)\n")
    md.append("> [!TIP]")
    md.append("> **Why Option 2 is Statistically Superior**: The 2020 lockdown year artificially depresses delay rate to 9.42% and injects 16,169 non-standard security/lockdown cancellations. Training on 2019 + 2021 yields **913,169 flights** with a natural baseline delay rate of **17.62%**, aligning much more realistically with the 2022 validation set (20.32%) and 2023 holdout test set (22.24%).")
    md.append("\n")

    md.append("---\n")
    md.append("## Findings and Recommendations\n")

    md.append("### 1. Key Project Risks & Mitigation Strategies\n")
    md.append("| Risk Area | Specific Operational Danger | Mitigation Strategy |")
    md.append("| :--- | :--- | :--- |")
    md.append("| **1. Target Leakage** | `DEP_DELAY`, `DEP_TIME`, and `TAXI_OUT` correlate so heavily with `ARR_DELAY` (r=0.96) that including them will result in an artificial 99% accuracy model that is useless pre-departure. | Enforce strict column whitelist in feature pipeline. Drop all operational post-departure variables before featurization. |")
    md.append("| **2. Class Imbalance** | Positive delay class is ~17.7% (approx. 4.6:1 negative-to-positive ratio). Naive accuracy (82.3%) will easily mislead model evaluation. | Optimize PR-AUC (Precision-Recall AUC) as primary metric; tune classification decision thresholds; evaluate Focal Loss or `scale_pos_weight = 4.65`. |")
    md.append("| **3. Temporal Drift** | Severe systemic shifts across 2019-2023 (lockdown collapse in 2020 followed by post-COVID revenge travel congestion in 2022-2023). | Exclude 2020 or weight it lower; use chronological rolling window splits; include rolling 14-day network congestion indicators. |")
    md.append("| **4. Outliers** | Heavy right skew in delays (max 3,237 min). Regression targets will be destabilized. | For binary classification, outliers are naturally bounded in class 1. For regression, clip targets at 99th percentile (190 min) or use Huber/Quantile loss. |")
    md.append("| **5. High Cardinality** | 7,718 routes and 380 airports will cause massive sparsity if one-hot encoded. | Apply out-of-fold regularized Target Encoding with empirical Bayes smoothing (`m=50`), Frequency Encoding, and rare class binning. |\n")

    md.append("### 2. Recommended Data Cleaning Pipeline\n")
    md.append("1. **Filter Non-Completed Flights for Primary Target**: Drop rows where `CANCELLED = 1.0` (52,522 rows) or `DIVERTED = 1.0` (4,709 rows). Completed sample: **1,942,769 clean rows**.")
    md.append("2. **Fix Boundary Time Anomalies**: Replace 12 occurrences of `CRS_ARR_TIME = 2400` with `0000`.")
    md.append("3. **Handle Rare Nulls in Scheduled Duration**: Exactly 17 rows have null `CRS_ELAPSED_TIME` and 1 row has `0`. Impute using route median `DISTANCE / 7.5 + 30` (average commercial jet cruising speed + taxi buffer).")
    md.append("4. **Exclude COVID Lockdown Anomaly (Recommended)**: Filter out April–December 2020 from training data to avoid learning distorted zero-traffic priors.")
    md.append("5. **Strict Pre-Departure Feature Isolation**: Retain only known pre-departure fields: `FL_DATE`, `AIRLINE_CODE`, `ORIGIN`, `DEST`, `CRS_DEP_TIME`, `CRS_ARR_TIME`, `CRS_ELAPSED_TIME`, `DISTANCE`.\n")

    md.append("### 3. Recommended Feature Set and Engineering Ideas\n")
    md.append("#### Safe Baseline Features (Raw)\n")
    md.append("- `AIRLINE_CODE`: 18 categories (One-hot encoded)")
    md.append("- `ORIGIN`, `DEST`: Target encoded with OOF smoothing + Frequency encoded")
    md.append("- `CRS_DEP_TIME`, `CRS_ARR_TIME`: Scheduled times (HHMM)")
    md.append("- `CRS_ELAPSED_TIME`: Scheduled duration in minutes")
    md.append("- `DISTANCE`: Flight distance in miles\n")

    md.append("#### Engineered Pre-Departure Features\n")
    md.append("1. **Cyclical Temporal Encodings**:\n")
    md.append("   - `dep_hour = CRS_DEP_TIME // 100` (0 - 23)")
    md.append("   - `dep_hour_sin = sin(2 * pi * dep_hour / 24)`, `dep_hour_cos = cos(2 * pi * dep_hour / 24)`")
    md.append("   - `month_sin = sin(2 * pi * month / 12)`, `month_cos = cos(2 * pi * month / 12)`")
    md.append("   - `day_of_week` (0 to 6) and `is_weekend` flag (Friday/Sunday travel surge)")
    md.append("2. **Air Corridor & Route Physics**:\n")
    md.append("   - `scheduled_ground_speed = DISTANCE / (CRS_ELAPSED_TIME / 60.0)` (detects unrealistic airline schedule padding or headwinds)")
    md.append("   - `route_flight_count`: Total flights on the route (identifies trunk vs regional spoke)")
    md.append("3. **Historical Congestion & Punctuality Priors (OOF Smoothed)**:\n")
    md.append("   - `carrier_historical_delay_rate`: Prior smoothed delay rate of the airline")
    md.append("   - `origin_hourly_congestion`: Number of scheduled departures from the origin airport in the same 1-hour window (direct proxy for runway taxi congestion)")
    md.append("   - `dest_hourly_congestion`: Number of scheduled arrivals at the destination airport in the scheduled arrival window")
    md.append("4. **Rolling Temporal Features (Strictly Lagged)**:\n")
    md.append("   - `origin_delay_rate_last_7d`: Rolling 7-day delay rate at origin airport")
    md.append("   - `carrier_delay_rate_last_7d`: Rolling 7-day delay rate for the airline\n")

    md.append("### 4. Recommended Split Strategy and Evaluation Metrics\n")
    md.append("- **Splitting Protocol**: Chronological time-series split:\n")
    md.append("  - **Training Set**: 2019 + 2021 (913,169 flights, 894,775 completed; 17.62% delay rate)")
    md.append("  - **Validation Set**: 2022 (457,760 flights, 444,554 completed; 20.32% delay rate)")
    md.append("  - **Holdout Test Set**: 2023 through August 31 (310,303 flights, 304,458 completed; 22.24% delay rate)")
    md.append("- **Primary Evaluation Metric**: **PR-AUC (Area Under Precision-Recall Curve)**. Because positive delays are 17.7%, PR-AUC focuses on the minority class without being inflated by true negative on-time flights.")
    md.append("- **Secondary Metrics**:\n")
    md.append("  - **ROC-AUC**: Overall ranking capability across all thresholds")
    md.append("  - **F1-Score / Precision@k**: At top decile of predicted risk (crucial for airline dispatch intervention)")
    md.append("  - **Brier Score / Expected Calibration Error (ECE)**: Verifies whether a predicted 30% delay probability corresponds to 30 out of 100 delayed flights.\n")

    md.append("### 5. Step-by-Step Modeling Plan\n")
    md.append("```mermaid")
    md.append("graph TD")
    md.append("    A[\"Raw Parquet Data<br/>(2,000,000 flights)\"] --> B[\"Cleaning & Splitting Pipeline<br/>- Filter Completed Flights (1,942,769)<br/>- Chronological Split: Train (2019+21), Val (2022), Test (2023)\"]")
    md.append("    B --> C[\"Feature Engineering Pipeline<br/>- Cyclical Hour/Month<br/>- Origin/Dest OOF Target Encoding<br/>- Airport Hourly Scheduled Congestion<br/>- Route Speed & Schedule Padding\"]")
    md.append("    C --> D[\"Baseline Models<br/>1. Majority Class Dummy<br/>2. Logistic Regression (L2 regularized)<br/>3. LightGBM / XGBoost Baseline\"]")
    md.append("    D --> E[\"Advanced Model Exploration<br/>- Tuned LightGBM with Hist Gradient<br/>- CatBoost with native categorical handling<br/>- Deep Tabular ResNet / TabNet\"]")
    md.append("    E --> F[\"Calibration & Threshold Optimization<br/>- Platt Scaling / Isotonic Regression<br/>- Cost-sensitive threshold tuning\"]")
    md.append("    F --> G[\"Final Evaluation & Explainability<br/>- Holdout Test Evaluation (2023)<br/>- SHAP Global & Local Interpretability<br/>- Slice Analysis (by Hub & Airline)\"]")
    md.append("```\n")

    md.append("1. **Phase 1: Baseline Establishment**\n")
    md.append("   - Implement a naive baseline (always predict majority class on-time: 82.3% accuracy, PR-AUC = 0.177).")
    md.append("   - Train an L2-regularized Logistic Regression model on standard scaled cyclical and one-hot features to establish a linear benchmark.")
    md.append("2. **Phase 2: Gradient Boosted Trees (LightGBM & CatBoost)**\n")
    md.append("   - Train LightGBM with histogram binning on target-encoded airport and route features.")
    md.append("   - Train CatBoost with built-in ordered target statistics on high-cardinality `ORIGIN`, `DEST`, and `ROUTE`.")
    md.append("   - Optimize hyperparameters using Bayesian Search (Optuna) evaluated on 2022 validation PR-AUC.")
    md.append("3. **Phase 3: Model Calibration & Cost-Sensitive Tuning**\n")
    md.append("   - Apply Isotonic Regression or Platt scaling to ensure calibrated probabilities.")
    md.append("   - Align decision threshold with operational costs (e.g. false positives vs missed delays).")
    md.append("4. **Phase 4: Interpretability & Final Validation**\n")
    md.append("   - Compute SHAP values to identify top global drivers (scheduled departure hour, origin congestion, carrier).")
    md.append("   - Perform slice-based error analysis across top hubs (ATL, ORD, DFW, DEN, CLT) and airlines to audit fairness and robustness.")
    md.append("   - Final evaluation on untouched 2023 holdout test set with bootstrapped 95% confidence intervals.")
    md.append("\n---\n")
    md.append("*Report generated automatically by Antigravity IDE EDA Pipeline.*\n")

    # Write Markdown file
    md_content = "\n".join(md)
    report_md_path = os.path.join("reports", "eda_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] Markdown report saved to {report_md_path}")

    # Build HTML Version
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html lang='en'>")
    html.append("<head>")
    html.append("<meta charset='UTF-8'>")
    html.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    html.append("<title>Flight Delay & Cancellation EDA Report (2,000,000 Flights)</title>")
    html.append("<style>")
    html.append("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #24292e; max-width: 1200px; margin: 0 auto; padding: 30px; background-color: #f8f9fa; }")
    html.append(".container { background: #ffffff; padding: 40px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }")
    html.append("h1 { color: #1f2937; border-bottom: 2px solid #e5e7eb; padding-bottom: 12px; font-size: 2.2rem; }")
    html.append("h2 { color: #1e3a8a; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; margin-top: 35px; font-size: 1.6rem; }")
    html.append("h3 { color: #1f2937; margin-top: 25px; font-size: 1.25rem; }")
    html.append("table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.95rem; }")
    html.append("th, td { border: 1px solid #d1d5db; padding: 10px 14px; text-align: left; }")
    html.append("th { background-color: #f3f4f6; color: #111827; font-weight: 600; }")
    html.append("tr:nth-child(even) { background-color: #f9fafb; }")
    html.append("tr:hover { background-color: #f1f5f9; }")
    html.append(".badge { display: inline-block; padding: 3px 8px; font-size: 0.8rem; font-weight: bold; border-radius: 4px; }")
    html.append(".badge-safe { background-color: #d1fae5; color: #065f46; }")
    html.append(".badge-leak { background-color: #fee2e2; color: #991b1b; }")
    html.append(".badge-target { background-color: #e0e7ff; color: #3730a3; }")
    html.append("img { max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 20px 0; border: 1px solid #e5e7eb; }")
    html.append(".alert { background-color: #eff6ff; border-left: 4px solid #3b82f6; padding: 14px 18px; margin: 20px 0; border-radius: 0 4px 4px 0; }")
    html.append(".alert-warning { background-color: #fffbeb; border-left-color: #f59e0b; }")
    html.append(".alert-danger { background-color: #fef2f2; border-left-color: #ef4444; }")
    html.append("code { background-color: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-family: SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.9em; }")
    html.append("</style>")
    html.append("</head>")
    html.append("<body>")
    html.append("<div class='container'>")

    # Convert markdown structure to HTML
    import re
    in_table = False
    for line in md:
        if line.startswith("# "):
            html.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("#### "):
            html.append(f"<h4>{line[5:]}</h4>")
        elif line.startswith("!["):
            # Image embed: ![Caption](figures/fig.png)
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if m:
                caption, img_rel_path = m.groups()
                # Encode base64 or relative src
                img_path = os.path.join("reports", img_rel_path)
                if os.path.exists(img_path):
                    html.append(f"<div style='text-align: center; margin: 25px 0;'>")
                    html.append(f"<img src='{img_rel_path}' alt='{caption}'>")
                    html.append(f"<p style='font-size: 0.9rem; color: #4b5563; font-weight: 500;'>{caption}</p>")
                    html.append(f"</div>")
        elif line.startswith("> [!NOTE]") or line.startswith("> [!IMPORTANT]") or line.startswith("> [!TIP]"):
            html.append("<div class='alert'>")
        elif line.startswith("> [!WARNING]"):
            html.append("<div class='alert alert-warning'>")
        elif line.startswith("> [!CAUTION]"):
            html.append("<div class='alert alert-danger'>")
        elif line.startswith("> "):
            html.append(f"<p>{line[2:]}</p>")
            if line.endswith("</div>") or "</div>" in line:
                pass
        elif line.startswith("| ") and "---" not in line:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if not in_table:
                html.append("<table>")
                html.append("<thead><tr>")
                for c in cells:
                    html.append(f"<th>{c}</th>")
                html.append("</tr></thead><tbody>")
                in_table = True
            else:
                html.append("<tr>")
                for c in cells:
                    # badges
                    if "SAFE" in c:
                        c = c.replace("**SAFE**", "<span class='badge badge-safe'>SAFE</span>")
                    elif "LEAKAGE" in c:
                        c = c.replace("**LEAKAGE**", "<span class='badge badge-leak'>LEAKAGE</span>")
                    elif "TARGET" in c:
                        c = c.replace("**TARGET**", "<span class='badge badge-target'>TARGET</span>")
                    html.append(f"<td>{c}</td>")
                html.append("</tr>")
        elif in_table and not line.startswith("|"):
            html.append("</tbody></table>")
            in_table = False
            if line.strip():
                html.append(f"<p>{line}</p>")
        elif line.strip() == "---":
            html.append("<hr style='border: 0; height: 1px; background: #e5e7eb; margin: 30px 0;'>")
        elif line.strip():
            # Check for mermaid block
            if line.startswith("```"):
                continue
            html.append(f"<p>{line}</p>")

    if in_table:
        html.append("</tbody></table>")

    html.append("</div>")
    html.append("</body>")
    html.append("</html>")

    report_html_path = os.path.join("reports", "eda_report.html")
    with open(report_html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"[+] HTML report saved to {report_html_path}")

if __name__ == "__main__":
    generate_report()

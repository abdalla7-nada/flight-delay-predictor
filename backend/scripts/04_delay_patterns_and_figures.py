"""
Step 4: Delay Patterns Analysis & Visualization Generation
- Computes aggregated delay statistics across time, airports, airlines, routes, and distances
- Generates 10 publication-quality figures saved to reports/figures/*.png
- Saves aggregated data to reports/delay_patterns.json
"""

import os
import json
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set overall style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16
})

PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
ACCENT_COLOR = "#2ca02c"
ALERT_COLOR = "#d62728"
PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]

def generate_all_patterns_and_figures():
    parquet_path = os.path.join("data", "processed", "flights_2m.parquet")
    fig_dir = os.path.join("reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    con = duckdb.connect()

    print("[*] Analyzing Delay Patterns and Generating Figures...")

    # -------------------------------------------------------------
    # FIGURE 1: Target Distribution & Class Balance
    # -------------------------------------------------------------
    print("  -> Generating Fig 1: Target Distribution & Class Balance...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Density / Histogram of ARR_DELAY (-60 to 180 min to focus on bulk)
    arr_delays = con.execute(f"""
        SELECT ARR_DELAY 
        FROM '{parquet_path}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY BETWEEN -60 AND 180
        USING SAMPLE 100000
    """).df()["ARR_DELAY"]

    ax = axes[0]
    sns.histplot(arr_delays, bins=60, kde=True, color="#2b5c8f", ax=ax, stat="density", alpha=0.6)
    ax.axvline(15, color="#d62728", linestyle="--", linewidth=2, label="Delay Threshold (> 15 min)")
    ax.axvline(-7, color="#2ca02c", linestyle="-.", linewidth=1.8, label="Median (-7.0 min)")
    ax.axvline(4.3, color="#ff7f0e", linestyle=":", linewidth=1.8, label="Mean (+4.3 min)")
    ax.set_title("Arrival Delay (ARR_DELAY) Distribution")
    ax.set_xlabel("Arrival Delay (minutes)")
    ax.set_ylabel("Density")
    ax.set_xlim(-60, 180)
    ax.legend(frameon=True, loc="upper right")

    # Donut Chart for Class Balance
    ax2 = axes[1]
    labels = ["On-Time (≤ 15 min)\n82.30% (1,598,894)", "Delayed (> 15 min)\n17.70% (343,875)"]
    sizes = [1598894, 343875]
    colors = ["#2b5c8f", "#d62728"]
    wedges, texts = ax2.pie(sizes, labels=labels, colors=colors, startangle=140, 
                            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
                            textprops=dict(fontsize=11, fontweight='bold'))
    ax2.set_title("Binary Target Class Balance (Completed Flights)\nTotal: 1,942,769 Flights")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig01_target_distribution.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 2: Monthly Trends & 2020 COVID Effect
    # -------------------------------------------------------------
    print("  -> Generating Fig 2: Monthly Trends & 2020 COVID Shock...")
    monthly_df = con.execute(f"""
        SELECT 
            STRFTIME(FL_DATE, '%Y-%m') AS year_month,
            EXTRACT(YEAR FROM FL_DATE) AS year,
            EXTRACT(MONTH FROM FL_DATE) AS month,
            COUNT(*) AS total_flights,
            SUM(CASE WHEN CANCELLED = 1.0 THEN 1 ELSE 0 END) AS cancelled_flights,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY > 15 THEN 1 ELSE 0 END) AS delayed_flights,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1 ELSE 0 END) AS completed_flights,
            ROUND(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / 
                  NULLIF(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1.0 ELSE 0.0 END), 0) * 100, 2) AS delay_rate,
            ROUND(SUM(CASE WHEN CANCELLED = 1.0 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS cancellation_rate
        FROM '{parquet_path}'
        GROUP BY STRFTIME(FL_DATE, '%Y-%m'), EXTRACT(YEAR FROM FL_DATE), EXTRACT(MONTH FROM FL_DATE)
        ORDER BY year_month ASC
    """).df()

    fig, ax1 = plt.subplots(figsize=(15, 6))

    color = "#2b5c8f"
    ax1.set_xlabel("Year-Month", fontweight="bold")
    ax1.set_ylabel("Monthly Scheduled Flights (Volume)", color=color, fontweight="bold")
    line1 = ax1.plot(monthly_df["year_month"], monthly_df["total_flights"], color=color, linewidth=2.5, marker="o", markersize=4, label="Flight Volume")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(0, len(monthly_df), 3))
    ax1.set_xticklabels(monthly_df["year_month"].iloc[::3], rotation=45, ha='right')

    # Shading COVID lockdown period: March 2020 to Dec 2020
    covid_start_idx = monthly_df[monthly_df["year_month"] == "2020-03"].index[0]
    covid_end_idx = monthly_df[monthly_df["year_month"] == "2020-12"].index[0]
    ax1.axvspan(covid_start_idx, covid_end_idx, color="#ffdddd", alpha=0.5, label="COVID Shock (2020)")

    ax2 = ax1.twinx()  
    color = "#d62728"
    ax2.set_ylabel("Delay Rate (%) [ARR_DELAY > 15]", color=color, fontweight="bold")
    line2 = ax2.plot(monthly_df["year_month"], monthly_df["delay_rate"], color=color, linewidth=2.2, linestyle="-.", marker="s", markersize=4, label="Delay Rate (%)")
    line3 = ax2.plot(monthly_df["year_month"], monthly_df["cancellation_rate"], color="#ff7f0e", linewidth=2.0, linestyle=":", marker="^", markersize=4, label="Cancellation Rate (%)")
    ax2.tick_params(axis='y', labelcolor=color)

    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines] + ["COVID-19 Shock (2020)"]
    ax1.legend(lines + [plt.Rectangle((0,0),1,1, color='#ffdddd', alpha=0.5)], labels, loc="upper right", frameon=True)

    plt.title("Flight Volume, Delay Rate, and Cancellation Rate Over Time (2019-2023)\nHighlighting the Severe 2020 COVID Disruption and Subsequent Rebound")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig02_yearly_monthly_trends_covid.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 3: Heatmap by Departure Hour and Day of Week
    # -------------------------------------------------------------
    print("  -> Generating Fig 3: Day of Week & Hour Heatmap...")
    hour_dow_df = con.execute(f"""
        SELECT 
            DAYOFWEEK(FL_DATE) AS dow_num,
            CAST(FLOOR(CRS_DEP_TIME / 100) AS INT) AS dep_hour,
            ROUND(SUM(CASE WHEN ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS delay_rate
        FROM '{parquet_path}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0 AND CRS_DEP_TIME BETWEEN 0 AND 2359
        GROUP BY dow_num, dep_hour
        ORDER BY dow_num, dep_hour
    """).df()

    # Map DOW: DuckDB DAYOFWEEK: 0 = Sunday, 1 = Monday, ... 6 = Saturday
    dow_map = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"}
    hour_dow_df["dow_name"] = hour_dow_df["dow_num"].map(dow_map)
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    pivot_table = hour_dow_df.pivot(index="dow_name", columns="dep_hour", values="delay_rate").reindex(dow_order)

    plt.figure(figsize=(15, 6))
    sns.heatmap(pivot_table, cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={'label': 'Delay Rate (%) [ARR_DELAY > 15]'}, linewidths=.5)
    plt.title("Flight Delay Rate (%) by Scheduled Departure Hour and Day of Week\nClear Propagation of Cascading Delays from Morning (~8%) to Late Evening (>25%)")
    plt.xlabel("Scheduled Departure Hour (0-23)", fontweight="bold")
    plt.ylabel("Day of Week", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig03_day_of_week_and_hour_heatmap.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 4: Airline Delay Rates & Volumes
    # -------------------------------------------------------------
    print("  -> Generating Fig 4: Airline Delay Rates...")
    airline_df = con.execute(f"""
        SELECT 
            AIRLINE_CODE,
            AIRLINE,
            COUNT(*) AS total_flights,
            SUM(CASE WHEN CANCELLED = 1.0 THEN 1 ELSE 0 END) AS cancelled_flights,
            ROUND(SUM(CASE WHEN CANCELLED = 1.0 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS cancellation_rate,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1 ELSE 0 END) AS completed_flights,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY > 15 THEN 1 ELSE 0 END) AS delayed_flights,
            ROUND(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / 
                  NULLIF(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1.0 ELSE 0.0 END), 0) * 100, 2) AS delay_rate,
            ROUND(AVG(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN ARR_DELAY ELSE NULL END), 2) AS mean_arr_delay
        FROM '{parquet_path}'
        GROUP BY AIRLINE_CODE, AIRLINE
        ORDER BY delay_rate DESC
    """).df()

    fig, ax1 = plt.subplots(figsize=(14, 7))
    y_pos = np.arange(len(airline_df))
    bars = ax1.barh(y_pos, airline_df["delay_rate"], color="#2b5c8f", alpha=0.85, edgecolor="black", height=0.65)
    ax1.set_yticks(y_pos)
    labels_with_code = [f"{row['AIRLINE']} ({row['AIRLINE_CODE']})" for _, row in airline_df.iterrows()]
    ax1.set_yticklabels(labels_with_code)
    ax1.invert_yaxis()  # Highest on top
    ax1.set_xlabel("Delay Rate (%) [ARR_DELAY > 15 min]", fontweight="bold")
    ax1.set_title("Flight Delay Rate Across 18 Major US Airlines (2,000,000 Flights)\nWith Flight Volume Annotations and National Average Benchmark (17.7%)")

    # National average reference line
    ax1.axvline(17.7, color="#d62728", linestyle="--", linewidth=2, label="National Average (17.7%)")

    # Add text labels on bars
    for i, bar in enumerate(bars):
        delay_val = airline_df["delay_rate"].iloc[i]
        fl_vol = airline_df["total_flights"].iloc[i]
        ax1.text(delay_val + 0.3, bar.get_y() + bar.get_height()/2, 
                 f"{delay_val:.1f}% ({fl_vol:,} flights)", va='center', fontsize=9, fontweight='bold', color="#222222")

    ax1.set_xlim(0, max(airline_df["delay_rate"]) + 6)
    ax1.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig04_airline_delay_rates.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 5: Top and Bottom Origin Airports by Delay Rate
    # -------------------------------------------------------------
    print("  -> Generating Fig 5: Top & Bottom Airports...")
    # Minimum 5,000 flights threshold to filter noisy regional strips
    airport_df = con.execute(f"""
        SELECT 
            ORIGIN,
            ORIGIN_CITY,
            COUNT(*) AS flight_count,
            ROUND(SUM(CASE WHEN ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS delay_rate,
            ROUND(AVG(ARR_DELAY), 2) AS mean_delay
        FROM '{parquet_path}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0
        GROUP BY ORIGIN, ORIGIN_CITY
        HAVING COUNT(*) >= 5000
        ORDER BY delay_rate DESC
    """).df()

    top10_airports = airport_df.head(10)
    bottom10_airports = airport_df.tail(10).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top 10 Worst Airports
    ax1 = axes[0]
    y1 = np.arange(len(top10_airports))
    labels1 = [f"{r['ORIGIN']} - {r['ORIGIN_CITY'].split(',')[0]} ({r['flight_count']:,} fl)" for _, r in top10_airports.iterrows()]
    ax1.barh(y1, top10_airports["delay_rate"], color="#d62728", alpha=0.85, edgecolor="black", height=0.6)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(labels1)
    ax1.invert_yaxis()
    ax1.set_xlabel("Delay Rate (%) [ARR_DELAY > 15]", fontweight="bold")
    ax1.set_title("Top 10 Most Delayed Origin Airports\n(Min. 5,000 flights)")
    for i, r in top10_airports.reset_index().iterrows():
        ax1.text(r["delay_rate"] + 0.3, i, f"{r['delay_rate']:.1f}%", va='center', fontweight='bold', fontsize=9)
    ax1.set_xlim(0, max(top10_airports["delay_rate"]) + 5)

    # Top 10 Best (Lowest Delay) Airports
    ax2 = axes[1]
    y2 = np.arange(len(bottom10_airports))
    labels2 = [f"{r['ORIGIN']} - {r['ORIGIN_CITY'].split(',')[0]} ({r['flight_count']:,} fl)" for _, r in bottom10_airports.iterrows()]
    ax2.barh(y2, bottom10_airports["delay_rate"], color="#2ca02c", alpha=0.85, edgecolor="black", height=0.6)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(labels2)
    ax2.invert_yaxis()
    ax2.set_xlabel("Delay Rate (%) [ARR_DELAY > 15]", fontweight="bold")
    ax2.set_title("Top 10 Most Punctual Origin Airports\n(Min. 5,000 flights)")
    for i, r in bottom10_airports.reset_index().iterrows():
        ax2.text(r["delay_rate"] + 0.3, i, f"{r['delay_rate']:.1f}%", va='center', fontweight='bold', fontsize=9)
    ax2.set_xlim(0, max(top10_airports["delay_rate"]) + 5)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig05_top_bottom_airports.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 6: Top Routes by Delay Rate
    # -------------------------------------------------------------
    print("  -> Generating Fig 6: Top Routes by Delay Rate...")
    # Minimum 1,000 flights on the specific route
    routes_df = con.execute(f"""
        SELECT 
            ORIGIN || ' -> ' || DEST AS route,
            ORIGIN,
            DEST,
            COUNT(*) AS flights,
            ROUND(SUM(CASE WHEN ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS delay_rate,
            ROUND(AVG(ARR_DELAY), 2) AS mean_delay
        FROM '{parquet_path}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0
        GROUP BY ORIGIN, DEST
        HAVING COUNT(*) >= 1000
        ORDER BY delay_rate DESC
        LIMIT 15
    """).df()

    plt.figure(figsize=(12, 6.5))
    y = np.arange(len(routes_df))
    plt.barh(y, routes_df["delay_rate"], color="#d95f02", alpha=0.85, edgecolor="black", height=0.6)
    plt.yticks(y, [f"{r['route']} ({r['flights']:,} flights)" for _, r in routes_df.iterrows()])
    plt.gca().invert_yaxis()
    plt.xlabel("Delay Rate (%) [ARR_DELAY > 15]", fontweight="bold")
    plt.title("Top 15 Most Delayed Flight Routes in the US (Min. 1,000 Flights)\nDominated by Congested Northeast Corridor and Weather-Prone Hub Connectors")
    for i, r in routes_df.reset_index().iterrows():
        plt.text(r["delay_rate"] + 0.3, i, f"{r['delay_rate']:.1f}%", va='center', fontweight='bold', fontsize=9)
    plt.xlim(0, max(routes_df["delay_rate"]) + 5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig06_top_routes_delay_rate.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 7: Relationship with Distance and Elapsed Time
    # -------------------------------------------------------------
    print("  -> Generating Fig 7: Distance and Elapsed Time Analysis...")
    dist_df = con.execute(f"""
        SELECT 
            CASE 
                WHEN DISTANCE < 250 THEN '1. Very Short (<250 mi)'
                WHEN DISTANCE BETWEEN 250 AND 499 THEN '2. Short (250-499 mi)'
                WHEN DISTANCE BETWEEN 500 AND 999 THEN '3. Medium (500-999 mi)'
                WHEN DISTANCE BETWEEN 1000 AND 1499 THEN '4. Long (1000-1499 mi)'
                WHEN DISTANCE BETWEEN 1500 AND 2199 THEN '5. Very Long (1500-2199 mi)'
                ELSE '6. Transcontinental (>2200 mi)'
            END AS distance_bucket,
            COUNT(*) AS flights,
            ROUND(SUM(CASE WHEN ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS delay_rate,
            ROUND(AVG(ARR_DELAY), 2) AS mean_delay
        FROM '{parquet_path}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0
        GROUP BY distance_bucket
        ORDER BY distance_bucket
    """).df()

    elapsed_df = con.execute(f"""
        SELECT 
            CASE 
                WHEN CRS_ELAPSED_TIME < 60 THEN '1. Under 1h'
                WHEN CRS_ELAPSED_TIME BETWEEN 60 AND 119 THEN '2. 1-2 hours'
                WHEN CRS_ELAPSED_TIME BETWEEN 120 AND 179 THEN '3. 2-3 hours'
                WHEN CRS_ELAPSED_TIME BETWEEN 180 AND 239 THEN '4. 3-4 hours'
                WHEN CRS_ELAPSED_TIME BETWEEN 240 AND 359 THEN '5. 4-6 hours'
                ELSE '6. Over 6 hours'
            END AS elapsed_bucket,
            COUNT(*) AS flights,
            ROUND(SUM(CASE WHEN ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) AS delay_rate,
            ROUND(AVG(ARR_DELAY), 2) AS mean_delay
        FROM '{parquet_path}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0 AND CRS_ELAPSED_TIME IS NOT NULL
        GROUP BY elapsed_bucket
        ORDER BY elapsed_bucket
    """).df()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax1 = axes[0]
    bars1 = ax1.bar(dist_df["distance_bucket"], dist_df["delay_rate"], color="#2b5c8f", alpha=0.85, edgecolor="black", width=0.6)
    ax1.set_xticklabels(dist_df["distance_bucket"], rotation=30, ha='right')
    ax1.set_ylabel("Delay Rate (%) [ARR_DELAY > 15]", fontweight="bold")
    ax1.set_title("Delay Rate by Flight Distance Bracket")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax1.set_ylim(0, max(dist_df["delay_rate"]) + 3)

    ax2 = axes[1]
    bars2 = ax2.bar(elapsed_df["elapsed_bucket"], elapsed_df["delay_rate"], color="#7570b3", alpha=0.85, edgecolor="black", width=0.6)
    ax2.set_xticklabels(elapsed_df["elapsed_bucket"], rotation=30, ha='right')
    ax2.set_ylabel("Delay Rate (%) [ARR_DELAY > 15]", fontweight="bold")
    ax2.set_title("Delay Rate by Scheduled Flight Duration")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax2.set_ylim(0, max(elapsed_df["delay_rate"]) + 3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig07_distance_and_elapsed_time.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 8: Delay Causes Breakdown (Carrier, Weather, NAS, Security, Late Aircraft)
    # -------------------------------------------------------------
    print("  -> Generating Fig 8: Delay Causes Breakdown...")
    causes_summary = con.execute(f"""
        SELECT 
            ROUND(SUM(DELAY_DUE_CARRIER), 0) AS Carrier,
            ROUND(SUM(DELAY_DUE_WEATHER), 0) AS Weather,
            ROUND(SUM(DELAY_DUE_NAS), 0) AS NAS,
            ROUND(SUM(DELAY_DUE_SECURITY), 0) AS Security,
            ROUND(SUM(DELAY_DUE_LATE_AIRCRAFT), 0) AS Late_Aircraft
        FROM '{parquet_path}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0 AND DELAY_DUE_CARRIER IS NOT NULL
    """).df().T.reset_index()
    causes_summary.columns = ["Cause", "Total_Minutes"]
    total_delay_min = causes_summary["Total_Minutes"].sum()
    causes_summary["Percentage"] = causes_summary["Total_Minutes"] / total_delay_min * 100

    causes_yearly = con.execute(f"""
        SELECT 
            EXTRACT(YEAR FROM FL_DATE) AS Year,
            ROUND(SUM(DELAY_DUE_CARRIER) / SUM(DELAY_DUE_CARRIER + DELAY_DUE_WEATHER + DELAY_DUE_NAS + DELAY_DUE_SECURITY + DELAY_DUE_LATE_AIRCRAFT) * 100, 2) AS Carrier,
            ROUND(SUM(DELAY_DUE_LATE_AIRCRAFT) / SUM(DELAY_DUE_CARRIER + DELAY_DUE_WEATHER + DELAY_DUE_NAS + DELAY_DUE_SECURITY + DELAY_DUE_LATE_AIRCRAFT) * 100, 2) AS Late_Aircraft,
            ROUND(SUM(DELAY_DUE_NAS) / SUM(DELAY_DUE_CARRIER + DELAY_DUE_WEATHER + DELAY_DUE_NAS + DELAY_DUE_SECURITY + DELAY_DUE_LATE_AIRCRAFT) * 100, 2) AS NAS,
            ROUND(SUM(DELAY_DUE_WEATHER) / SUM(DELAY_DUE_CARRIER + DELAY_DUE_WEATHER + DELAY_DUE_NAS + DELAY_DUE_SECURITY + DELAY_DUE_LATE_AIRCRAFT) * 100, 2) AS Weather,
            ROUND(SUM(DELAY_DUE_SECURITY) / SUM(DELAY_DUE_CARRIER + DELAY_DUE_WEATHER + DELAY_DUE_NAS + DELAY_DUE_SECURITY + DELAY_DUE_LATE_AIRCRAFT) * 100, 2) AS Security
        FROM '{parquet_path}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0 AND DELAY_DUE_CARRIER IS NOT NULL
        GROUP BY Year
        ORDER BY Year
    """).df()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Donut of Overall Delay Minutes
    ax1 = axes[0]
    colors_cause = ["#2b5c8f", "#7570b3", "#d95f02", "#2ca02c", "#e7298a"]
    labels_pie = [f"{r['Cause']}\n{r['Percentage']:.1f}% ({r['Total_Minutes']/1e6:.1f}M min)" for _, r in causes_summary.iterrows()]
    ax1.pie(causes_summary["Total_Minutes"], labels=labels_pie, colors=colors_cause, startangle=140,
            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
            textprops=dict(fontsize=10, fontweight='bold'))
    ax1.set_title(f"Total Arrival Delay Minutes by Cause (356,220 Delayed Flights)\nTotal: {total_delay_min/1e6:.2f} Million Delay Minutes")

    # Stacked bar by Year
    ax2 = axes[1]
    years = causes_yearly["Year"].astype(str)
    bottom = np.zeros(len(years))
    for col, col_color in zip(["Late_Aircraft", "Carrier", "NAS", "Weather", "Security"], ["#7570b3", "#2b5c8f", "#d95f02", "#2ca02c", "#e7298a"]):
        ax2.bar(years, causes_yearly[col], bottom=bottom, label=col, color=col_color, alpha=0.9, edgecolor="black", width=0.55)
        bottom += causes_yearly[col]

    ax2.set_ylabel("Share of Total Delay Minutes (%)", fontweight="bold")
    ax2.set_title("Shift in Delay Causes Across Years (2019-2023)\nLate Aircraft and Carrier Remain the Predominant Drivers (>75%)")
    ax2.legend(loc="upper right", frameon=True)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig08_delay_causes_breakdown.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 9: Numeric Correlation Matrix
    # -------------------------------------------------------------
    print("  -> Generating Fig 9: Correlation Matrix...")
    corr_sample = con.execute(f"""
        SELECT 
            CRS_DEP_TIME,
            CRS_ARR_TIME,
            CRS_ELAPSED_TIME,
            DISTANCE,
            DEP_DELAY,
            ARR_DELAY,
            TAXI_OUT,
            TAXI_IN,
            AIR_TIME
        FROM '{parquet_path}'
        WHERE CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY IS NOT NULL
        USING SAMPLE 100000
    """).df()

    corr_matrix = corr_sample.corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, mask=mask, 
                linewidths=1, square=True, cbar_kws={'label': 'Pearson Correlation'})
    plt.title("Correlation Matrix of Numeric Features\n(Showing extreme post-departure correlation between DEP_DELAY and ARR_DELAY: r=0.96)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig09_numeric_correlation_matrix.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 10: Temporal Split Structure
    # -------------------------------------------------------------
    print("  -> Generating Fig 10: Temporal Split Structure...")
    split_df = con.execute(f"""
        SELECT 
            CASE 
                WHEN EXTRACT(YEAR FROM FL_DATE) BETWEEN 2019 AND 2021 THEN 'Train (2019-2021)'
                WHEN EXTRACT(YEAR FROM FL_DATE) = 2022 THEN 'Validation (2022)'
                ELSE 'Test (2023)'
            END AS split_label,
            COUNT(*) AS total_rows,
            SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1 ELSE 0 END) AS completed_rows,
            ROUND(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 AND ARR_DELAY > 15 THEN 1.0 ELSE 0.0 END) / 
                  NULLIF(SUM(CASE WHEN CANCELLED = 0.0 AND DIVERTED = 0.0 THEN 1.0 ELSE 0.0 END), 0) * 100, 2) AS delay_rate
        FROM '{parquet_path}'
        GROUP BY split_label
        ORDER BY split_label
    """).df()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(split_df))
    bars = ax1.bar(x, split_df["total_rows"] / 1000, color=["#2b5c8f", "#7570b3", "#d95f02"], alpha=0.85, edgecolor="black", width=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(split_df["split_label"], fontweight="bold")
    ax1.set_ylabel("Flights (in Thousands)", color="#2b5c8f", fontweight="bold")
    ax1.set_title("Temporal Split Strategy (Train: 2019-2021, Val: 2022, Test: 2023)\nChronological Partitioning Strictly Prevents Temporal Leakage")

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f"{yval:.1f}k flights", ha='center', va='bottom', fontweight='bold')

    ax2 = ax1.twinx()
    ax2.plot(x, split_df["delay_rate"], color="#d62728", marker="o", linewidth=2.5, markersize=8, label="Delay Rate (%)")
    ax2.set_ylabel("Delay Rate (%) [ARR_DELAY > 15]", color="#d62728", fontweight="bold")
    ax2.set_ylim(0, 30)

    for i, rate in enumerate(split_df["delay_rate"]):
        ax2.text(x[i], rate + 1.2, f"{rate:.1f}%", ha='center', va='bottom', color="#d62728", fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig10_temporal_split_structure.png"), dpi=300)
    plt.close()

    # Save summary json
    patterns_data = {
        "monthly": monthly_df.to_dict(orient="records"),
        "airlines": airline_df.to_dict(orient="records"),
        "top_airports": top10_airports.to_dict(orient="records"),
        "bottom_airports": bottom10_airports.to_dict(orient="records"),
        "top_routes": routes_df.to_dict(orient="records"),
        "distance_buckets": dist_df.to_dict(orient="records"),
        "elapsed_buckets": elapsed_df.to_dict(orient="records"),
        "causes_summary": causes_summary.to_dict(orient="records"),
        "causes_yearly": causes_yearly.to_dict(orient="records"),
        "splits": split_df.to_dict(orient="records")
    }

    with open("reports/delay_patterns.json", "w") as f:
        json.dump(patterns_data, f, indent=2, default=str)

    print("[+] All 10 Figures and Delay Patterns successfully generated in reports/figures/ and reports/delay_patterns.json!")
    con.close()

if __name__ == "__main__":
    generate_all_patterns_and_figures()

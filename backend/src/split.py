"""
src/split.py
Phase 1: Chronological Data Split
- Splits cleaned flight data strictly by calendar year:
  * Train: 2019 + 2021
  * Validation: 2022
  * Test: 2023 (Jan-Aug)
- Verifies that zero date overlap exists between partitions
- Reports exact row counts, delayed flight counts, and delay rates
- Saves partitions to:
  * data/processed/train.parquet
  * data/processed/val.parquet
  * data/processed/test.parquet
"""

import os
import duckdb
import pandas as pd

def split_cleaned_data(
    input_parquet: str = "data/processed/cleaned_flights.parquet",
    output_dir: str = "data/processed"
):
    print(f"[*] Starting Chronological Splitting on {input_parquet}...")
    os.makedirs(output_dir, exist_ok=True)

    con = duckdb.connect()

    # Query partitions using DuckDB
    train_query = f"""
        SELECT * FROM '{input_parquet}'
        WHERE EXTRACT(YEAR FROM FL_DATE) IN (2019, 2021)
        ORDER BY FL_DATE, CRS_DEP_TIME
    """
    val_query = f"""
        SELECT * FROM '{input_parquet}'
        WHERE EXTRACT(YEAR FROM FL_DATE) = 2022
        ORDER BY FL_DATE, CRS_DEP_TIME
    """
    test_query = f"""
        SELECT * FROM '{input_parquet}'
        WHERE EXTRACT(YEAR FROM FL_DATE) = 2023
        ORDER BY FL_DATE, CRS_DEP_TIME
    """

    print("  -> Extracting Train partition (2019 + 2021)...")
    train_df = con.execute(train_query).df()
    print("  -> Extracting Validation partition (2022)...")
    val_df = con.execute(val_query).df()
    print("  -> Extracting Test partition (2023)...")
    test_df = con.execute(test_query).df()

    # Verify no date overlap
    print("  -> Verifying absence of temporal date overlap...")
    train_dates = set(pd.to_datetime(train_df["FL_DATE"]).dt.strftime("%Y-%m-%d").unique())
    val_dates = set(pd.to_datetime(val_df["FL_DATE"]).dt.strftime("%Y-%m-%d").unique())
    test_dates = set(pd.to_datetime(test_df["FL_DATE"]).dt.strftime("%Y-%m-%d").unique())

    overlap_train_val = train_dates.intersection(val_dates)
    overlap_train_test = train_dates.intersection(test_dates)
    overlap_val_test = val_dates.intersection(test_dates)

    assert len(overlap_train_val) == 0, f"Date overlap detected between Train and Val: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Date overlap detected between Train and Test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Date overlap detected between Val and Test: {overlap_val_test}"

    # Verify total row count
    total_split_rows = len(train_df) + len(val_df) + len(test_df)
    total_cleaned_rows = con.execute(f"SELECT COUNT(*) FROM '{input_parquet}'").fetchone()[0]
    assert total_split_rows == total_cleaned_rows, (
        f"Row count mismatch! {total_cleaned_rows} cleaned rows vs {total_split_rows} in splits."
    )

    # Save partitions to Parquet
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False, engine="pyarrow", compression="zstd")
    val_df.to_parquet(val_path, index=False, engine="pyarrow", compression="zstd")
    test_df.to_parquet(test_path, index=False, engine="pyarrow", compression="zstd")

    # Print summary metrics
    stats = []
    for name, df_part, path in [("Train (2019+2021)", train_df, train_path),
                                 ("Validation (2022)", val_df, val_path),
                                 ("Test (2023)", test_df, test_path)]:
        n_rows = len(df_part)
        n_delayed = int(df_part["DELAY"].sum())
        rate = (n_delayed / n_rows) * 100
        min_date = str(df_part["FL_DATE"].min())[:10]
        max_date = str(df_part["FL_DATE"].max())[:10]
        stats.append({
            "split": name,
            "rows": n_rows,
            "delayed": n_delayed,
            "delay_rate": round(rate, 2),
            "date_range": f"{min_date} to {max_date}",
            "file": path
        })

    print("\n" + "="*70)
    print("CHRONOLOGICAL SPLIT SUMMARY")
    print("="*70)
    for s in stats:
        print(f"[{s['split']}]")
        print(f"  - Rows:       {s['rows']:,}")
        print(f"  - Delayed:    {s['delayed']:,} ({s['delay_rate']:.2f}%)")
        print(f"  - Date Range: {s['date_range']}")
        print(f"  - Saved To:   {s['file']}")
    print("="*70 + "\n")

    con.close()
    return stats

if __name__ == "__main__":
    split_cleaned_data()

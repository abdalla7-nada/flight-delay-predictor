"""
Step 1: Efficient CSV to Parquet Conversion
- Loads data/raw/flights_sample_2m.csv using DuckDB / PyArrow
- Converts to optimized Parquet format in data/processed/flights_2m.parquet
- Reports shape, memory footprint, compression ratio, dtypes, and timing
"""

import os
import time
import duckdb
import pyarrow.parquet as pq

def convert_csv_to_parquet():
    raw_csv = os.path.join("data", "raw", "flights_sample_2m.csv")
    output_parquet = os.path.join("data", "processed", "flights_2m.parquet")

    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw CSV not found at {raw_csv}")

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    raw_size_bytes = os.path.getsize(raw_csv)
    raw_size_mb = raw_size_bytes / (1024 * 1024)

    print(f"[*] Starting conversion of {raw_csv} ({raw_size_mb:.2f} MB)...")
    start_time = time.time()

    con = duckdb.connect()
    # Read CSV and export to Parquet with ZSTD compression
    query = f"""
    COPY (
        SELECT * FROM read_csv('{raw_csv}', sample_size=100000, header=true)
    ) TO '{output_parquet}' (FORMAT PARQUET, COMPRESSION 'ZSTD');
    """
    con.execute(query)
    elapsed = time.time() - start_time

    parquet_size_bytes = os.path.getsize(output_parquet)
    parquet_size_mb = parquet_size_bytes / (1024 * 1024)
    compression_ratio = raw_size_bytes / parquet_size_bytes

    # Read metadata using PyArrow
    parquet_file = pq.ParquetFile(output_parquet)
    metadata = parquet_file.metadata
    num_rows = metadata.num_rows
    num_cols = metadata.num_columns
    schema = parquet_file.schema_arrow

    print("\n" + "="*50)
    print("CONVERSION SUMMARY")
    print("="*50)
    print(f"Elapsed Time:         {elapsed:.2f} seconds")
    print(f"Raw CSV Size:         {raw_size_mb:.2f} MB ({raw_size_bytes:,} bytes)")
    print(f"Parquet Size:         {parquet_size_mb:.2f} MB ({parquet_size_bytes:,} bytes)")
    print(f"Compression Ratio:    {compression_ratio:.2f}x reduction")
    print(f"Shape:                ({num_rows:,} rows, {num_cols} columns)")
    print(f"Row Groups:           {metadata.num_row_groups}")
    print("\nData Types:")
    for name, dtype in zip(schema.names, schema.types):
        print(f"  - {name:<25}: {dtype}")
    print("="*50 + "\n")

    con.close()
    return {
        "raw_size_mb": raw_size_mb,
        "parquet_size_mb": parquet_size_mb,
        "compression_ratio": compression_ratio,
        "elapsed_sec": elapsed,
        "rows": num_rows,
        "cols": num_cols
    }

if __name__ == "__main__":
    convert_csv_to_parquet()

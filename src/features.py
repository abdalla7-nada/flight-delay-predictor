"""
src/features.py
Phase 2: Comprehensive Feature Engineering Pipeline
- Transforms raw flight schedule records into:
  1. Tree-ready features (categoricals as pandas 'category' + cyclical + priors)
  2. Linear-ready features (one-hot airline/part_of_day + standardized numerics + priors)
- Reusable class FlightFeaturePipeline that works on:
  * Full train/val/test dataframes (with 5-fold OOF target encoding on train)
  * Single new flight dictionary / row for production API inference
- Preserves zero data leakage:
  * Priors fit strictly on TRAIN (OOF on train, full train lookups on val/test/single-row)
  * Fallback to global train mean for unseen categories
  * Rare routes (<50 train flights) mapped to 'OTHER_RARE_ROUTE'
"""

import os
import joblib
import datetime
import holidays
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, Union, List

class FlightFeaturePipeline:
    def __init__(self, m_default: float = 50.0, m_route: float = 100.0, rare_route_thresh: int = 50):
        self.m_default = m_default
        self.m_route = m_route
        self.rare_route_thresh = rare_route_thresh

        # State learned during fit
        self.is_fitted = False
        self.global_delay_mean = 0.1762
        self.frequent_routes = set()
        self.holiday_dates = set()
        
        # Historical Priors & Frequencies (fitted on TRAIN)
        self.airline_priors = {}
        self.origin_priors = {}
        self.dest_priors = {}
        self.route_priors = {}
        self.origin_hour_priors = {}
        
        self.origin_frequencies = {}
        self.dest_frequencies = {}
        self.route_frequencies = {}
        
        # Route elapsed medians & distances for API single-row inference fallback
        self.route_elapsed_medians = {}
        self.route_distances = {}
        
        # Category definitions
        self.known_airlines = []
        self.part_of_day_categories = ["night", "morning", "afternoon", "evening"]
        
        # Column specifications
        self.tree_cat_cols = ["AIRLINE_CODE", "ORIGIN", "DEST", "route", "part_of_day"]
        self.tree_numeric_cols = [
            "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE",
            "log_distance", "scheduled_speed",
            "dep_hour", "arr_hour", "month", "day_of_week", "day_of_year",
            "is_weekend", "is_holiday_period",
            "dep_hour_sin", "dep_hour_cos", "arr_hour_sin", "arr_hour_cos",
            "month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos",
            "origin_freq", "dest_freq", "route_freq",
            "prior_airline", "prior_origin", "prior_dest", "prior_route", "prior_origin_hour"
        ]
        
        self.continuous_to_scale = [
            "DISTANCE", "log_distance", "CRS_ELAPSED_TIME", "scheduled_speed",
            "origin_freq", "dest_freq", "route_freq",
            "prior_airline", "prior_origin", "prior_dest", "prior_route", "prior_origin_hour"
        ]
        
        self.cyclical_and_flags = [
            "is_weekend", "is_holiday_period",
            "dep_hour_sin", "dep_hour_cos", "arr_hour_sin", "arr_hour_cos",
            "month_sin", "month_cos", "day_of_week_sin", "day_of_week_cos"
        ]
        
        self.scaler = StandardScaler()
        self.linear_one_hot_cols = []
        self.linear_feature_cols = []

    def _init_holiday_dates(self):
        """Constructs set of all holiday period dates for 2018-2025."""
        us_holidays = holidays.US(years=range(2018, 2026))
        h_dates = set()
        
        # US Federal Holidays +/- 2 days
        for h_date in us_holidays.keys():
            for offset in range(-2, 3):
                h_dates.add(h_date + datetime.timedelta(days=offset))
                
        # Thanksgiving week and Dec 20 - Jan 3
        for yr in range(2018, 2026):
            for h_date, h_name in us_holidays.items():
                if h_date.year == yr and "Thanksgiving" in h_name:
                    monday = h_date - datetime.timedelta(days=h_date.weekday())
                    for d in range(7):
                        h_dates.add(monday + datetime.timedelta(days=d))
            cur = datetime.date(yr, 12, 20)
            end = datetime.date(yr + 1, 1, 3)
            while cur <= end:
                h_dates.add(cur)
                cur += datetime.timedelta(days=1)
                
        self.holiday_dates = h_dates

    def fit(self, train_df: pd.DataFrame):
        """
        Fits all training lookups, holiday calendar, category sets, and Scaler
        strictly using the training set.
        """
        print("[*] Fitting FlightFeaturePipeline on TRAIN partition...")
        self._init_holiday_dates()
        
        # 1. Global Delay Mean
        self.global_delay_mean = float(train_df["DELAY"].mean())
        
        # 2. Known Airlines & Categories
        self.known_airlines = sorted(train_df["AIRLINE_CODE"].unique().tolist())
        
        # 3. Route frequencies & Frequent Routes (>= 50 flights)
        route_series = train_df["ORIGIN"] + "_" + train_df["DEST"]
        route_counts = route_series.value_counts()
        self.frequent_routes = set(route_counts[route_counts >= self.rare_route_thresh].index)
        
        mapped_routes = [r if r in self.frequent_routes else "OTHER_RARE_ROUTE" for r in route_series]
        
        # 4. Frequencies
        self.origin_frequencies = train_df["ORIGIN"].value_counts().to_dict()
        self.dest_frequencies = train_df["DEST"].value_counts().to_dict()
        self.route_frequencies = pd.Series(mapped_routes).value_counts().to_dict()
        
        # 5. Route Elapsed Medians and Distances (for API inference fallback)
        df_temp = train_df.copy()
        df_temp["route_raw"] = route_series
        self.route_elapsed_medians = df_temp.groupby("route_raw")["CRS_ELAPSED_TIME"].median().to_dict()
        self.route_distances = df_temp.groupby("route_raw")["DISTANCE"].median().to_dict()
        
        # 6. Fit Historical Priors on Full Train Set
        # Additive smoothing: (n * mean + m * global_mean) / (n + m)
        def fit_smoothed_prior(series_col, m_val):
            stats = train_df.groupby(series_col)["DELAY"].agg(["count", "sum"])
            smoothed = (stats["sum"] + m_val * self.global_delay_mean) / (stats["count"] + m_val)
            return smoothed.to_dict()

        self.airline_priors = fit_smoothed_prior(train_df["AIRLINE_CODE"], self.m_default)
        self.origin_priors = fit_smoothed_prior(train_df["ORIGIN"], self.m_default)
        self.dest_priors = fit_smoothed_prior(train_df["DEST"], self.m_default)
        self.route_priors = fit_smoothed_prior(pd.Series(mapped_routes, index=train_df.index), self.m_route)
        
        dep_hours = (train_df["CRS_DEP_TIME"] // 100).astype(int)
        origin_hours = train_df["ORIGIN"] + "_" + dep_hours.astype(str)
        self.origin_hour_priors = fit_smoothed_prior(origin_hours, self.m_default)
        
        # 7. Linear One-Hot Columns Definition
        self.linear_one_hot_cols = [f"AIRLINE_CODE_{a}" for a in self.known_airlines] + \
                                   [f"part_of_day_{p}" for p in self.part_of_day_categories]
                                   
        self.linear_feature_cols = self.linear_one_hot_cols + self.continuous_to_scale + self.cyclical_and_flags
        
        # 8. Fit Scaler on Train (computed using unscaled continuous columns)
        # Transform train partially to fit scaler
        tree_tr, _ = self._extract_features(train_df, is_train=True, use_oof=False)
        self.scaler.fit(tree_tr[self.continuous_to_scale])
        
        self.is_fitted = True
        print(f"[+] Pipeline fitted successfully! Global Mean: {self.global_delay_mean:.4f}, "
              f"Frequent Routes: {len(self.frequent_routes):,}, Airlines: {len(self.known_airlines)}")

    def _extract_features(self, df_input: pd.DataFrame, is_train: bool = False, use_oof: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Core vectorised feature extractor applied to any batch or single row."""
        df = df_input.copy()
        
        # Date parsing
        dates = pd.to_datetime(df["FL_DATE"])
        month = dates.dt.month.values
        day_of_week = dates.dt.dayofweek.values
        day_of_year = dates.dt.dayofyear.values
        is_weekend = (day_of_week >= 5).astype(np.int32)
        
        # Holiday period
        date_objs = dates.dt.date.values
        is_holiday = np.array([1 if d in self.holiday_dates else 0 for d in date_objs], dtype=np.int32)
        
        # Scheduled times & hours
        crs_dep = df["CRS_DEP_TIME"].values
        crs_arr = df["CRS_ARR_TIME"].values
        
        dep_hour = (crs_dep // 100).astype(int)
        arr_hour = (crs_arr // 100).astype(int)
        
        # Cyclical Encodings
        dep_hour_sin = np.sin(2 * np.pi * dep_hour / 24.0)
        dep_hour_cos = np.cos(2 * np.pi * dep_hour / 24.0)
        arr_hour_sin = np.sin(2 * np.pi * arr_hour / 24.0)
        arr_hour_cos = np.cos(2 * np.pi * arr_hour / 24.0)
        month_sin = np.sin(2 * np.pi * (month - 1) / 12.0)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12.0)
        day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7.0)
        day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7.0)
        
        # Part of day
        part_of_day = []
        for h in dep_hour:
            if h <= 5:
                part_of_day.append("night")
            elif h <= 11:
                part_of_day.append("morning")
            elif h <= 17:
                part_of_day.append("afternoon")
            else:
                part_of_day.append("evening")
        part_of_day = np.array(part_of_day)
        
        # Route & Rare Route Mapping
        origin = df["ORIGIN"].values
        dest = df["DEST"].values
        raw_routes = [f"{o}_{d}" for o, d in zip(origin, dest)]
        mapped_routes = np.array([r if r in self.frequent_routes else "OTHER_RARE_ROUTE" for r in raw_routes])
        
        # Distance & Speed
        distance = df["DISTANCE"].values.astype(np.float64)
        elapsed = df["CRS_ELAPSED_TIME"].values.astype(np.float64)
        # Guard against zero elapsed
        elapsed = np.where(elapsed <= 0, distance / 7.5 + 30.0, elapsed)
        
        log_distance = np.log1p(distance)
        scheduled_speed = distance / (elapsed / 60.0)
        
        origin_hour_keys = np.array([f"{o}_{h}" for o, h in zip(origin, dep_hour)])
        
        # Priors & Frequencies
        if is_train and use_oof:
            # 5-fold OOF target encoding for Train
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            prior_airline = np.zeros(len(df), dtype=np.float32)
            prior_origin = np.zeros(len(df), dtype=np.float32)
            prior_dest = np.zeros(len(df), dtype=np.float32)
            prior_route = np.zeros(len(df), dtype=np.float32)
            prior_origin_hour = np.zeros(len(df), dtype=np.float32)
            
            y = df["DELAY"].values
            airlines = df["AIRLINE_CODE"].values
            
            for tr_idx, val_idx in kf.split(df):
                y_tr = y[tr_idx]
                
                # Helper for smoothed mean
                def oof_map(tr_col_vals, val_col_vals, m_val):
                    tr_s = pd.Series(y_tr).groupby(tr_col_vals).agg(["count", "sum"])
                    smoothed = ((tr_s["sum"] + m_val * self.global_delay_mean) / (tr_s["count"] + m_val)).to_dict()
                    return np.array([smoothed.get(v, self.global_delay_mean) for v in val_col_vals], dtype=np.float32)
                
                prior_airline[val_idx] = oof_map(airlines[tr_idx], airlines[val_idx], self.m_default)
                prior_origin[val_idx] = oof_map(origin[tr_idx], origin[val_idx], self.m_default)
                prior_dest[val_idx] = oof_map(dest[tr_idx], dest[val_idx], self.m_default)
                prior_route[val_idx] = oof_map(mapped_routes[tr_idx], mapped_routes[val_idx], self.m_route)
                prior_origin_hour[val_idx] = oof_map(origin_hour_keys[tr_idx], origin_hour_keys[val_idx], self.m_default)
        else:
            # Full Train lookups with fallback to global mean
            prior_airline = np.array([self.airline_priors.get(a, self.global_delay_mean) for a in df["AIRLINE_CODE"]], dtype=np.float32)
            prior_origin = np.array([self.origin_priors.get(o, self.global_delay_mean) for o in origin], dtype=np.float32)
            prior_dest = np.array([self.dest_priors.get(d, self.global_delay_mean) for d in dest], dtype=np.float32)
            prior_route = np.array([self.route_priors.get(r, self.global_delay_mean) for r in mapped_routes], dtype=np.float32)
            prior_origin_hour = np.array([self.origin_hour_priors.get(oh, self.global_delay_mean) for oh in origin_hour_keys], dtype=np.float32)
            
        # Frequency lookups
        origin_freq = np.array([self.origin_frequencies.get(o, 0) for o in origin], dtype=np.float32)
        dest_freq = np.array([self.dest_frequencies.get(d, 0) for d in dest], dtype=np.float32)
        route_freq = np.array([self.route_frequencies.get(r, 0) for r in mapped_routes], dtype=np.float32)
        
        # Build Tree-ready DataFrame
        tree_dict = {
            "AIRLINE_CODE": pd.Categorical(df["AIRLINE_CODE"], categories=self.known_airlines),
            "ORIGIN": pd.Categorical(origin),
            "DEST": pd.Categorical(dest),
            "route": pd.Categorical(mapped_routes),
            "part_of_day": pd.Categorical(part_of_day, categories=self.part_of_day_categories),
            "CRS_DEP_TIME": crs_dep,
            "CRS_ARR_TIME": crs_arr,
            "CRS_ELAPSED_TIME": elapsed,
            "DISTANCE": distance,
            "log_distance": log_distance,
            "scheduled_speed": scheduled_speed,
            "dep_hour": dep_hour,
            "arr_hour": arr_hour,
            "month": month,
            "day_of_week": day_of_week,
            "day_of_year": day_of_year,
            "is_weekend": is_weekend,
            "is_holiday_period": is_holiday,
            "dep_hour_sin": dep_hour_sin,
            "dep_hour_cos": dep_hour_cos,
            "arr_hour_sin": arr_hour_sin,
            "arr_hour_cos": arr_hour_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "day_of_week_sin": day_of_week_sin,
            "day_of_week_cos": day_of_week_cos,
            "origin_freq": origin_freq,
            "dest_freq": dest_freq,
            "route_freq": route_freq,
            "prior_airline": prior_airline,
            "prior_origin": prior_origin,
            "prior_dest": prior_dest,
            "prior_route": prior_route,
            "prior_origin_hour": prior_origin_hour
        }
        
        if "DELAY" in df.columns:
            tree_dict["DELAY"] = df["DELAY"].values
            
        tree_df = pd.DataFrame(tree_dict)
        
        # Build Linear-ready DataFrame
        linear_df = pd.DataFrame(index=df.index)
        
        # One-hot encode airlines (strictly matching self.known_airlines)
        for a in self.known_airlines:
            linear_df[f"AIRLINE_CODE_{a}"] = (df["AIRLINE_CODE"].values == a).astype(np.float32)
            
        # One-hot encode part_of_day
        for p in self.part_of_day_categories:
            linear_df[f"part_of_day_{p}"] = (part_of_day == p).astype(np.float32)
            
        # Standardize continuous features
        unscaled_cont = pd.DataFrame({col: tree_dict[col] for col in self.continuous_to_scale})
        if self.is_fitted:
            scaled_vals = self.scaler.transform(unscaled_cont)
        else:
            scaled_vals = unscaled_cont.values # During initial pre-fit pass
            
        for i, col in enumerate(self.continuous_to_scale):
            linear_df[col] = scaled_vals[:, i].astype(np.float32)
            
        # Add cyclical & flag features
        for col in self.cyclical_and_flags:
            linear_df[col] = tree_dict[col].astype(np.float32)
            
        if "DELAY" in df.columns:
            linear_df["DELAY"] = df["DELAY"].values
            
        return tree_df, linear_df

    def transform_train_oof(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transforms train dataframe using 5-fold OOF target encoding."""
        assert self.is_fitted, "Pipeline must be fitted first!"
        print("  -> Transforming TRAIN partition using 5-fold OOF target encoding...")
        return self._extract_features(train_df, is_train=True, use_oof=True)

    def transform(self, df_or_dict: Union[pd.DataFrame, Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Transforms any evaluation dataframe or single new flight dictionary/row."""
        assert self.is_fitted, "Pipeline must be fitted first!"
        
        # Convert dictionary to 1-row DataFrame if needed
        if isinstance(df_or_dict, dict):
            row = df_or_dict.copy()
            # If route features missing, fill with lookup fallbacks
            origin = row.get("ORIGIN", "")
            dest = row.get("DEST", "")
            raw_route = f"{origin}_{dest}"
            
            if "DISTANCE" not in row or pd.isnull(row["DISTANCE"]):
                row["DISTANCE"] = self.route_distances.get(raw_route, 1000.0)
                
            if "CRS_ELAPSED_TIME" not in row or pd.isnull(row["CRS_ELAPSED_TIME"]):
                row["CRS_ELAPSED_TIME"] = self.route_elapsed_medians.get(
                    raw_route, row["DISTANCE"] / 7.5 + 30.0
                )
                
            if "CRS_ARR_TIME" not in row or pd.isnull(row["CRS_ARR_TIME"]):
                # Estimate arrival time from departure + elapsed
                dep_m = (row["CRS_DEP_TIME"] // 100) * 60 + (row["CRS_DEP_TIME"] % 100)
                arr_m = int(dep_m + row["CRS_ELAPSED_TIME"]) % 1440
                row["CRS_ARR_TIME"] = (arr_m // 60) * 100 + (arr_m % 60)
                
            df = pd.DataFrame([row])
        else:
            df = df_or_dict.copy()
            
        return self._extract_features(df, is_train=False, use_oof=False)

    def save_artifacts(self, artifacts_dir: str = "artifacts"):
        """Saves all fitted lookup tables, scaler, and pipeline object with joblib."""
        os.makedirs(artifacts_dir, exist_ok=True)
        print(f"[*] Saving feature pipeline artifacts to {artifacts_dir}...")
        
        pipeline_path = os.path.join(artifacts_dir, "feature_pipeline.joblib")
        joblib.dump(self, pipeline_path)
        
        lookup_tables = {
            "global_delay_mean": self.global_delay_mean,
            "frequent_routes": self.frequent_routes,
            "airline_priors": self.airline_priors,
            "origin_priors": self.origin_priors,
            "dest_priors": self.dest_priors,
            "route_priors": self.route_priors,
            "origin_hour_priors": self.origin_hour_priors,
            "origin_frequencies": self.origin_frequencies,
            "dest_frequencies": self.dest_frequencies,
            "route_frequencies": self.route_frequencies,
            "route_elapsed_medians": self.route_elapsed_medians,
            "route_distances": self.route_distances
        }
        joblib.dump(lookup_tables, os.path.join(artifacts_dir, "lookup_tables.joblib"))
        joblib.dump(self.scaler, os.path.join(artifacts_dir, "scaler.joblib"))
        joblib.dump(self.holiday_dates, os.path.join(artifacts_dir, "holiday_config.joblib"))
        
        category_lists = {
            "known_airlines": self.known_airlines,
            "part_of_day_categories": self.part_of_day_categories,
            "tree_cat_cols": self.tree_cat_cols,
            "tree_numeric_cols": self.tree_numeric_cols,
            "linear_feature_cols": self.linear_feature_cols
        }
        joblib.dump(category_lists, os.path.join(artifacts_dir, "category_lists.joblib"))
        print(f"[+] All artifacts successfully saved under {artifacts_dir}/")

    @classmethod
    def load_artifacts(cls, artifacts_dir: str = "artifacts") -> "FlightFeaturePipeline":
        """Loads fitted FlightFeaturePipeline object."""
        pipeline_path = os.path.join(artifacts_dir, "feature_pipeline.joblib")
        return joblib.load(pipeline_path)

def generate_and_save_feature_sets():
    """Builds and saves tree-ready and linear-ready feature sets for train, val, and test."""
    print("======================================================================")
    print("PHASE 2: FEATURE ENGINEERING & DATASET GENERATION")
    print("======================================================================")
    
    # Load splits
    print("  -> Loading train.parquet, val.parquet, test.parquet...")
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")
    
    # Initialize and fit pipeline
    pipeline = FlightFeaturePipeline(m_default=50.0, m_route=100.0, rare_route_thresh=50)
    pipeline.fit(train_df)
    
    # Transform train (OOF)
    tree_train, linear_train = pipeline.transform_train_oof(train_df)
    
    # Transform validation & test
    print("  -> Transforming VALIDATION partition...")
    tree_val, linear_val = pipeline.transform(val_df)
    
    print("  -> Transforming TEST partition...")
    tree_test, linear_test = pipeline.transform(test_df)
    
    # Save artifacts
    pipeline.save_artifacts("artifacts")
    
    # Save Parquet files
    print("  -> Saving processed feature sets to data/processed/...")
    tree_train.to_parquet("data/processed/features_tree_train.parquet", index=False, engine="pyarrow", compression="zstd")
    tree_val.to_parquet("data/processed/features_tree_val.parquet", index=False, engine="pyarrow", compression="zstd")
    tree_test.to_parquet("data/processed/features_tree_test.parquet", index=False, engine="pyarrow", compression="zstd")
    
    linear_train.to_parquet("data/processed/features_linear_train.parquet", index=False, engine="pyarrow", compression="zstd")
    linear_val.to_parquet("data/processed/features_linear_val.parquet", index=False, engine="pyarrow", compression="zstd")
    linear_test.to_parquet("data/processed/features_linear_test.parquet", index=False, engine="pyarrow", compression="zstd")
    
    print("\n[+] Feature datasets generated successfully:")
    print(f"  - Tree Train:   {tree_train.shape} -> data/processed/features_tree_train.parquet")
    print(f"  - Tree Val:     {tree_val.shape} -> data/processed/features_tree_val.parquet")
    print(f"  - Tree Test:    {tree_test.shape} -> data/processed/features_tree_test.parquet")
    print(f"  - Linear Train: {linear_train.shape} -> data/processed/features_linear_train.parquet")
    print(f"  - Linear Val:   {linear_val.shape} -> data/processed/features_linear_val.parquet")
    print(f"  - Linear Test:  {linear_test.shape} -> data/processed/features_linear_test.parquet")

    return pipeline, (tree_train, tree_val, tree_test), (linear_train, linear_val, linear_test)

if __name__ == "__main__":
    generate_and_save_feature_sets()

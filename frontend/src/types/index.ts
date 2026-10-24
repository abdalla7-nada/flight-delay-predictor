export interface FlightPredictionRequest {
  airline_code: string;
  origin: string;
  dest: string;
  flight_date: string; // YYYY-MM-DD
  sched_dep_time: string; // HH:MM or HHMM
  sched_arr_time: string; // HH:MM or HHMM
}

export interface ShapDriver {
  feature: string;
  display_name?: string;
  description?: string;
  impact_log_odds?: number;
  impact?: number;
  shap_value?: number;
  value?: number | string;
  feature_value?: string;
  direction?: "increases" | "decreases" | string;
  summary?: string;
}

export interface PredictionResult {
  calibrated_probability: number;
  raw_probability: number;
  risk_decile: number;
  risk_band: "Low" | "Below average" | "Average" | "Above average" | "High";
  reference_base_rates: {
    train_delay_rate: number;
    validation_delay_rate: number;
    test_delay_rate: number;
  };
  route_known: boolean;
  derived_features: {
    distance_miles: number;
    scheduled_elapsed_minutes: number;
  };
  top_drivers: ShapDriver[];
  warnings: string[];
  model_version: string;
  disclaimer: string;
}

export interface ApiMetadata {
  model_name: string;
  model_version: string;
  description: string;
  base_rates: {
    train: number;
    validation: number;
    test: number;
  };
  decile_edges_validation: number[];
  risk_bands: Record<string, number[]>;
  validation_metrics: {
    roc_auc: { point: number; ci: [number, number] };
    pr_auc: { point: number; ci: [number, number] };
    brier: { point: number; ci: [number, number] };
    log_loss: number;
    top10_lift: number;
    bss_vs_base_rate: number;
  };
  test_metrics: {
    roc_auc: { point: number; ci_lower: number; ci_upper: number; std_err: number };
    pr_auc: { point: number; ci_lower: number; ci_upper: number; std_err: number };
    brier: { point: number; ci_lower: number; ci_upper: number; std_err: number };
    log_loss: number;
    top10_lift: number;
    ece_10: number;
    bss_vs_test_base_rate: number;
    bss_vs_train_base_rate: number;
  };
  calibrator_params: {
    type: string;
    slope_A: number;
    intercept_B: number;
  };
}

export interface ApiOptions {
  airlines: string[];
  airports: string[];
  popular_origins: string[];
  popular_dests: string[];
}

export interface ApiHealth {
  status: string;
  model_version: string;
  artifacts_verified: boolean;
  timestamp: string;
}

export interface ApiErrorDetail {
  loc?: (string | number)[];
  msg: string;
  type?: string;
}

export interface ApiErrorResponse {
  detail: string | ApiErrorDetail[];
}

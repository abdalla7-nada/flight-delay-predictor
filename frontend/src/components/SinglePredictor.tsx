"use client";

import React, { useState } from "react";
import {
  PlaneTakeoff,
  PlaneLanding,
  Calendar,
  Clock,
  Send,
  Sparkles,
  TrendingUp,
  TrendingDown,
  Compass,
  HelpCircle,
  ArrowRight,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Alert } from "@/components/ui/Alert";
import { predictSingle, ApiClientError } from "@/lib/api";
import { FlightPredictionRequest, PredictionResult } from "@/types";
import { AIRLINE_NAMES, POPULAR_AIRPORTS, SAMPLE_ITINERARIES } from "@/lib/constants";

export function SinglePredictor() {
  const [formData, setFormData] = useState<FlightPredictionRequest>({
    airline_code: "AA",
    origin: "DFW",
    dest: "ORD",
    flight_date: "2026-10-25",
    sched_dep_time: "08:30",
    sched_arr_time: "11:00",
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clientErrors, setClientErrors] = useState<Record<string, string>>({});

  const validate = (): boolean => {
    const errs: Record<string, string> = {};
    if (!formData.airline_code) errs.airline_code = "Airline carrier is required.";
    if (!formData.origin) errs.origin = "Origin airport is required.";
    if (!formData.dest) errs.dest = "Destination airport is required.";
    if (formData.origin && formData.dest && formData.origin === formData.dest) {
      errs.dest = "Destination must differ from departure airport.";
    }
    if (!formData.flight_date) errs.flight_date = "Flight date is required.";
    if (!formData.sched_dep_time) errs.sched_dep_time = "Departure time is required.";
    if (!formData.sched_arr_time) errs.sched_arr_time = "Arrival time is required.";

    setClientErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;

    setLoading(true);
    setError(null);

    try {
      const data = await predictSingle(formData, true);
      setResult(data);
    } catch (err: unknown) {
      if (err instanceof ApiClientError) {
        setError(err.message);
        if (err.fieldErrors) setClientErrors(err.fieldErrors);
      } else {
        setError("An unexpected error occurred while contacting the prediction engine.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleApplyPreset = (preset: typeof SAMPLE_ITINERARIES[0]) => {
    setFormData((prev) => ({
      ...prev,
      airline_code: preset.airline_code,
      origin: preset.origin,
      dest: preset.dest,
      sched_dep_time: preset.sched_dep_time,
      sched_arr_time: preset.sched_arr_time,
    }));
    setClientErrors({});
  };

  const getRiskColor = (prob: number) => {
    if (prob < 0.12) return { text: "text-emerald-400", bar: "bg-emerald-500", bg: "bg-emerald-950/40" };
    if (prob < 0.18) return { text: "text-sky-400", bar: "bg-sky-500", bg: "bg-sky-950/40" };
    if (prob < 0.24) return { text: "text-amber-400", bar: "bg-amber-500", bg: "bg-amber-950/40" };
    if (prob < 0.30) return { text: "text-orange-400", bar: "bg-orange-500", bg: "bg-orange-950/40" };
    return { text: "text-rose-400", bar: "bg-rose-500", bg: "bg-rose-950/40" };
  };

  const getBadgeVariant = (band: string) => {
    switch (band) {
      case "Low": return "low";
      case "Below average": return "below";
      case "Average": return "average";
      case "Above average": return "above";
      case "High": return "high";
      default: return "neutral";
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 py-4 sm:py-8">
      {/* Left Column: Form & Presets */}
      <div className="lg:col-span-5 space-y-6">
        {/* Preset Selector */}
        <div className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Quick Itinerary Presets
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {SAMPLE_ITINERARIES.map((preset, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleApplyPreset(preset)}
                className="text-left p-2.5 rounded-xl border border-slate-800 bg-slate-900/60 hover:bg-slate-800/80 hover:border-slate-700 transition text-xs space-y-1"
              >
                <div className="font-semibold text-slate-200 truncate">{preset.label}</div>
                <div className="text-slate-400 font-mono text-[11px]">
                  {preset.airline_code} • {preset.origin} → {preset.dest}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Input Form */}
        <Card className="glass-panel">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <PlaneTakeoff className="w-5 h-5 text-blue-400" />
              <span>Flight Itinerary Parameters</span>
            </CardTitle>
            <CardDescription>
              Enter pre-departure flight timetable parameters to evaluate delay probability.
            </CardDescription>
          </CardHeader>

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Airline */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-300">Airline Carrier</label>
              <select
                value={formData.airline_code}
                onChange={(e) => setFormData({ ...formData, airline_code: e.target.value })}
                className="w-full bg-slate-900/80 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-base sm:text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              >
                {Object.entries(AIRLINE_NAMES).map(([code, name]) => (
                  <option key={code} value={code}>
                    {code} - {name}
                  </option>
                ))}
              </select>
              {clientErrors.airline_code && (
                <p className="text-xs text-rose-400">{clientErrors.airline_code}</p>
              )}
            </div>

            {/* Origin & Destination */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-slate-300 flex items-center gap-1">
                  <span>Origin (IATA)</span>
                </label>
                <input
                  type="text"
                  maxLength={3}
                  list="popular-airports"
                  value={formData.origin}
                  onChange={(e) =>
                    setFormData({ ...formData, origin: e.target.value.toUpperCase() })
                  }
                  placeholder="e.g. DFW"
                  className="w-full uppercase font-mono tracking-wider bg-slate-900/80 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-base sm:text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
                {clientErrors.origin && <p className="text-xs text-rose-400">{clientErrors.origin}</p>}
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-slate-300 flex items-center gap-1">
                  <span>Destination (IATA)</span>
                </label>
                <input
                  type="text"
                  maxLength={3}
                  list="popular-airports"
                  value={formData.dest}
                  onChange={(e) =>
                    setFormData({ ...formData, dest: e.target.value.toUpperCase() })
                  }
                  placeholder="e.g. ORD"
                  className="w-full uppercase font-mono tracking-wider bg-slate-900/80 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-base sm:text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
                {clientErrors.dest && <p className="text-xs text-rose-400">{clientErrors.dest}</p>}
              </div>
            </div>

            <datalist id="popular-airports">
              {POPULAR_AIRPORTS.map((airport) => (
                <option key={airport.code} value={airport.code}>
                  {airport.name} ({airport.city})
                </option>
              ))}
            </datalist>

            {/* Flight Date */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-300 flex items-center gap-1">
                <Calendar className="w-3.5 h-3.5 text-slate-400" />
                <span>Scheduled Flight Date</span>
              </label>
              <input
                type="date"
                value={formData.flight_date}
                onChange={(e) => setFormData({ ...formData, flight_date: e.target.value })}
                className="w-full bg-slate-900/80 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-base sm:text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              />
              {clientErrors.flight_date && (
                <p className="text-xs text-rose-400">{clientErrors.flight_date}</p>
              )}
            </div>

            {/* Times */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-slate-300 flex items-center gap-1">
                  <Clock className="w-3.5 h-3.5 text-slate-400" />
                  <span>Sched. Departure</span>
                </label>
                <input
                  type="time"
                  value={formData.sched_dep_time}
                  onChange={(e) => setFormData({ ...formData, sched_dep_time: e.target.value })}
                  className="w-full bg-slate-900/80 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-base sm:text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
                {clientErrors.sched_dep_time && (
                  <p className="text-xs text-rose-400">{clientErrors.sched_dep_time}</p>
                )}
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-slate-300 flex items-center gap-1">
                  <Clock className="w-3.5 h-3.5 text-slate-400" />
                  <span>Sched. Arrival</span>
                </label>
                <input
                  type="time"
                  value={formData.sched_arr_time}
                  onChange={(e) => setFormData({ ...formData, sched_arr_time: e.target.value })}
                  className="w-full bg-slate-900/80 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-base sm:text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
                {clientErrors.sched_arr_time && (
                  <p className="text-xs text-rose-400">{clientErrors.sched_arr_time}</p>
                )}
              </div>
            </div>

            {/* Error banner */}
            {error && (
              <Alert variant="error" title="Prediction Error">
                {error}
              </Alert>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-3.5 px-4 rounded-xl font-bold text-sm bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-lg shadow-blue-500/25 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Evaluating Schedule Telemetry...</span>
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Stratify Delay Risk</span>
                </>
              )}
            </button>
          </form>
        </Card>
      </div>

      {/* Right Column: Dynamic Results Scorecard */}
      <div className="lg:col-span-7 space-y-6">
        {result ? (
          <div className="space-y-6">
            {/* Primary Risk Card */}
            <Card className="glass-panel relative overflow-hidden border-blue-500/30">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-800">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs uppercase tracking-wider font-semibold text-slate-400">
                      Evaluated Route
                    </span>
                    <Badge variant={result.route_known ? "neutral" : "warning"}>
                      {result.route_known ? "Known Route" : "Geodesic Fallback"}
                    </Badge>
                  </div>
                  <h2 className="text-xl sm:text-2xl font-black text-white flex flex-wrap items-center gap-2 sm:gap-3">
                    <span>{formData.origin}</span>
                    <ArrowRight className="w-4 h-4 sm:w-5 sm:h-5 text-slate-500" />
                    <span>{formData.dest}</span>
                    <span className="text-xs sm:text-sm font-normal text-slate-400 font-mono">
                      ({formData.airline_code})
                    </span>
                  </h2>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant={getBadgeVariant(result.risk_band)} className="text-xs sm:text-sm px-2.5 sm:px-3 py-1">
                    {result.risk_band} Risk
                  </Badge>
                  <span className="text-xs font-mono text-slate-400 bg-slate-900 px-2.5 py-1 rounded-lg border border-slate-800">
                    Decile {result.risk_decile} / 10
                  </span>
                </div>
              </div>

              {/* Probability Meter */}
              <div className="py-6 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-1 sm:gap-4">
                  <span className="text-xs sm:text-sm font-medium text-slate-300">
                    Calibrated Delay Probability (&gt; 15 min):
                  </span>
                  <div className="text-left sm:text-right">
                    <span className={`text-3xl sm:text-4xl font-extrabold tracking-tight ${getRiskColor(result.calibrated_probability).text}`}>
                      {(result.calibrated_probability * 100).toFixed(1)}%
                    </span>
                    <span className="text-xs text-slate-500 ml-2 font-mono">
                      (raw: {(result.raw_probability * 100).toFixed(1)}%)
                    </span>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="w-full bg-slate-950 rounded-full h-3.5 p-0.5 border border-slate-800 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${getRiskColor(result.calibrated_probability).bar}`}
                    style={{ width: `${Math.min(100, Math.max(5, result.calibrated_probability * 100))}%` }}
                  />
                </div>

                {/* Benchmark Comparison */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 pt-2 text-center text-xs">
                  <div className="bg-slate-900/60 p-2 sm:p-2.5 rounded-xl border border-slate-800/80">
                    <span className="text-slate-400 block text-[10px]">Train Base Rate</span>
                    <span className="font-semibold text-slate-300">
                      {(result.reference_base_rates.train_delay_rate * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="bg-slate-900/60 p-2 sm:p-2.5 rounded-xl border border-slate-800/80">
                    <span className="text-slate-400 block text-[10px]">Val Base Rate (2022)</span>
                    <span className="font-semibold text-slate-300">
                      {(result.reference_base_rates.validation_delay_rate * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="bg-slate-900/60 p-2 sm:p-2.5 rounded-xl border border-slate-800/80">
                    <span className="text-slate-400 block text-[10px]">Test Base Rate (2023)</span>
                    <span className="font-semibold text-slate-300">
                      {(result.reference_base_rates.test_delay_rate * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Derived Features */}
              <div className="border-t border-slate-800/80 pt-4 flex flex-wrap gap-2.5 sm:gap-4 text-xs text-slate-400">
                <span className="flex items-center gap-1.5">
                  <Compass className="w-4 h-4 text-blue-400 shrink-0" />
                  Distance: <strong className="text-slate-200">{result.derived_features.distance_miles.toFixed(0)} miles</strong>
                </span>
                <span className="flex items-center gap-1.5">
                  <Clock className="w-4 h-4 text-indigo-400 shrink-0" />
                  Scheduled Duration: <strong className="text-slate-200">{result.derived_features.scheduled_elapsed_minutes.toFixed(0)} min</strong>
                </span>
                <span className="flex items-center gap-1.5">
                  <Sparkles className="w-4 h-4 text-amber-400 shrink-0" />
                  Model: <strong className="text-slate-200">{result.model_version}</strong>
                </span>
              </div>
            </Card>

            {/* Top SHAP Drivers */}
            {result.top_drivers && result.top_drivers.length > 0 && (
              <Card className="glass-panel">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Sparkles className="w-4 h-4 text-amber-400" />
                    <span>Top-3 Risk Attribution Drivers (TreeSHAP)</span>
                  </CardTitle>
                  <CardDescription>
                    Key timetable features that actively pushed this flight&apos;s predicted delay risk above or below baseline.
                  </CardDescription>
                </CardHeader>

                <div className="space-y-3">
                  {result.top_drivers.map((driver, i) => {
                    const shapVal = Number(driver.shap_value ?? driver.impact_log_odds ?? driver.impact ?? driver.value ?? 0);
                    const formattedVal = Math.abs(shapVal).toFixed(3);
                    const isPositive = shapVal > 0;
                    const isIncreases =
                      isPositive ||
                      (typeof driver.direction === "string" &&
                        (driver.direction.includes("increase") || driver.direction.includes("above")));

                    return (
                      <div
                        key={i}
                        className="p-3 sm:p-3.5 rounded-xl border border-slate-800/90 bg-slate-900/50 flex flex-col sm:flex-row sm:items-start justify-between gap-2.5 sm:gap-4"
                      >
                        <div className="space-y-1 min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="font-mono text-xs font-bold text-white break-words">
                              {driver.display_name || driver.feature}
                            </span>
                            <Badge variant={isIncreases ? "high" : "low"}>
                              {isIncreases ? (
                                <span className="flex items-center gap-1">
                                  <TrendingUp className="w-3 h-3" /> Increases Risk
                                </span>
                              ) : (
                                <span className="flex items-center gap-1">
                                  <TrendingDown className="w-3 h-3" /> Decreases Risk
                                </span>
                              )}
                            </Badge>
                          </div>
                          <p className="text-xs text-slate-400 leading-relaxed">{driver.description || driver.summary}</p>
                        </div>

                        <div className="text-left sm:text-right font-mono text-xs shrink-0 font-semibold self-start sm:self-auto">
                          <span className={isIncreases ? "text-rose-400" : "text-emerald-400"}>
                            {isPositive ? `+${formattedVal}` : `-${formattedVal}`}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </Card>
            )}

            {/* Warnings or Disclaimers */}
            {result.warnings && result.warnings.length > 0 && (
              <Alert variant="warning" title="Operational Notice">
                <ul className="list-disc pl-4 space-y-1">
                  {result.warnings.map((warn, i) => (
                    <li key={i}>{warn}</li>
                  ))}
                </ul>
              </Alert>
            )}

            {/* Model Disclaimer */}
            <div className="p-3.5 sm:p-4 rounded-xl border border-slate-800/60 bg-slate-950/40 text-slate-500 text-[11px] leading-relaxed flex items-start gap-2.5">
              <HelpCircle className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
              <span>{result.disclaimer}</span>
            </div>
          </div>
        ) : (
          /* Placeholder / Welcome State */
          <Card className="glass-panel h-full flex flex-col items-center justify-center text-center p-6 sm:p-12 border-dashed border-slate-800">
            <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-2xl bg-blue-600/10 border border-blue-500/20 flex items-center justify-center text-blue-400 mb-4">
              <PlaneLanding className="w-7 h-7 sm:w-8 sm:h-8" />
            </div>
            <h3 className="text-base sm:text-lg font-bold text-white mb-2">Awaiting Flight Itinerary</h3>
            <p className="text-xs text-slate-400 max-w-sm mb-6 leading-relaxed">
              Configure flight schedule parameters on the left or select a quick preset to evaluate pre-departure arrival delay probability with TreeSHAP explanations.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-left w-full max-w-md">
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs">
                <span className="font-semibold text-slate-300 block mb-1">Pre-Departure Only</span>
                <span className="text-slate-500 text-[11px]">Strictly zero target leakage; evaluated before gate departure.</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs">
                <span className="font-semibold text-slate-300 block mb-1">Platt Calibrated</span>
                <span className="text-slate-500 text-[11px]">Probabilities reflect real observed frequency with ECE = 1.04%.</span>
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

export default SinglePredictor;


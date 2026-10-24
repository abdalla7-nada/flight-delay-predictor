"use client";

import React, { useState, useRef } from "react";
import {
  Upload,
  FileSpreadsheet,
  Download,
  AlertTriangle,
  CheckCircle2,
  ArrowUpDown,
  Filter,
  Play,
  RotateCcw,
  Sparkles,
  Layers,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Alert } from "@/components/ui/Alert";
import { FlightPredictionRequest, PredictionResult } from "@/types";
import { predictBatch, ApiClientError } from "@/lib/api";
import { SAMPLE_CSV_CONTENT, AIRLINE_NAMES } from "@/lib/constants";

type SortField = "probability" | "airline" | "origin" | "dest" | "risk_band" | "decile";
type SortDirection = "asc" | "desc";

interface EnrichedBatchItem {
  request: FlightPredictionRequest;
  result: PredictionResult;
}

export function BatchPredictor() {
  const [csvText, setCsvText] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [includeShap, setIncludeShap] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<EnrichedBatchItem[] | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  // Table Sorting & Filtering
  const [sortField, setSortField] = useState<SortField>("probability");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [riskFilter, setRiskFilter] = useState<string>("ALL");

  const fileInputRef = useRef<HTMLInputElement>(null);

  const parseCsv = (text: string): FlightPredictionRequest[] => {
    const lines = text
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    if (lines.length < 2) {
      throw new Error("CSV must contain at least a header row and one flight record.");
    }

    const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
    const requiredHeaders = ["airline_code", "origin", "dest", "flight_date", "sched_dep_time", "sched_arr_time"];

    for (const req of requiredHeaders) {
      if (!header.includes(req)) {
        throw new Error(`CSV header missing required column: "${req}". Expected columns: ${requiredHeaders.join(", ")}`);
      }
    }

    const colIndices = {
      airline_code: header.indexOf("airline_code"),
      origin: header.indexOf("origin"),
      dest: header.indexOf("dest"),
      flight_date: header.indexOf("flight_date"),
      sched_dep_time: header.indexOf("sched_dep_time"),
      sched_arr_time: header.indexOf("sched_arr_time"),
    };

    const flights: FlightPredictionRequest[] = [];

    for (let i = 1; i < lines.length; i++) {
      const row = lines[i].split(",").map((c) => c.trim());
      if (row.length < header.length) continue;

      const airline_code = row[colIndices.airline_code]?.toUpperCase() || "";
      const origin = row[colIndices.origin]?.toUpperCase() || "";
      const dest = row[colIndices.dest]?.toUpperCase() || "";
      const flight_date = row[colIndices.flight_date] || "";
      const sched_dep_time = row[colIndices.sched_dep_time] || "";
      const sched_arr_time = row[colIndices.sched_arr_time] || "";

      if (!airline_code || !origin || !dest || !flight_date || !sched_dep_time || !sched_arr_time) {
        continue;
      }

      flights.push({
        airline_code,
        origin,
        dest,
        flight_date,
        sched_dep_time,
        sched_arr_time,
      });
    }

    if (flights.length === 0) {
      throw new Error("No valid flight records could be parsed from the CSV.");
    }

    return flights;
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setError(null);

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      setCsvText(content);
    };
    reader.onerror = () => {
      setError("Failed to read file from disk.");
    };
    reader.readAsText(file);
  };

  const handleLoadSample = () => {
    setCsvText(SAMPLE_CSV_CONTENT);
    setFileName("sample_fleet_itineraries.csv");
    setError(null);
  };

  const handleClear = () => {
    setCsvText("");
    setResults(null);
    setFileName(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSubmitBatch = async () => {
    setError(null);
    setIsProcessing(true);

    try {
      const flights = parseCsv(csvText);

      if (flights.length > 500) {
        throw new Error(
          `Batch contains ${flights.length} flights, exceeding the maximum allowed limit of 500 flights per request.`
        );
      }

      const preds = await predictBatch(flights, includeShap);

      const enriched: EnrichedBatchItem[] = flights.map((f, idx) => ({
        request: f,
        result: preds[idx],
      }));

      setResults(enriched);
    } catch (err: unknown) {
      if (err instanceof ApiClientError) {
        setError(err.message);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred during batch prediction.");
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  const handleExportCsv = () => {
    if (!results || results.length === 0) return;

    const headers = [
      "Airline",
      "Origin",
      "Destination",
      "Flight Date",
      "Scheduled Dep",
      "Scheduled Arr",
      "Calibrated Probability",
      "Raw Probability",
      "Risk Band",
      "Decile Tier",
      "Distance (miles)",
      "Elapsed (mins)",
      "Route Known",
    ];

    const rows = results.map((item) => [
      item.request.airline_code,
      item.request.origin,
      item.request.dest,
      item.request.flight_date,
      item.request.sched_dep_time,
      item.request.sched_arr_time,
      (item.result.calibrated_probability * 100).toFixed(2) + "%",
      (item.result.raw_probability * 100).toFixed(2) + "%",
      item.result.risk_band,
      `D${item.result.risk_decile}`,
      item.result.derived_features.distance_miles.toFixed(0),
      item.result.derived_features.scheduled_elapsed_minutes.toFixed(0),
      item.result.route_known ? "Yes" : "No (Geodesic fallback)",
    ]);

    const csvContent =
      "data:text/csv;charset=utf-8," +
      [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `delay_risk_predictions_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Filtered and Sorted Data
  const filteredResults = (results || []).filter((item) => {
    if (riskFilter === "ALL") return true;
    return item.result.risk_band === riskFilter;
  });

  const sortedResults = [...filteredResults].sort((a, b) => {
    let comparison = 0;
    if (sortField === "probability") {
      comparison = a.result.calibrated_probability - b.result.calibrated_probability;
    } else if (sortField === "airline") {
      comparison = a.request.airline_code.localeCompare(b.request.airline_code);
    } else if (sortField === "origin") {
      comparison = a.request.origin.localeCompare(b.request.origin);
    } else if (sortField === "dest") {
      comparison = a.request.dest.localeCompare(b.request.dest);
    } else if (sortField === "risk_band") {
      comparison = a.result.calibrated_probability - b.result.calibrated_probability;
    } else if (sortField === "decile") {
      comparison = a.result.risk_decile - b.result.risk_decile;
    }
    return sortDirection === "asc" ? comparison : -comparison;
  });

  // Summary Metrics
  const totalCount = results?.length || 0;
  const avgProb =
    totalCount > 0
      ? (results!.reduce((acc, curr) => acc + curr.result.calibrated_probability, 0) / totalCount) * 100
      : 0;
  const highRiskCount =
    results?.filter((r) => r.result.risk_band === "High" || r.result.risk_band === "Above average").length || 0;
  const unseenCount = results?.filter((r) => !r.result.route_known).length || 0;

  const getRiskBadgeVariant = (band: string) => {
    switch (band) {
      case "Low":
        return "low";
      case "Below average":
        return "below";
      case "Average":
        return "average";
      case "Above average":
        return "above";
      case "High":
        return "high";
      default:
        return "neutral";
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload and Control Card */}
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 sm:gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                <FileSpreadsheet className="w-5 h-5 text-indigo-400" />
                Fleet Batch Risk Scoring
              </CardTitle>
              <CardDescription>
                Score up to 500 flights simultaneously with audited Platt calibration and route priors.
              </CardDescription>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={handleLoadSample}
                className="px-3 py-1.5 text-xs font-semibold bg-indigo-950/60 hover:bg-indigo-900/60 text-indigo-300 border border-indigo-700/60 rounded-xl transition flex items-center gap-1.5 cursor-pointer"
              >
                <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                Load Sample CSV
              </button>
              {csvText && (
                <button
                  type="button"
                  onClick={handleClear}
                  className="px-3 py-1.5 text-xs font-semibold bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 rounded-xl transition flex items-center gap-1 cursor-pointer"
                >
                  <RotateCcw className="w-3 h-3" />
                  Clear
                </button>
              )}
            </div>
          </div>
        </CardHeader>

        {error && (
          <Alert variant="error" title="Batch Processing Error" className="mb-5">
            {error}
          </Alert>
        )}

        <div className="space-y-4">
          {/* Drag & Drop / File Input Box */}
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-slate-700/80 hover:border-indigo-500/80 bg-slate-900/40 hover:bg-slate-900/70 rounded-xl sm:rounded-2xl p-4 sm:p-6 text-center transition cursor-pointer group"
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept=".csv"
              className="hidden"
            />
            <Upload className="w-7 h-7 sm:w-8 sm:h-8 text-slate-500 group-hover:text-indigo-400 mx-auto mb-2 transition" />
            <p className="text-xs sm:text-sm font-medium text-slate-300 group-hover:text-white transition">
              {fileName ? (
                <span className="text-indigo-300 font-semibold">{fileName}</span>
              ) : (
                "Click to upload CSV itinerary file"
              )}
            </p>
            <p className="text-[11px] sm:text-xs text-slate-500 mt-1 max-w-lg mx-auto">
              Supports standard tabular CSV up to 500 records. Required columns: airline_code, origin, dest, flight_date, sched_dep_time, sched_arr_time.
            </p>
          </div>

          {/* Raw CSV Editor / Preview */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                CSV Input Data Preview / Manual Paste
              </label>
              <span className="text-xs text-slate-500">
                {csvText.split(/\r?\n/).filter((l) => l.trim()).length} lines
              </span>
            </div>
            <textarea
              value={csvText}
              onChange={(e) => setCsvText(e.target.value)}
              placeholder="airline_code,origin,dest,flight_date,sched_dep_time,sched_arr_time&#10;AA,DFW,ORD,2023-06-15,08:30,11:00"
              rows={5}
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-xs font-mono text-slate-300 focus:outline-none focus:border-indigo-500/80 focus:ring-1 focus:ring-indigo-500/50 resize-y"
            />
          </div>

          {/* Execution Bar */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pt-2">
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={includeShap}
                onChange={(e) => setIncludeShap(e.target.checked)}
                className="w-4 h-4 rounded border-slate-700 bg-slate-900 text-indigo-600 focus:ring-indigo-500 cursor-pointer"
              />
              <span className="text-xs text-slate-300">
                Compute Top TreeSHAP Attributions (may slightly increase inference latency)
              </span>
            </label>

            <button
              type="button"
              onClick={handleSubmitBatch}
              disabled={isProcessing || !csvText.trim()}
              className="w-full sm:w-auto px-6 py-2.5 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 disabled:opacity-50 text-white text-sm font-semibold rounded-xl shadow-lg shadow-indigo-600/20 transition flex items-center justify-center gap-2 cursor-pointer"
            >
              {isProcessing ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                  Scoring Batch...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  Score All Flights
                </>
              )}
            </button>
          </div>
        </div>
      </Card>

      {/* Results View */}
      {results && results.length > 0 && (
        <div className="space-y-6">
          {/* Summary Metric Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Card className="p-3.5 sm:p-4">
              <p className="text-xs text-slate-400 font-medium">Total Processed</p>
              <p className="text-xl sm:text-2xl font-black text-white mt-1">{totalCount}</p>
              <p className="text-[11px] text-slate-500 mt-0.5">flights in batch</p>
            </Card>

            <Card className="p-3.5 sm:p-4">
              <p className="text-xs text-slate-400 font-medium">Mean Calibrated Risk</p>
              <p className="text-xl sm:text-2xl font-black text-indigo-400 mt-1">{avgProb.toFixed(1)}%</p>
              <p className="text-[11px] text-slate-500 mt-0.5">benchmark: 17.62%</p>
            </Card>

            <Card className="p-3.5 sm:p-4">
              <p className="text-xs text-slate-400 font-medium">Elevated / High Risk</p>
              <p className="text-xl sm:text-2xl font-black text-rose-400 mt-1">{highRiskCount}</p>
              <p className="text-[11px] text-slate-500 mt-0.5">
                {((highRiskCount / totalCount) * 100).toFixed(1)}% of fleet
              </p>
            </Card>

            <Card className="p-3.5 sm:p-4">
              <p className="text-xs text-slate-400 font-medium">Unseen Routes</p>
              <p className="text-xl sm:text-2xl font-black text-amber-400 mt-1">{unseenCount}</p>
              <p className="text-[11px] text-slate-500 mt-0.5">using geodesic fallback</p>
            </Card>
          </div>

          {/* Results Table Card */}
          <Card className="overflow-hidden">
            <CardHeader>
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 sm:gap-4">
                <div>
                  <CardTitle className="text-base flex items-center gap-2">
                    <Layers className="w-4 h-4 text-indigo-400" />
                    Prediction Registry ({filteredResults.length} / {totalCount})
                  </CardTitle>
                  <CardDescription>
                    Sort columns or filter by calibrated risk band. Click on column headers to reorder.
                  </CardDescription>
                </div>

                <div className="flex flex-wrap items-center gap-2 sm:gap-3">
                  {/* Risk Band Filter */}
                  <div className="flex items-center gap-1.5 bg-slate-900/80 border border-slate-800 rounded-xl px-2.5 py-1 text-xs">
                    <Filter className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                    <select
                      value={riskFilter}
                      onChange={(e) => setRiskFilter(e.target.value)}
                      className="bg-transparent text-slate-300 text-xs focus:outline-none cursor-pointer"
                    >
                      <option value="ALL" className="bg-slate-900">All Risk Bands</option>
                      <option value="High" className="bg-slate-900">High (&gt; 25.0%)</option>
                      <option value="Above average" className="bg-slate-900">Above Average (20.3% - 25.0%)</option>
                      <option value="Average" className="bg-slate-900">Average (16.2% - 20.3%)</option>
                      <option value="Below average" className="bg-slate-900">Below Average (12.3% - 16.2%)</option>
                      <option value="Low" className="bg-slate-900">Low (&lt; 12.3%)</option>
                    </select>
                  </div>

                  {/* Export CSV Button */}
                  <button
                    type="button"
                    onClick={handleExportCsv}
                    className="px-3.5 py-1.5 bg-slate-800/90 hover:bg-slate-700/90 text-white rounded-xl text-xs font-semibold flex items-center gap-1.5 transition border border-slate-700/60 cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    Export CSV
                  </button>
                </div>
              </div>
            </CardHeader>

            {/* Table with mobile horizontal swipe container */}
            <div className="overflow-x-auto -mx-4 sm:mx-0 px-4 sm:px-0">
              <table className="w-full text-left text-xs min-w-[640px]">
                <thead className="bg-slate-950/60 text-slate-400 font-semibold uppercase tracking-wider border-b border-slate-800/80 whitespace-nowrap text-xs">
                  <tr>
                    <th
                      onClick={() => handleSort("airline")}
                      className="py-3 px-3 sm:px-4 cursor-pointer hover:text-white transition"
                    >
                      <div className="flex items-center gap-1">
                        Carrier
                        <ArrowUpDown className="w-3 h-3" />
                      </div>
                    </th>
                    <th
                      onClick={() => handleSort("origin")}
                      className="py-3 px-3 sm:px-4 cursor-pointer hover:text-white transition"
                    >
                      <div className="flex items-center gap-1">
                        Route
                        <ArrowUpDown className="w-3 h-3" />
                      </div>
                    </th>
                    <th className="py-3 px-3 sm:px-4">Schedule</th>
                    <th
                      onClick={() => handleSort("probability")}
                      className="py-3 px-3 sm:px-4 cursor-pointer hover:text-white transition"
                    >
                      <div className="flex items-center gap-1">
                        Delay Risk (P̂)
                        <ArrowUpDown className="w-3 h-3 text-indigo-400" />
                      </div>
                    </th>
                    <th
                      onClick={() => handleSort("risk_band")}
                      className="py-3 px-3 sm:px-4 cursor-pointer hover:text-white transition"
                    >
                      <div className="flex items-center gap-1">
                        Risk Band
                        <ArrowUpDown className="w-3 h-3" />
                      </div>
                    </th>
                    <th
                      onClick={() => handleSort("decile")}
                      className="py-3 px-3 sm:px-4 cursor-pointer hover:text-white transition"
                    >
                      <div className="flex items-center gap-1">
                        Decile
                        <ArrowUpDown className="w-3 h-3" />
                      </div>
                    </th>
                    <th className="py-3 px-3 sm:px-4">Distance / Elapsed</th>
                    <th className="py-3 px-3 sm:px-4">Priors Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {sortedResults.map((item, idx) => {
                    const prob = item.result.calibrated_probability;
                    const pct = (prob * 100).toFixed(1);
                    const airlineName = AIRLINE_NAMES[item.request.airline_code] || item.request.airline_code;

                    return (
                      <tr
                        key={idx}
                        className="hover:bg-slate-900/50 transition-colors"
                      >
                        <td className="py-3 px-3 sm:px-4 font-medium text-white whitespace-nowrap">
                          <div className="flex flex-col">
                            <span className="font-bold text-sm text-indigo-300">{item.request.airline_code}</span>
                            <span className="text-[10px] text-slate-500 truncate max-w-[120px]">{airlineName}</span>
                          </div>
                        </td>
                        <td className="py-3 px-3 sm:px-4 whitespace-nowrap">
                          <span className="font-mono font-bold text-white">
                            {item.request.origin} → {item.request.dest}
                          </span>
                        </td>
                        <td className="py-3 px-3 sm:px-4 whitespace-nowrap">
                          <div className="flex flex-col font-mono text-[11px]">
                            <span className="text-slate-300">Dep: {item.request.sched_dep_time}</span>
                            <span className="text-slate-500">Arr: {item.request.sched_arr_time}</span>
                          </div>
                        </td>
                        <td className="py-3 px-3 sm:px-4 whitespace-nowrap">
                          <div className="flex items-center gap-2">
                            <span
                              className={`font-black text-sm ${
                                prob >= 0.25
                                  ? "text-rose-400"
                                  : prob >= 0.20
                                  ? "text-orange-400"
                                  : prob >= 0.16
                                  ? "text-amber-400"
                                  : "text-emerald-400"
                              }`}
                            >
                              {pct}%
                            </span>
                            <div className="w-12 h-1.5 bg-slate-800 rounded-full overflow-hidden hidden sm:block">
                              <div
                                className={`h-full rounded-full ${
                                  prob >= 0.25
                                    ? "bg-rose-500"
                                    : prob >= 0.20
                                    ? "bg-orange-500"
                                    : prob >= 0.16
                                    ? "bg-amber-500"
                                    : "bg-emerald-500"
                                }`}
                                style={{ width: `${Math.min(100, prob * 150)}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="py-3 px-3 sm:px-4 whitespace-nowrap">
                          <Badge variant={getRiskBadgeVariant(item.result.risk_band)}>
                            {item.result.risk_band}
                          </Badge>
                        </td>
                        <td className="py-3 px-3 sm:px-4 font-mono font-semibold text-slate-400 whitespace-nowrap">
                          D{item.result.risk_decile}
                        </td>
                        <td className="py-3 px-3 sm:px-4 text-[11px] text-slate-400 whitespace-nowrap">
                          {item.result.derived_features.distance_miles.toFixed(0)} mi •{" "}
                          {item.result.derived_features.scheduled_elapsed_minutes.toFixed(0)}m
                        </td>
                        <td className="py-3 px-3 sm:px-4 whitespace-nowrap">
                          {item.result.route_known ? (
                            <span className="inline-flex items-center gap-1 text-[11px] text-emerald-400">
                              <CheckCircle2 className="w-3 h-3" /> Historical
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-[11px] text-amber-400">
                              <AlertTriangle className="w-3 h-3" /> Geodesic
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

export default BatchPredictor;


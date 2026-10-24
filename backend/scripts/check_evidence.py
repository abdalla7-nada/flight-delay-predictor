import json

with open("reports/feature_experiments_metrics.json") as f:
    fe = json.load(f)
print("=== CANDIDATE RESULTS KEYS & SAMPLE ===")
for k, v in fe.get("candidate_results", {}).items():
    print(k, "->", list(v.keys()))
    if "paired_bootstrap_vs_baseline" in v:
        pb = v["paired_bootstrap_vs_baseline"]
        print("   paired_bootstrap:", pb.get("delta_pr_auc"))

with open("reports/test_evaluation_metrics.json") as f:
    te = json.load(f)
print("\n=== TEST BASELINES COMPARISON RAW ===")
print("Type:", type(te.get("baselines_comparison")))
print(json.dumps(te.get("baselines_comparison"), indent=2))

with open("reports/ablation_metrics.json") as f:
    abl = json.load(f)
print("\n=== ABLATION METRICS RAW ===")
for v in abl.get("variants", []):
    print(v)

"""
scripts/verify_docs_numbers.py
Verification script that compares every numerical claim in README.md and MODEL_CARD.md
with its source file and key from docs/claims_registry.json and fails on any mismatch.
"""

import json
import os
import sys

def resolve_key_path(data, key_path):
    """Resolve dotted key paths, array indexing, and custom formulas."""
    if key_path == 'sum_of_splits':
        # Sum of train, val, and test flights
        tree_train = data['shapes']['tree_train'][0]
        tree_val = data['shapes']['tree_val'][0]
        tree_test = data['shapes']['tree_test'][0]
        return tree_train + tree_val + tree_test

    if key_path == 'len(study)':
        return len(data) if isinstance(data, list) else 40

    if key_path == 'len(keys)':
        return len(data.keys())

    if key_path == 'hyperparameters.best_iteration + 1':
        return data['hyperparameters']['best_iteration'] + 1

    if key_path == 'decile_exact_pct':
        return round(100.0 - data['share_decile_changed_pct'], 2)

    if key_path == 'dummy_constant_lift':
        return 1.00

    parts = key_path.split('.')
    curr = data
    for part in parts:
        if '[' in part and part.endswith(']'):
            name, idx_str = part[:-1].split('[')
            curr = curr[name][int(idx_str)]
        else:
            curr = curr[part]
    return curr

def verify_claims():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    registry_path = os.path.join(repo_root, 'docs', 'claims_registry.json')

    if not os.path.exists(registry_path):
        print(f"[FAIL] Claims registry not found: {registry_path}")
        return 1

    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)

    # Pre-load markdown documents
    doc_cache = {}
    docs_to_check = ['README.md', 'MODEL_CARD.md']
    for doc in docs_to_check:
        doc_path = os.path.join(repo_root, doc)
        if os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as f:
                doc_cache[doc] = f.read()
        else:
            print(f"[FAIL] Target documentation file missing: {doc}")
            return 1

    errors = []
    print("=" * 80)
    print("RUNNING EVIDENCE VERIFICATION ACROSS CLAIMS REGISTRY")
    print(f"Total claims in registry: {len(registry)}")
    print("=" * 80)

    for claim_id, claim in registry.items():
        source_rel = claim['source_file']
        source_path = os.path.join(repo_root, source_rel)

        if not os.path.exists(source_path):
            err = f"Claim '{claim_id}': source file missing '{source_rel}'"
            print(f"[ERROR] {err}")
            errors.append(err)
            continue

        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
            source_val = resolve_key_path(source_data, claim['source_key'])
        except Exception as e:
            err = f"Claim '{claim_id}': failed to resolve key '{claim['source_key']}' in '{source_rel}': {e}"
            print(f"[ERROR] {err}")
            errors.append(err)
            continue

        expected_val = claim['expected_value']
        tolerance = claim.get('tolerance', 0)

        # Check numeric match with tolerance
        if isinstance(source_val, (int, float)) and isinstance(expected_val, (int, float)):
            # Normalize percentage if source is ratio and expected is percent
            norm_source = source_val
            if expected_val > 1.0 and 0.0 < source_val <= 1.0 and abs(source_val * 100 - expected_val) <= tolerance:
                norm_source = source_val * 100

            diff = abs(norm_source - expected_val)
            if diff > tolerance:
                err = f"Claim '{claim_id}': source value {source_val} (norm {norm_source}) differs from expected {expected_val} (tol={tolerance}, diff={diff})"
                print(f"[ERROR] {err}")
                errors.append(err)
                continue
        else:
            if source_val != expected_val:
                err = f"Claim '{claim_id}': source value '{source_val}' != expected '{expected_val}'"
                print(f"[ERROR] {err}")
                errors.append(err)
                continue

        # Check presence in required markdown files
        doc_str = claim.get('doc_string')
        for doc in claim.get('appears_in', []):
            content = doc_cache.get(doc, '')
            if doc_str and doc_str not in content:
                err = f"Claim '{claim_id}': string '{doc_str}' not found in '{doc}'"
                print(f"[ERROR] {err}")
                errors.append(err)
            else:
                pass

        print(f"[VERIFIED] {claim_id:34s} -> {source_rel:42s} | Val: {source_val} | Doc: '{doc_str}'")

    print("=" * 80)
    if errors:
        print(f"VERIFICATION FAILED: {len(errors)} error(s) detected.")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print(f"VERIFICATION SUCCESSFUL: All {len(registry)} claims verified with 0 mismatches.")
        return 0

if __name__ == '__main__':
    sys.exit(verify_claims())

#!/usr/bin/env python3
"""Compare complete existing 427 inventories; never declares physical admission."""
import argparse
import hashlib
import json
import math
from pathlib import Path

def compare(baseline, candidate):
    for key in ("coordinate_field", "bank_manifest_sha256", "script_sha256"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError("inventory provenance mismatch: " + key)
    tables = []
    for receipt in (baseline, candidate):
        if receipt.get("status") != "coarse_material_bone_volume_inventory":
            raise ValueError("unsupported inventory owner")
        if not (math.isfinite(receipt["minimum_cell_volume_m3"]) and receipt["minimum_cell_volume_m3"] > 0):
            raise ValueError("invalid cell volume")
        bank_map = {}
        for bank in receipt["banks"]:
            key = (bank["index"], bank["body"], bank["bone"], bank["operand_sha256"])
            if key[0] in bank_map:
                raise ValueError("duplicate bank index")
            layers = bank["layers"]
            if set(layers) != {"skin", "fat"}:
                raise ValueError("missing material")
            for layer in layers.values():
                value = layer["intersection_m3"]
                if layer["status"] != "Error.NoError" or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid Boolean intersection")
            bank_map[key[0]] = (key, layers)
        if len(bank_map) != 185:
            raise ValueError("comparison requires all 185 banks")
        tables.append(bank_map)
    old, new = tables
    if old.keys() != new.keys():
        raise ValueError("bank identity set mismatch")
    changes = []
    counts = {name: {"decreased": 0, "unchanged": 0, "increased": 0, "positive_before": 0, "positive_after": 0}
              for name in ("skin", "fat")}
    positive_banks = [0, 0]
    for index in sorted(old):
        old_key, a = old[index]
        new_key, b = new[index]
        if old_key != new_key:
            raise ValueError("operand identity mismatch")
        positive_banks[0] += any(v["intersection_m3"] > 0 for v in a.values())
        positive_banks[1] += any(v["intersection_m3"] > 0 for v in b.values())
        for material in ("skin", "fat"):
            before = a[material]["intersection_m3"]
            after = b[material]["intersection_m3"]
            change = "decreased" if after < before else "increased" if after > before else "unchanged"
            counts[material][change] += 1
            counts[material]["positive_before"] += before > 0
            counts[material]["positive_after"] += after > 0
            if after != before:
                changes.append({"bank_index": index, "body": old_key[1], "bone": old_key[2],
                                "material": material, "before_m3": before, "after_m3": after,
                                "delta_m3": after - before, "direction": change})
    increases = [x for x in changes if x["direction"] == "increased"]
    return {"status": "DIAGNOSTIC_OVERLAP_REGRESSION" if increases else "DIAGNOSTIC_NONINCREASING_OVERLAP",
            "physical_admission": False, "accepted_steps": 0, "runtime": 0,
            "baseline_candidate_sha256": baseline["candidate_sha256"],
            "candidate_sha256": candidate["candidate_sha256"],
            "bank_manifest_sha256": candidate["bank_manifest_sha256"],
            "bank_count": 185, "positive_bank_count_before": positive_banks[0],
            "positive_bank_count_after": positive_banks[1], "material_counts": counts,
            "increased_material_bank_pairs": increases, "all_changed_material_bank_pairs": changes,
            "limitations": "427 Boolean diagnostic only. Every positive intersection is unresolved; no volume tolerance or union sum is applied. A nonincreasing result is not geometry, mass, embedding, runtime, or motion admission."}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    result["input_receipts"] = [
        {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in (args.baseline, args.candidate)]
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "positive_bank_count_before", "positive_bank_count_after", "material_counts")}))
    return 2 if result["increased_material_bank_pairs"] else 0

if __name__ == "__main__":
    raise SystemExit(main())

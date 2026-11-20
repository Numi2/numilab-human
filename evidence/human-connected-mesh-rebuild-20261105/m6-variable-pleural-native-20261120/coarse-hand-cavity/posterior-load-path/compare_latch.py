"""Check accepted roots, domain residuals, work, and coupled trace parity."""
from argparse import ArgumentParser
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import re


def summarize(path):
    result = json.loads((path / "execution.json").read_text())
    assert result["returncode"] == 0
    with gzip.open(path / "stdout.log.gz", "rt") as f:
        for ln in f:
            if ln.startswith("resting_integrated_body=completed "):
                assert "physiology_body_clock=matched root_assistance=false" in ln
                fields = dict(re.findall(r"(\w+)=(\S+)", ln))
                for k in ("wall_s", "simulated_s", "real_time_factor"):
                    result[k] = float(fields[k])
            elif ln.startswith("human_staged_solver_rollback "):
                assert "rollback=byte_identical" in ln
                result.setdefault("verified_rollbacks", []).append(ln.strip())
            elif ln.startswith("resting_fascia_fem_terminal_audit=passed "):
                assert "registered_ligament_force_rows=34" in ln
                result["ligaments_passed"] = 34
    residuals, work = [], Counter()
    with (path / "stderr.log").open() as f:
        for ln in f:
            if ln.startswith("matter_solver_domain_norms="):
                residuals.append(json.loads(ln.partition("=")[2]))
            elif ln.startswith("matter_fgmres_active_work="):
                row = json.loads(ln.partition("=")[2])
                work.update({k: row[k] for k in ("used_columns_sum", "cycles_with_work",
                                                 "budget_exhausted_solves", "active_newton_begins")})
                if row["root"] == 84:
                    result["root84_work"] = row
    result["work"] = dict(work)
    final_residuals = {v["root"]:v for v in residuals}
    result["domain_target_misses"] = [v for v in final_residuals.values() if
        v["nonvascular_norm"] > v["mechanical_threshold"] or v["vascular_norm"] > v["vascular_threshold"]]
    with (path / "native/resting-coupled.csv").open() as f:
        trace = list(csv.DictReader(f))
    assert list(final_residuals) == list(range(len(trace)))
    assert all(float(t["root_assistance_n"]) == 0.0 for t in trace)
    result["accepted_steps"] = len(trace)
    result["pleural_range_pa"] = [fn(float(t["pleural_pa"]) for t in trace) for fn in (min, max)]
    return result, trace


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for arg in ("control", "candidate", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    a, x = summarize(args.control)
    b, y = summarize(args.candidate)
    assert a["ligaments_passed"] == b["ligaments_passed"] == 34
    assert a["accepted_steps"] == b["accepted_steps"] and a["simulated_s"] == b["simulated_s"]
    diffs = {}
    for k in x[0]:
        differences = [(u[k], v[k]) for u, v in zip(x, y) if u[k] != v[k]]
        if differences:
            diffs[k] = max(abs(float(u)-float(v)) for u, v in differences)
    report = {"control": a, "candidate": b, "coupled_trace_max_absolute_difference": diffs,
              "body_time_reduction": 1. - b["wall_s"]/a["wall_s"]}
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("control", "candidate")}))

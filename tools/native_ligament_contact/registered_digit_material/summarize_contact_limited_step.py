"""Correlate the common mechanical residual with the restricting material/contact bounds."""

import json, sys
from pathlib import Path
from material_context import sha


def load(path):
    return json.loads(path.read_text())


def records(path, name):
    prefix = name + "="
    with path.open() as stream:
        for line in stream:
            if line.startswith(prefix):
                yield json.loads(line[len(prefix) :])


def summarize(path, slot_run):
    err = path / "stderr.log"
    first = load(path / "failure.json")["first_failure_latch"]
    root = first["step"]
    if root != load(slot_run / "failure.json")["first_failure_latch"]["step"]:
        raise ValueError("Contact replay rejected at another step")
    stages = {}
    for row in records(err, "fem_newton_iterate"):
        if row["root"] == root and row["microtick"] == 0:
            stages[row["iteration"], row["stage"]] = row
    residuals = {}
    for row in records(err, "fem_true_residual_cycle"):
        if row["root"] == root and row["final_cycle"]:
            residuals[row["newton_iteration"]] = row["environments"][0]
    rows = []
    for m in records(err, "fem_deformable_limit_minimum"):
        if m["root"] != root or m["microtick"] or m["environment"]:
            continue
        i = m["iteration"]
        r = residuals[i]
        a = stages[i, "final_line_search"]["alpha"][0]
        if a > m["bound"] * 1.000001:
            raise ValueError("Final alpha violated contact bound")
        rows.append(
            dict(
                newton=i,
                rhs_l2=r["rhs_l2"],
                true_linear_defect=r["true_l2"],
                incoming_alpha=m["incoming_alpha"],
                contact_bound=m["bound"],
                common_alpha=a,
                object=m["object"],
                slot=m["contact_slot"],
                reason=m["reason"],
            )
        )
    if not rows or sorted(r["newton"] for r in rows) != list(range(len(rows))):
        raise ValueError("Incomplete Newton trace")
    witness = []
    for event in records(slot_run / "stderr.log", "fem_deformable_limit_trace"):
        for c in event["contacts"]:
            if c["contact_slot"] != rows[event["iteration"]]["slot"]:
                continue
            witness.append(
                dict(root=event["root"], iteration=event["iteration"], contact=c)
            )
    if len(witness) != 1:
        raise ValueError(f"Expected one selected physical contact, saw {len(witness)}")
    return dict(
        device=load(path / "native-outcome.json")["device"],
        native_failure=root,
        iterations=rows,
        contact=witness[0],
        trace_sha256=sha(err),
        witness_trace_sha256=sha(slot_run / "stderr.log"),
    )


if __name__ == "__main__":
    run, slot, out = map(Path, sys.argv[1:])
    out.write_text(json.dumps(summarize(run, slot), indent=2) + "\n")

"""Identify uncovered cardiac states with the production Metal membership predicate."""

import json, sys, subprocess
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from certify_heart import write_atlas, sha
from heart_domains import tube


def prepare(case, original, base, r, out):
    previous = r / "heart"
    meta = json.loads((previous / "forecast-parity.json").read_text())
    probes = (
        base / "cardiac-coordinates-probe",
        base / "cardiac-coordinates-probe.metallib",
    )
    field, boxes = map(Path, (meta["geometry_map"], meta["domain_boxes"]))
    for p, key in zip(
        (*probes, field, boxes),
        ("probe_sha256", "shader_sha256", "geometry_map_sha256", "domain_boxes_sha256"),
        strict=True,
    ):
        if sha(p) != meta[key]:
            raise ValueError(f"Cardiac authoring input changed: {p}")
    if (
        sha(original)
        != "87bd3cfad504b0b146d14e7b4f4e5aecb03815fac55fc5c3461ec7e77ab410c7"
    ):
        raise ValueError("Unexpected anatomy")
    targets = out / "forecast.csv.volumes.bin"
    coordinates = out / "coordinates.bin"
    if targets.stat().st_size != 30000 * 16:
        raise ValueError("Incomplete authoring forecast")
    subprocess.run([*probes, targets, field, boxes, coordinates], check=True)
    q = np.fromfile(coordinates, "<f4").reshape(-1, 16)
    latched = json.loads((case / "failure.json").read_text())["first_failure_latch"]
    idx = latched["step"]
    baseline = np.fromfile(previous / "coordinates.bin", "<f4").reshape(-1, 16)
    if q[:7500].tobytes() != baseline[:7500].tobytes():
        raise ValueError("Initial forecast differs from control")
    if not np.array_equal(
        q[idx, :7],
        np.asarray(latched["common_first"] + latched["common_second"][:3], dtype="<f4"),
    ):
        raise ValueError("First failed native candidate and forecast disagree")
    results = out / "prior-gate.bin"
    subprocess.run(
        [
            base / "cardiac-tube-membership-probe",
            r / "heart-build/test-membership.metallib",
            coordinates,
            previous / "unused-tube.bin",
            results,
        ],
        check=True,
    )
    codes = np.fromfile(results, "<u4")
    invalid = np.flatnonzero(codes == 0xFFFFFFFF)
    if (
        len(codes) != len(q)
        or invalid[0] != idx
        or not np.all(np.isin(q.view("<u4")[:, 8], [0, 5]))
    ):
        raise ValueError("Live rejection or solver mismatch")
    points = q[:, :7].astype(float)
    clusters = (
        fcluster(
            linkage(points[invalid], method="complete", metric="chebyshev"),
            0.040,
            criterion="distance",
        )
        if len(invalid) > 1
        else [1]
    )
    records = []
    for ids in sorted(
        [invalid[np.asarray(clusters) == c] for c in np.unique(clusters)], key=np.min
    ):
        t, p = tube(points[ids])
        records.append(dict(ids=ids.tolist(), tube=t.tolist(), pivot=p))
    write_atlas(original, out / "four-domains.bin", records)
    manifest = dict(
        input_sha256=sha(original),
        forecast_sha256=sha(coordinates),
        old_table_sha256=sha(r / "heart-build/test-membership.metallib"),
        extended_table_input_sha256=sha(out / "four-domains.bin"),
        domains=records,
    )
    (out / "four-domains.json").write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n"
    )
    summary = dict(
        original_control_geometry_map=meta["geometry_map_sha256"],
        targets=sha(targets),
        accepted_prefix_matched=7500,
        first_failure_step=idx + 1,
        first_failure_coordinate_parity=True,
        prior_gate_uncovered=len(invalid),
        parent_groups=[len(v["ids"]) for v in records],
        prior_gate=sha(results),
        failed_native_trace=sha(case / "failure.json"),
    )
    (out / "forecast-parity.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary, flush=True)
    with (out / "four-audit.log").open("w") as f:
        subprocess.run(
            [
                previous / "parent-audit",
                out / "four-domains.bin",
                out / "four-witness.csv",
                "split-tube",
            ],
            stdout=f,
            check=True,
        )
    subprocess.run(
        [sys.executable, r / "refine_heart.py", out, previous / "leaf-audit"],
        check=True,
    )
    from check_heart_proof import check

    lineage = check(out, original)
    (out / "verified-lineage.json").write_text(json.dumps(lineage, indent=2) + "\n")
    print(lineage, flush=True)


if __name__ == "__main__":
    prepare(*(Path(p).resolve() for p in sys.argv[1:]))

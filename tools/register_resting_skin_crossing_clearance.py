#!/usr/bin/env python3
"""Bounded inferred local skin fit using exact existing predicates; no scene admission."""

from pathlib import Path
import sys, json, hashlib, time, traceback, signal, argparse
import numpy as np

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "src"))
from numilab_human import resting_anatomy_verify as v, gate_preparation_boundary as gp

parser = argparse.ArgumentParser(
    description="Bounded local skin reference repair from one failed exact pair gate. Source preparation only."
)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--capture", type=Path, required=True)
parser.add_argument("--witness", type=Path, required=True)
parser.add_argument("--target-stable-id", type=int, required=True)
parser.add_argument("--context-muscle-stable-id", type=int, action="append", default=[])
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--method", choices=("witness", "projected-envelope"), default="witness"
)
parser.add_argument(
    "--direction", choices=("source-normal", "front-z"), default="source-normal"
)
parser.add_argument("--maximum-displacement-m", type=float, default=0.015)
args = parser.parse_args()
assert 0 < args.maximum_displacement_m <= 0.04, (
    "construction displacement budget must be in (0, .04] metres"
)
S = args.source.resolve(strict=True)
P = args.capture.resolve(strict=True)
W = args.witness.resolve(strict=True)
O = args.output.resolve()
target = (51005, args.target_stable_id)
target_keys = [target] + [(51005, sid) for sid in args.context_muscle_stable_id]
assert len(set(target_keys)) == len(target_keys), "duplicate muscle target"
assert not args.context_muscle_stable_id or args.method == "projected-envelope", (
    "context construction requires projected envelope"
)
witness = json.loads(W.read_text())
assert witness.get("passed") is False and {tuple(k) for k in witness["identities"]} == {
    (51007, 1),
    target,
}, "witness typed identities do not match requested skin/muscle pair"
assert witness.get("unexpected_triangle_pairs"), (
    "repair requires actual crossing witnesses"
)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
write = lambda p, d: p.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
assert not O.exists()
pre = gp.freeze_preparation(
    O.with_suffix(".preparation.json"),
    dict(
        status="PASS_PINNED_LOCAL_SKIN_REGISTRATION_PREPARATION",
        predicate_phase_started=False,
        scope="Local inferred skin registration from exact failed pair witnesses; source identity/material/physical owners retained. Four iterations/1200s, explicit configured construction displacement budget; front-z mode preserves source x/y on anterior sheet. Protect source skin positions z<.16m, preserve faces/GIDs, full positive material field on later export. Local self/pair gates only; whole-scene/motion/native admission still required.",
    ),
    runner_argv=[sys.executable, *sys.argv],
    input_pins={
        str(p): sha(p)
        for p in [
            Path(__file__),
            S,
            P,
            P.with_suffix(".receipt.json"),
            W,
            Path(v.__file__),
            Path(gp.__file__),
        ]
    },
)
gp.verify_frozen_preparation(pre)
O.mkdir()
(O / "construction-source.py").write_bytes(Path(__file__).read_bytes())
assert sha(O / "construction-source.py") == sha(__file__)
start = time.monotonic()
signal.signal(
    signal.SIGALRM,
    lambda *_: (_ for _ in ()).throw(
        TimeoutError("bounded1200s local skin construction")
    ),
)
signal.alarm(1200)
try:
    cap, _ = v.load_capture(P, P.with_suffix(".receipt.json"))
    mp, mf, mn, mg = cap[target]
    muscles = {
        key: v.prepare_surface(
            key, *cap[key][:2], normals=cap[key][2], source_vertex_ids=cap[key][3]
        )
        for key in target_keys
    }
    with np.load(S, allow_pickle=False) as z:
        p = z["capture_positions_m"].copy()
        f = z["capture_faces"].copy()
        g = z["capture_global_vertex_ids"].copy()
    pp = p.astype(float)
    area = np.cross(pp[f[:, 1]] - pp[f[:, 0]], pp[f[:, 2]] - pp[f[:, 0]])
    direction = np.zeros_like(pp)
    for c in range(3):
        np.add.at(direction, f[:, c], area)
    length = np.linalg.norm(direction, axis=1)
    used = np.unique(f)
    assert np.all(length[used] > 0)
    direction[length > 0] /= length[length > 0, None]
    direction[length == 0] = [0, 0, 1]
    if args.direction == "front-z" or args.method == "projected-envelope":
        direction[:] = [0, 0, 1]
    neighbors = [set() for _ in p]
    for tri in f:
        a, b, c = map(int, tri)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    lower = np.zeros(len(p))
    pairs = witness["unexpected_triangle_pairs"]
    history = []
    protected = pp[:, 2] < 0.16
    if args.method == "projected-envelope":
        from scipy.spatial import cKDTree

        mt = np.concatenate(
            [cap[key][0][cap[key][1]].astype(float) for key in target_keys]
        )
        ma = np.cross(mt[:, 1] - mt[:, 0], mt[:, 2] - mt[:, 0])
        mt = mt[ma[:, 2] > 1e-14]
        assert len(mt) > 0, "target needs anterior-facing projected triangles"
        mlo = mt[:, :, :2].min(1)
        mhi = mt[:, :, :2].max(1)
        mc = mt[:, :, :2].mean(1)
        mr = np.linalg.norm(mt[:, :, :2] - mc[:, None, :], axis=2).max(1)
        tree = cKDTree(mc)

        def cross2(a, b):
            return a[0] * b[1] - a[1] * b[0]

        def bary(point, tri):
            q = np.linalg.solve(
                np.column_stack((tri[1] - tri[0], tri[2] - tri[0])), point - tri[0]
            )
            return np.array([1 - q.sum(), *q])

        def intersect(subject, clip):
            poly = [x.copy() for x in subject]
            orientation = np.sign(cross2(clip[1] - clip[0], clip[2] - clip[0]))
            for ca, cb in zip(clip, np.roll(clip, -1, axis=0)):
                out = []
                for pa, pb in zip(poly, poly[1:] + poly[:1]):
                    da = orientation * cross2(cb - ca, pa - ca)
                    db = orientation * cross2(cb - ca, pb - ca)
                    if da >= 0:
                        out.append(pa)
                    if (da >= 0) != (db >= 0):
                        out.append(pa + da / (da - db) * (pb - pa))
                poly = out
                if not poly:
                    break
            return poly

        count = 0
        constrained = []
        for si, tri in enumerate(pp[f]):
            if np.any(protected[f[si]]) or area[si, 2] <= 1e-14:
                continue
            xy = tri[:, :2]
            lo = xy.min(0)
            hi = xy.max(0)
            if np.any(hi < mlo.min(0)) or np.any(lo > mhi.max(0)):
                continue
            center = xy.mean(0)
            radius = np.linalg.norm(xy - center, axis=1).max()
            deficit = 0.0
            for mi in tree.query_ball_point(center, radius + mr.max()):
                if np.any(hi < mlo[mi]) or np.any(lo > mhi[mi]):
                    continue
                polygon = intersect(xy, mt[mi, :, :2])
                if len(polygon) < 3:
                    continue
                for point in polygon:
                    sb = bary(point, xy)
                    mb = bary(point, mt[mi, :, :2])
                    deficit = max(
                        deficit, float(mb @ mt[mi, :, 2] - sb @ tri[:, 2] + 0.0002)
                    )
                count += 1
            if deficit > 0:
                lower[f[si]] = np.maximum(lower[f[si]], deficit)
                constrained.append(si)
        write(
            O / "projected-envelope-construction.json",
            dict(
                method="complete anterior target triangle projection; each source skin face receives a common conservative upper-envelope displacement before halo smoothing",
                target_typed_identities=target_keys,
                projected_overlap_pair_count=count,
                constrained_skin_faces=constrained,
                maximum_required_displacement_m=float(lower.max()),
                projection_is_construction_only=True,
                exact_predicate_gate_unchanged=True,
            ),
        )
    for iteration in range(1 if args.method == "projected-envelope" else 4):
        if args.method == "witness":
            for si, mi in pairs:
                n = area[si] / np.linalg.norm(area[si])
                plane = float(np.max(mp[mf[mi]].astype(float) @ n)) + 0.0002
                for vi in f[si]:
                    denom = float(direction[vi] @ n)
                    if denom < 0.25:
                        raise ValueError(
                            "witness skin face inconsistent with outward vertex direction; change method"
                        )
                    lower[vi] = max(lower[vi], (plane - float(pp[vi] @ n)) / denom)
        assert not np.any(lower[protected] > 0), (
            "actual witness demands protected back-side motion"
        )
        displacement = lower.copy()
        for ring in range(4):
            updated = displacement.copy()
            for vi in np.flatnonzero(displacement):
                for ni in neighbors[vi]:
                    if not protected[ni]:
                        updated[ni] = max(updated[ni], 0.65 * displacement[vi])
            displacement = updated
        assert not np.any(displacement[protected] > 0), (
            "protected back-side skin would move"
        )
        assert displacement.max() <= args.maximum_displacement_m, (
            "configured local construction displacement bound exceeded"
        )
        xyz = (pp + displacement[:, None] * direction).astype("<f4")
        out = O / f"variant-{iteration:02d}"
        out.mkdir()
        np.savez_compressed(
            out / "candidate.npz",
            capture_positions_m=xyz,
            capture_faces=f,
            capture_global_vertex_ids=g,
        )
        sf = v.prepare_surface((51007, 1), xyz, f, source_vertex_ids=g)
        selfaudit = sf.self_audit()
        write(out / "self.json", selfaudit)
        row = dict(
            iteration=iteration,
            maximum_displacement_m=float(displacement.max()),
            changed_vertices=int(np.any(xyz != p, axis=1).sum()),
            self_count=selfaudit["count"],
            passed=False,
            protected_back_side_exact=bool(
                np.array_equal(xyz[protected], p[protected])
            ),
        )
        if selfaudit["count"]:
            history.append(row)
            write(O / "history.json", history)
            break
        row["passed"] = True
        row["target_checks"] = []
        for key, muscle in muscles.items():
            result = v.audit_surface_pair(sf, muscle)
            write(out / f"pair-{key[1]}.json", result)
            row["target_checks"].append(
                dict(
                    identity=key,
                    passed=result["passed"],
                    triangle_pair_count=result["triangle_pair_count"],
                )
            )
            if not result["passed"]:
                row["passed"] = False
                break
        row["triangle_pair_count"] = result["triangle_pair_count"]
        history.append(row)
        write(O / "history.json", history)
        print(json.dumps(row), flush=True)
        if row["passed"]:
            break
        pairs = result["unexpected_triangle_pairs"]
        if not pairs:
            raise ValueError("containment failure needs different construction")
    write(
        O / "report.json",
        dict(
            status="PASS_LOCAL_SKIN_SELF_AND_REQUESTED_MUSCLE_ONLY"
            if history[-1]["passed"]
            else "FAIL_BOUNDED_LOCAL_CLEARANCE",
            method=args.method,
            direction=args.direction,
            maximum_displacement_budget_m=args.maximum_displacement_m,
            history=history,
            elapsed_seconds=time.monotonic() - start,
            anatomical_admission=False,
            native_admission=False,
        ),
    )
except BaseException as e:
    write(
        O / "failure.json",
        dict(
            error=repr(e),
            traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic() - start,
        ),
    )
    raise
finally:
    signal.alarm(0)

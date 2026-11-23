"""Reconstruct a connected forearm skin binding around the registered muscle force origins."""

import json, sys, runpy, subprocess, hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation
from track_elbow_attachments import prepare
from registered_forearms import read, sha
from reweight_forearm_skin import pose_rows, ramp
from measure_forearm_enclosure import Enclosure, measure
from verify_native_extensor_pair import loaded
from compare_hand_interfaces import query
from retain_bearing_skin_rows import preserve


def make_graph(neutral, faces):
    edges = np.unique(
        np.sort(
            np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
            axis=1,
        ),
        axis=0,
    )
    lengths = np.linalg.norm(neutral[edges[:, 0]] - neutral[edges[:, 1]], axis=1)
    return csr_matrix(
        (np.r_[lengths, lengths] + 1e-12, (edges.T.ravel(), edges[:, ::-1].T.ravel())),
        shape=(len(neutral),) * 2,
    )


def fit_weights(s, ids, faces, frames, rr, case):
    rest, v, _ = frames["neutral"]
    w = s["weights"][ids].astype(float)
    body = s["binds"]["body"]
    # Material topology keeps the torso and the neighbouring arm separate even
    # where those surfaces approach. Equal source positions connect duplicated seams.
    unique, back = np.unique(s["verts"][ids, :3], axis=0, return_inverse=True)
    back = back.ravel()
    _, rep = np.unique(back, return_index=True)
    graph = make_graph(v[rep], back[faces])
    delta = []

    def point(m):
        p = rest[m["body"]]
        return (
            Rotation.from_quat(p["quaternion_xyzw"]).apply(m["local_m"])
            + p["position_m"]
        )

    for origin_id, brd_id, humerus in [(137, 109, 41), (138, 110, 91)]:
        ext = next(x for x in rr["banks"] if x["stable"] == origin_id)
        brd = next(x for x in rr["banks"] if x["stable"] == brd_id)
        origins = np.array([point(p) for p in ext["sites"] if p["role"] == "origin"])
        a = origins.mean(0)
        wrist = point(brd["sites"][-1])
        upper = point(brd["sites"][0])
        axis = wrist - a
        L = np.linalg.norm(axis)
        axis /= L
        u = (v - a) @ axis
        rad = np.linalg.norm(v - a - u[:, None] * axis, axis=1)
        group = (body >= humerus) & (body <= humerus + 2)
        arm = w[:, group].sum(1)
        # Start on vertices with local skeletal support around the named origins;
        # extend in surface distance. The longitudinal taper preserves the wrist and shoulder.
        core = (u > -0.075) & (u < L * 0.59) & (rad < 0.080) & (arm > 0.54)
        ds = dijkstra(
            graph, indices=np.unique(back[core]), min_only=True, directed=False
        )[back]
        width = (
            ramp((0.065 - ds) / 0.060)
            * ramp((u + 0.140) / 0.070)
            * ramp((L + 0.035 - u) / 0.080)
            * ramp((arm - 0.12) / 0.28)
        )
        target = np.zeros_like(w)
        target[:, group] = w[:, group] / np.maximum(arm[:, None], 1e-20)
        if not np.allclose(target[width > 0].sum(1), 1):
            raise ValueError("No supported arm donor")
        delta.append(
            dict(
                delta=(target - w) * width[:, None],
                u=u,
                r=rad,
                core=int(core.sum()),
                origins=origins.tolist(),
            )
        )
    if np.any(
        (np.max(abs(delta[0]["delta"]), axis=1) > 0)
        & (np.max(abs(delta[1]["delta"]), axis=1) > 0)
    ):
        raise ValueError("Ambiguous bilateral skin")
    return delta[0]["delta"] + delta[1]["delta"], [d["core"] for d in delta]


def middle(case):
    frame = case / "step-3999.mrvpack.zst"
    r = read(case / "step-3999.receipt.json")
    raw = subprocess.check_output(["zstd", "-qdc", frame])
    if hashlib.sha256(raw).hexdigest() != r["pack_file_sha256"]:
        raise ValueError("Unmatched middle capture")
    return r, {p["body_index"]: p for p in r["accepted_body_poses"]}, raw


def prepare_pose(case, frame, s, ids, f, affected, rr, api, pr, exe):
    r, pose, raw = (
        middle(case) if frame == "middle" else loaded(case, frame == "release", False)
    )
    ((_, _, skin, faces),) = pr(raw, pose, skin_only=True)
    if not np.array_equal(f, faces):
        raise ValueError("Different skin topology")
    P = api["reference_transforms"](s, pose_rows(pose), [0.0, 0.0, 0.0])
    baseline = api["positions"](s, *P)[ids]
    active = affected[f].any(1)
    vertices = np.unique(f[active])
    lo = skin[vertices].min(0) - 0.080
    hi = skin[vertices].max(0) + 0.080
    all_faces = [f]
    fixed = []
    tags = [(51007, 1)] * len(f)
    offset = len(skin)
    muscles = {}
    for key, _, v, tri in pr(raw, pose):
        if key == (51007, 1):
            continue
        if key[0] == 51005 and key[1] in (109, 110, 137, 138):
            muscles[key[1]] = (v, tri)
        t = v[tri]
        incl = ((t.max(1) >= lo) & (t.min(1) <= hi)).all(1)
        if incl.any():
            local, rows = np.unique(tri[incl], return_inverse=True)
            fixed.append(v[local])
            all_faces.append(rows.reshape(-1, 3) + offset)
            offset += len(local)
            tags.extend([key] * int(incl.sum()))
    triangles = np.concatenate(all_faces)
    fixed = np.concatenate(fixed)
    flag = np.pad(active, (0, len(triangles) - len(f)))
    keys, bank = np.unique(tags, axis=0, return_inverse=True)
    bank = bank.ravel()

    def classify(v, exe, native_skin_only=False):
        if np.any(v[vertices] < lo) or np.any(v[vertices] > hi):
            raise ValueError("Left conservative reach")
        hits = query(exe, np.concatenate([v, fixed]), triangles, flag)

        def by(h):
            counts = np.bincount([bank[max(i, j)] for i, j in h], minlength=len(keys))
            return {f"{k[0]}/{k[1]}": int(n) for k, n in zip(keys, counts) if n}

        return by(hits), hits

    old, hit = classify(skin, exe)
    sites = {
        a["source_site"]: a
        for b in rr["banks"]
        for a in b["sites"]
        if a["role"] == "origin"
    }
    sites_world = np.array(
        [
            Rotation.from_quat(pose[m["body"]]["quaternion_xyzw"]).apply(m["local_m"])
            + pose[m["body"]]["position_m"]
            for m in sites.values()
        ]
    )
    return dict(
        skin=skin,
        transforms=P,
        material=baseline,
        old=old,
        hits=hit,
        classify=classify,
        keys=keys,
        bank=bank,
        sites=sites_world,
        muscles=muscles,
    )


def proposal(
    case,
    base,
    reader,
    dec,
    lib,
    routes,
    exe,
    out,
    outer_only=False,
    all_banks=False,
    parent_fit=None,
    other_run=None,
):
    out.mkdir(exist_ok=True)
    s, ids, f, frames, rr = prepare(case, base, reader, dec, lib, routes)
    api = runpy.run_path(str(dec))
    sys.path.insert(0, str(reader.parent / "src"))
    pr = runpy.run_path(str(reader))["primitives"]
    X = api["reference_transforms"](s, pose_rows(frames["neutral"][0]), [0, 0, 0])
    neutral = api["positions"](s, *X)
    dw, core = fit_weights(s, ids, f, frames, rr, case)
    affected = np.any(dw != 0, axis=1)
    if parent_fit is not None:
        prior = api["decode"](parent_fit.read_bytes())
        if (
            s["source_sha256"] != prior["source_sha256"]
            or s["inds"].tobytes() != prior["inds"].tobytes()
            or s["binds"].tobytes() != prior["binds"].tobytes()
        ):
            raise ValueError("Different parent registration")
        dw = prior["weights"][ids].astype(float) - s["weights"][ids].astype(float)
        affected = np.any(dw != 0, axis=1)
    bearing, bp = preserve(s, neutral, ids, [case])
    ctx = {
        k: prepare_pose(case, k, s, ids, f, affected, rr, api, pr, exe)
        for k in ("release", "middle", "terminal")
    }
    trials = []

    def evaluate(factor, name):
        w = s["weights"].copy()
        w[ids] = w[ids].astype(float) + factor[:, None] * dw
        w = np.maximum(w, 0.0)
        blob, r = api["rebind"](s, w, *X)
        p = api["decode"](blob)
        record = dict(
            name=name,
            poses={},
            source_rest_error=r["reference_error_max_m"],
            sha256=hashlib.sha256(blob).hexdigest(),
            rows=len(r["changed_rows"]),
        )
        before = neutral[s["inds"]]
        after = api["positions"](p, *X)[s["inds"]]
        cr = lambda t: np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n0, n1 = cr(before), cr(after)
        len0 = np.linalg.norm(n0, axis=1)
        len1 = np.linalg.norm(n1, axis=1)
        valid = len0 > 1e-14
        cosine = np.einsum("ij,ij->i", n0[valid], n1[valid]) / np.maximum(
            1e-30, len0[valid] * len1[valid]
        )
        record["min_reference_cosine"] = float(cosine.min())
        bad_skin = (
            set(f[np.flatnonzero(valid)[cosine < 0.9999]].ravel())
            if all_banks
            else set()
        )
        bad_vessel = set()
        for key, c in ctx.items():
            dv = api["positions"](p, *c["transforms"])[ids] - c["material"]
            v = (c["skin"].astype(float) + dv).astype("f4")
            counts, hits = c["classify"](v, exe)
            before = c["old"]
            report = {
                kind: [
                    sum(n for k, n in z.items() if k.startswith(str(kind) + "/"))
                    for z in (before, counts)
                ]
                for kind in (51004, 51005, 51007, 51011)
            }
            bad_banks = {k for k, n in counts.items() if n > before.get(k, 0)}
            for i, j in hits - c["hits"]:
                if j < len(f):
                    bad_skin.update(f[i])
                    bad_skin.update(f[j])
                else:
                    owner = c["keys"][c["bank"][j]]
                    if owner[0] == 51011:
                        bad_vessel.update(f[i])
                    if all_banks and f"{owner[0]}/{owner[1]}" in bad_banks:
                        bad_skin.update(f[i])
            shell = Enclosure(lib, v, f)
            cls, dist = shell.query(c["sites"])
            try:
                exposure = {k: measure(shell, *geo) for k, geo in c["muscles"].items()}
            finally:
                shell.close()
            record["poses"][key] = dict(
                pairs=report,
                banks=counts,
                new_banks=sorted(counts.keys() - before.keys()),
                origin_max_outside_mm=float(dist[cls == 0].max(initial=0) * 1000),
                external_origins=int((cls == 0).sum()),
                ambiguous_origins=int((cls < 0).sum()),
                muscles=exposure,
            )
        print(
            name,
            [p["pairs"] for p in record["poses"].values()],
            {k: round(v["exterior_area_fraction"], 3) for k, v in exposure.items()},
            "outside origins",
            record["poses"]["terminal"]["external_origins"],
            flush=True,
        )
        trials.append(record)
        return blob, bad_skin, bad_vessel

    if parent_fit is not None:
        from split_material_domains import domains

        _, part, changed, contact, incident = domains(s, prior, [case, other_run])
        touched = set(part[incident]) - {-1}
        unique, back = np.unique(s["verts"][ids, :3], axis=0, return_inverse=True)
        back = back.ravel()
        _, rep = np.unique(back, return_index=True)
        mat = make_graph(neutral[ids[rep]], back[f])
        distance = dijkstra(
            mat, indices=np.unique(back[incident]), directed=False, min_only=True
        )[back]
        free = ~np.isin(part, list(touched))
        evaluate(np.zeros(len(ids)), "control")
        bp.update(
            parent_fit=sha(parent_fit),
            quadrature_sources=[
                sha(x / "resting-local-bed-footprint.csv") for x in (case, other_run)
            ],
            selected_sources=[
                sha(x / "resting-support-selected-skin-witness.csv")
                for x in (case, other_run)
            ],
        )
        for width, label in (
            (0.0, "free-domains"),
            (0.030, "surface-distance"),
            (0.055, "wide-transition"),
        ):
            g = (
                free.astype(float)
                if not width
                else np.where(free, 1.0, ramp(distance / width))
            )
            for k in range(8):
                blob, ss, _ = evaluate(g, label if not k else f"{label}-guard-{k}")
                if not ss:
                    break
                # Fold and intersection guards follow the welded surface. A witness
                # on one side of an arm cannot suppress a different material sheet.
                local = dijkstra(
                    mat,
                    indices=np.unique(back[sorted(ss)]),
                    directed=False,
                    min_only=True,
                )[back]
                g *= ramp(local / 0.014)
            (out / (label + ".nhskin")).write_bytes(blob)
    elif outer_only:
        ww = s["weights"].copy()
        ww[ids] = np.maximum(0.0, ww[ids].astype(float) + dw)
        test = api["decode"](api["rebind"](s, ww, *X)[0])
        co = []
        for name in ("middle", "terminal"):
            state = ctx[name]
            v = state["skin"].astype(float)
            t = v[f]
            nf = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            sign = np.sign(np.einsum("ij,ij->", t[:, 0] - t[0, 0], nf))
            normal = np.zeros_like(v)
            for j in range(3):
                np.add.at(normal, f[:, j], sign * nf)
            norm = np.linalg.norm(normal, axis=1)
            normal = np.divide(
                normal,
                norm[:, None],
                out=np.zeros_like(normal),
                where=norm[:, None] > 0,
            )
            dv = api["positions"](test, *state["transforms"])[ids] - state["material"]
            co.append(np.einsum("ij,ij->i", normal, dv))
        # Use deformation that encloses the external cortex and extensor routes.
        # The flexor side retains its donor distribution where the same anatomy
        # shift would retract the skin into the flexor compartment.
        signed = np.minimum(*co)
        n = neutral[ids]
        e = np.unique(
            np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1),
            axis=0,
        )
        length = np.linalg.norm(n[e[:, 0]] - n[e[:, 1]], axis=1)
        edgewt = 1 / np.maximum(length, 0.0005)
        mat = csr_matrix(
            (np.tile(edgewt, 2), (e.T.ravel(), e[:, ::-1].T.ravel())),
            shape=(len(ids),) * 2,
        )
        degree = np.asarray(mat.sum(1)).ravel()
        tmp = signed.copy()
        for _ in range(6):
            tmp = 0.5 * tmp + 0.5 * (mat @ tmp) / np.maximum(degree, 1.0)
        gate = ramp((tmp + 0.002) / 0.012)
        evaluate(np.zeros(len(ids)), "control")
        options = [
            (1.0, 0.0, "outer"),
            (1.0, 1.0, "outer-bearing"),
            (0.65, 0.0, "outer-partial"),
        ]
        if all_banks:
            options = [(1.0, 0.0, "enclosing"), (1.0, 1.0, "enclosing-bearing")]
        for alpha, beta, label in options:
            g = alpha * gate * (1 - beta * (1 - bearing))
            blob, ss, vv = evaluate(g, label)
            # Preserve existing material at newly folded surface witnesses, with a
            # continuous transition into the surrounding anatomical patch.
            for k in range(10 if all_banks else 4):
                if not ss:
                    break
                dist = cKDTree(n[sorted(ss)]).query(n)[0]
                g *= ramp((dist - 0.001) / (0.012 if all_banks else 0.020))
                blob, ss, vv = evaluate(g, f"{label}-guard-{k}")
            (out / (label + ".nhskin")).write_bytes(blob)
    else:
        for strength, protect in [(0, 0), (0.55, 1), (1, 1), (0.55, 0), (1, 0)]:
            g = strength * (bearing if protect else np.ones(len(ids)))
            name = f"local-{strength}-{protect}"
            blob, ss, vv = evaluate(g, name)
            if strength == 1 and protect == 1:
                # Smoothly retain original bindings on any skin faces where the repair
                # introduces a nonadjacent fold. The physical frame stays unconstrained.
                for k in range(4):
                    if not ss:
                        break
                    d = cKDTree(neutral[ids[sorted(ss)]]).query(neutral[ids])[0]
                    g = g * ramp((d - 0.002) / 0.018)
                    name = f"fold-protected-{k}"
                    blob, ss, vv = evaluate(g, name)
                (out / "fold-protected.nhskin").write_bytes(blob)
            if strength == 1 and not protect:
                (out / "local-full.nhskin").write_bytes(blob)
    (out / "registration-test.json").write_text(
        json.dumps(
            dict(
                source=hashlib.sha256(s["raw"]).hexdigest(),
                bearing=bp,
                cores=core,
                trials=trials,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("case", "base", "reader", "dec", "lib", "routes", "exe", "out"):
        parser.add_argument(arg, type=lambda x: Path(x).resolve())
    parser.add_argument("--outer", dest="outer_only", action="store_true")
    parser.add_argument("--all-banks", action="store_true")
    parser.add_argument("--parent-fit", type=Path)
    parser.add_argument("--other-run", type=Path)
    args = parser.parse_args()
    if (args.parent_fit is None) != (args.other_run is None):
        parser.error("A parent fit and its native run must be supplied together")
    proposal(**vars(args))

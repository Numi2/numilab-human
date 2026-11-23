"""Fit a common skin binding across matched poses with exact surrounding collision witnesses."""

import json
import runpy
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from compare_hand_interfaces import query
from registered_forearms import sha
from verify_native_extensor_pair import loaded
from reweight_forearm_skin import pose_rows, ramp
from measure_forearm_enclosure import Enclosure


def solve(
    source, capture, reader, decoder, predicate, lib, directory, bearing_cases=()
):
    api = runpy.run_path(str(decoder))
    skin = api["decode"](source.read_bytes())
    transfer = np.load(directory / "weight-transfer.npz")
    rows, target, A, b = [transfer[k] for k in ("rows", "weights", "A", "b")]
    base_w = skin["weights"]
    neutral = api["positions"](skin, A, b)
    ids, faces = np.unique(skin["inds"], return_inverse=True)
    faces = faces.reshape(-1, 3)
    active = np.isin(ids, rows)
    changed = active[faces].any(1)
    near_vertices = np.unique(faces[changed])
    sys.path.insert(0, str(reader.parent / "src"))
    primitives = runpy.run_path(str(reader))["primitives"]

    def payload(g):
        w = base_w.copy()
        w[rows] = w[rows].astype(float) + g[:, None] * (target - w[rows])
        return api["rebind"](skin, w, A, b)[0]

    full = api["decode"](payload(np.ones(len(rows))))
    frames = []
    for name, initial in [
        ("physical-16", True),
        ("physical-8", False),
        ("physical-16", False),
    ]:
        receipt, poses, raw = loaded(capture / name, initial, False)
        pose = api["reference_transforms"](skin, pose_rows(poses), [0, 0, 0])
        ref = api["positions"](skin, *pose)[ids]
        ((_, _, v, f),) = primitives(raw, poses, skin_only=True)
        if not np.array_equal(f, faces):
            raise ValueError("Skin face ancestry differs")
        full_pos = v + api["positions"](full, *pose)[ids] - ref
        bounds = np.concatenate([v[near_vertices], full_pos[near_vertices]])
        # The full scene is checked in the swept box. Each final trial must lie inside it.
        lo, hi = bounds.min(0) - 0.015, bounds.max(0) + 0.015
        others = []
        tris = [faces]
        tags = [(51007, 1)] * len(faces)
        n = len(v)
        muscles = {}
        for key, _, x, t in primitives(raw, poses):
            if key == (51007, 1):
                continue
            if key[0] == 51005 and key[1] in (109, 110, 137, 138):
                muscles[key[1]] = x
            g = x[t]
            keep = ((g.max(1) >= lo) & (g.min(1) <= hi)).all(1)
            if np.any(keep):
                ii, ff = np.unique(t[keep], return_inverse=True)
                others.append(x[ii])
                tris.append(ff.reshape(-1, 3) + n)
                tags.extend([key] * int(keep.sum()))
                n += len(ii)
        all_f = np.concatenate(tris)
        fixed = np.concatenate(others)
        flag = np.pad(changed, (0, len(all_f) - len(faces)))
        keys, bank = np.unique(tags, axis=0, return_inverse=True)
        bank = bank.ravel()
        base = query(predicate, np.concatenate([v, fixed]), all_f, flag)
        frames.append(
            dict(
                name="initial" if initial else name,
                v=v,
                pose=pose,
                ref=ref,
                lo=lo,
                hi=hi,
                fixed=fixed,
                indices=all_f,
                flag=flag,
                keys=keys,
                bank=bank,
                base=base,
                muscles=muscles,
            )
        )
    envelope = np.ones(len(rows))
    footprint = {}
    if bearing_cases:
        from retain_bearing_skin_rows import preserve

        envelope, footprint = preserve(skin, neutral, rows, bearing_cases)
    alpha = 0.7 * envelope
    protected = set()
    runs = []
    for step in range(9):
        blob = payload(alpha)
        trial = api["decode"](blob)
        new_seeds = set()
        records = {}
        whole_gain = 0
        for frame in frames:
            v = frame["v"].copy()
            delta = api["positions"](trial, *frame["pose"])[ids] - frame["ref"]
            v[active] = (v[active].astype(float) + delta[active]).astype("f4")
            if (v[near_vertices] < frame["lo"]).any() or (
                v[near_vertices] > frame["hi"]
            ).any():
                raise ValueError("Trial left its conservative broad phase")
            found = query(
                predicate,
                np.concatenate([v, frame["fixed"]]),
                frame["indices"],
                frame["flag"],
            )
            base = frame["base"]
            keys = frame["keys"]
            bank = frame["bank"]

            def counts(hits):
                return np.bincount([bank[y] for x, y in hits], minlength=len(keys))

            old = counts(base)
            new = counts(found)
            bad = set(np.flatnonzero(new > old))
            by = {}
            for cls in set(keys[:, 0]):
                mask = keys[:, 0] == cls
                by[int(cls)] = [int(old[mask].sum()), int(new[mask].sum())]
            for x, y in found - base:
                if bank[y] in bad:
                    new_seeds.update(faces[x])
                    if y < len(faces):
                        new_seeds.update(faces[y])
            rec = dict(
                collisions=by,
                affected_bank_increases={
                    str(keys[i].tolist()): int(new[i] - old[i]) for i in sorted(bad)
                },
                changed_pair_count=[len(base), len(found)],
            )
            whole_gain += len(base) - len(found)
            if not bad:
                en = Enclosure(lib, v, faces)
                try:
                    rec["exterior"] = {}
                    for sid, p in frame["muscles"].items():
                        label, dist = en.query(p)
                        rec["exterior"][sid] = [
                            int((label == 0).sum()),
                            float(dist[label == 0].max(initial=0) * 1000),
                            int((label < 0).sum()),
                        ]
                finally:
                    en.close()
            records[frame["name"]] = rec
        tx = api["positions"](trial, A, b)
        ref_tri = neutral[skin["inds"]]
        test_tri = tx[skin["inds"]]
        cr = lambda t: np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        nn = cr(ref_tri)
        rn = cr(test_tri)
        d0 = np.linalg.norm(nn, axis=1)
        d1 = np.linalg.norm(rn, axis=1)
        nz = d0 > 1e-14
        dot = np.einsum("ij,ij->i", nn, rn)
        bad_norm = np.flatnonzero(nz & (dot < 0.9999 * d0 * d1))
        if len(bad_norm):
            new_seeds.update(faces[bad_norm].ravel())
        record = dict(
            iteration=step,
            poses=records,
            total_gain=whole_gain,
            candidate_sha256=__import__("hashlib").sha256(blob).hexdigest(),
        )
        runs.append(record)
        print(
            step,
            whole_gain,
            len(new_seeds),
            [r["affected_bank_increases"] for r in records.values()],
            flush=True,
        )
        if not new_seeds and whole_gain > 0:
            out = directory / "constrained.nhskin"
            paths = trial["inds"]
            x = api["positions"](trial, A, b)
            old = neutral[paths]
            new = x[paths]
            n0 = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
            n1 = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
            nonzero = np.linalg.norm(n0, axis=1) > 1e-14
            lens = np.linalg.norm(n0[nonzero], axis=1) * np.linalg.norm(
                n1[nonzero], axis=1
            )
            if (
                not np.array_equal(skin["inds"], paths)
                or skin["binds"].tobytes() != trial["binds"].tobytes()
                or trial["verts"][:, 3:6].tobytes() != skin["verts"][:, 3:6].tobytes()
            ):
                raise ValueError("Changed registration")
            record.update(
                reference_max_m=float(np.linalg.norm(x - neutral, axis=1).max()),
                min_reference_cosine=float(
                    (np.einsum("ij,ij->i", n0[nonzero], n1[nonzero]) / lens).min()
                ),
                changed_weight_rows=int(
                    np.any(trial["weights"] != base_w, axis=1).sum()
                ),
            )
            if (
                record["reference_max_m"] > 2e-7
                or record["min_reference_cosine"] < 0.9999
            ):
                raise ValueError("Neutral source integrity lost")
            out.write_bytes(blob)
            break
        if not new_seeds - protected:
            break
        protected.update(new_seeds)
        # A smooth guard over material-space distance also protects duplicated seam coordinates.
        dd, _ = cKDTree(neutral[ids[sorted(protected)]]).query(neutral[rows])
        alpha = 0.7 * envelope * ramp((dd - 0.002) / 0.022)
    (directory / "constrained.json").write_text(
        json.dumps(
            dict(
                source=sha(source),
                transfer=sha(directory / "weight-transfer.npz"),
                preserved_bearing_sources=footprint,
                runs=runs,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    solve(
        *(Path(p).resolve() for p in sys.argv[1:8]),
        bearing_cases=[Path(p).resolve() for p in sys.argv[8:]],
    )

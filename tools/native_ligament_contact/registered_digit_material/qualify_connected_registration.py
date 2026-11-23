"""Check the emitted registration against its neutral material and all measured local motion banks."""

import argparse
import json
import runpy
import shutil
from pathlib import Path

import numpy as np

from registered_forearms import read, sha
from reweight_forearm_skin import pose_rows


def qualify(source, bodies, fit, label, decoder, out):
    launch = read(source / "launch.json")
    args = launch["argv"]
    old = Path(args[args.index("--skin-payload") + 1])
    new = fit / (label + ".nhskin")
    test = read(fit / "registration-test.json")
    if not (test["source"] == sha(old) == launch["pins"][str(old)]):
        raise ValueError("Different control registration")
    matches = [t for t in test["trials"] if t["sha256"] == sha(new)]
    if not matches:
        raise ValueError("No measured candidate")
    trial = matches[-1]
    (base,) = [t for t in test["trials"] if t["name"] == "control"]
    if base["sha256"] != sha(old) or trial["sha256"] == base["sha256"]:
        raise ValueError("Unmatched control or empty change")
    api = runpy.run_path(str(decoder))
    s, t = [api["decode"](p.read_bytes()) for p in (old, new)]
    for k in ("source_sha256", "fingerprint", "nv", "nb", "ni"):
        if s[k] != t[k]:
            raise ValueError("Source identity changed")
    for k in ("binds", "inds"):
        if s[k].tobytes() != t[k].tobytes():
            raise ValueError("Topology or bindings changed")
    poses = {
        p["body_index"]: dict(
            position_m=p["rest"]["position_world_m"],
            quaternion_xyzw=p["rest"]["orientation_world_xyzw"],
        )
        for p in read(bodies)["all_bodies"]
    }
    transforms = api["reference_transforms"](s, pose_rows(poses), [0, 0, 0])
    x, y = [api["positions"](a, *transforms) for a in (s, t)]
    err = float(np.linalg.norm(x - y, axis=1).max())
    api["validate_weights"](t["weights"])
    changed = np.flatnonzero(np.any(s["weights"] != t["weights"], axis=1))
    if set(changed) - set(s["inds"].ravel()):
        raise ValueError("Changed unused material")

    def normals(pos):
        f = pos[s["inds"]]
        return np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])

    n0, n1 = normals(x), normals(y)
    l0, l1 = np.linalg.norm(n0, axis=1), np.linalg.norm(n1, axis=1)
    if not np.array_equal(l0 > 1e-14, l1 > 1e-14):
        raise ValueError("Rest degeneracy changed")
    valid = l0 > 1e-14
    cos = float(
        np.min(np.einsum("ij,ij->i", n0[valid], n1[valid]) / (l0[valid] * l1[valid]))
    )
    poses = {}
    for key in ("release", "middle", "terminal"):
        p, ref = trial["poses"][key], base["poses"][key]
        inc = {
            k: [ref["banks"].get(k, 0), n]
            for k, n in p["banks"].items()
            if n > ref["banks"].get(k, 0)
        }
        poses[key] = dict(
            affected_bank_increases=inc,
            external_origins=p["external_origins"],
            ambiguous_origins=p["ambiguous_origins"],
            pairs=p["pairs"],
        )
    run = dict(
        candidate_sha256=sha(new),
        changed_weight_rows=len(changed),
        reference_max_m=err,
        min_reference_cosine=cos,
        poses=poses,
        trial=trial["name"],
    )
    if (
        err > 2e-7
        or cos < 0.9999
        or any(r["affected_bank_increases"] for r in poses.values())
    ):
        raise ValueError("Failed neutral geometry or per-bank screen: " + str(run))
    out.mkdir()
    shutil.copyfile(new, out / "constrained.nhskin")
    if sha(out / "constrained.nhskin") != sha(new):
        raise ValueError("Copy changed")
    (out / "constrained.json").write_text(
        json.dumps(
            dict(
                source=sha(old),
                rest_pose=sha(bodies),
                registration_test_sha256=sha(fit / "registration-test.json"),
                runs=[run],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "bodies", "fit", "label", "decoder", "out"):
        parser.add_argument(arg, type=str if arg == "label" else Path)
    qualify(**vars(parser.parse_args()))

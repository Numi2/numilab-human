"""Rebind a certified reference and its matching loaded physical coordinates to the existing native owner."""

from pathlib import Path
import json, subprocess, sys
import numpy as np
from material_context import sha
from fit_dorsal_material import inputs
from torso_cell_geometry import geom
from install_registered_torso_volume import main as install


def bind(work, root, case, baseline, pose, owner, config, registered, out):
    proof = json.loads((case / "material/native-role-verification.json").read_text())
    if not proof["passed"]:
        raise ValueError("Production material role readers rejected")
    x, y, t, role, ids = inputs(baseline)
    delta = np.load(case / "field.npz")
    nodes = delta["nodes"]
    emitter = json.loads((case / "material-identity.json").read_text())
    if (
        emitter["native_source"] != ids
        or emitter["reference"] != sha(case / "mesh/mesh.node")
        or emitter["current"] != sha(case / "current.node")
    ):
        raise ValueError("Unmatched coordinate generation")

    def read(p):
        return np.loadtxt(p, skiprows=1, usecols=(1, 2, 3), dtype="f8")

    ref = read(case / "mesh/mesh.node")
    current = read(case / "current.node")
    if not np.array_equal(ref, np.float32(ref)) or not np.array_equal(
        current, np.float32(current)
    ):
        raise ValueError("Unexpected coordinate precision")
    inc = np.zeros(len(x))
    inc[nodes] = delta["delta_x_m"]
    for prior, now in ((x, ref), (y, current)):
        if not np.array_equal(prior[:, 1:], now[:, 1:]) or not np.array_equal(
            prior[:, 0].astype(float) + inc, now[:, 0]
        ):
            raise ValueError("Unmatched physical morph")
    provenance = json.loads((pose / "loaded-registration.json").read_text())
    band = json.loads((pose / "load.json").read_text())
    if (
        provenance["source"]["reference_sha256"] != ids["reference_sha256"]
        or provenance["source"]["elements_sha256"] != ids["elements_sha256"]
    ):
        raise ValueError("Loaded pose ancestry changed")
    anchor = np.array([r["volume_row"] for r in provenance["attachment_rows"]])
    requested = np.array([r["target_m"] for r in provenance["attachment_rows"]])
    if (inc[anchor] != 0).any() or np.linalg.norm(
        current[anchor] - requested, axis=1
    ).max() > 2e-7:
        raise ValueError("Moved physical attachment")
    outer = np.load(root / "physical.npz")["Orows"]
    loaded = (
        np.sort(outer[current[outer, 2] < band["width_m"]])
        if "width_m" in band
        else np.sort(outer[current[outer, 2] < float.fromhex("0x1.4f8b580000000p-17")])
    )
    if (
        not np.array_equal(loaded, np.sort(band["nodes"]))
        or np.any(inc[loaded])
        or current[outer, 2].min() != band["minimum_m"]
    ):
        raise ValueError("Physical bed contact changed")
    quality = {}
    sv = [float("inf"), 0.0]
    ratio = [float("inf"), 0.0]
    for i in (15, 16):
        mask = t[role == i]
        result = {"cells": len(mask), "invalid": 0, "inverted": 0}
        for start in range(0, len(mask), 16384):
            sel = mask[start : start + 16384]
            v, a = geom(ref, sel)
            z, b = geom(current, sel)
            J = z / v
            X = ref[sel]
            Y = current[sel]
            F = np.linalg.solve(X[:, 1:] - X[:, :1], Y[:, 1:] - Y[:, :1])
            s = np.linalg.svd(F, compute_uv=False)
            sv = [min(sv[0], float(s.min())), max(sv[1], float(s.max()))]
            ratio = [min(ratio[0], float(J.min())), max(ratio[1], float(J.max()))]
            result["invalid"] += int(
                np.sum(
                    (v <= 1e-18)
                    | (z <= 1e-18)
                    | (a < 1e-5)
                    | (b < 1e-5)
                    | (J < 0.4)
                    | (J > 2.5)
                )
            )
            result["inverted"] += int(np.sum((v <= 0) | (z <= 0)))
        if result["invalid"]:
            raise ValueError(("Invalid physical volume", result))
        quality[i] = result
    if sv[0] < 2 / 3 or sv[1] > 1.5:
        raise ValueError(("Unqualified seated stretch", sv))
    out.mkdir(exist_ok=False)
    info = json.loads((owner / "identities.json").read_text())
    header = "NumiHumanLoadedSkinPoseData.inc"
    main = "numilab_human_myosim_visual_probe.mm"
    if (
        info["binary_sha256"] != sha(owner / "numi-human-native")
        or info["source"] != sha(owner / main)
        or info["generated_header"] != sha(owner / header)
    ):
        raise ValueError("Parent owner identity changed")
    # Copy-on-write only the generated hash identities. The registered donor transforms,
    # bed, insertion targets and all runtime mechanics are exactly the parent owner.
    for f in owner.iterdir():
        if (
            f.is_file()
            and f.name != header
            and (
                f.suffix in (".inc", ".hpp", ".metal", ".metalinc", ".metallib")
                or f.name == main
            )
        ):
            (out / f.name).hardlink_to(f)
    (out / "metalrobo").symlink_to(
        (owner / "metalrobo").resolve(), target_is_directory=True
    )
    source = (owner / header).read_text()
    for old, new in (
        (ids["installed_pose_sha256"], emitter["current"]),
        (ids["reference_sha256"], emitter["reference"]),
    ):
        if source.count(old) != 1:
            raise ValueError("Wrong compiled material identity")
        source = source.replace(old, new)
    (out / header).write_text(source)
    commands = [
        [s.replace(str(owner) + "/", str(out) + "/") for s in cmd]
        for cmd in info["commands"]
    ]
    for cmd in commands:
        subprocess.run(
            cmd,
            cwd=json.loads((baseline / "launch.json").read_text())["cwd"],
            check=True,
        )
    info.update(
        commands=commands,
        parent=info["binary_sha256"],
        binary_sha256=sha(out / "numi-human-native"),
        binary=sha(out / "numi-human-native"),
        generated_header=sha(out / header),
        initial_coordinate_node=emitter["current"],
    )
    (out / "identities.json").write_text(json.dumps(info, indent=2) + "\n")
    install(config, registered / "material", case, case / "supported-launch.json")
    s = json.loads((case / "supported-launch.json").read_text())
    key = "NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"
    old = s["environment"][key]
    if s["pins"].pop(old) != ids["installed_pose_sha256"]:
        raise ValueError("Unmatched starting pose")
    s["environment"][key] = str(case / "current.node")
    s["pins"][str(case / "current.node")] = emitter["current"]
    s["current_pose_sha256"] = emitter["current"]
    (case / "supported-launch.json").write_text(json.dumps(s, indent=2) + "\n")
    report = dict(
        roles=quality,
        principal_stretch=sv,
        relative_volume=ratio,
        unchanged_contact_rows=len(loaded),
        unchanged_attachments=len(anchor),
        mass_reader=sha(case / "material/native-role-verification.json"),
        launch=sha(case / "supported-launch.json"),
        native_binary=info["binary_sha256"],
        reference=emitter["reference"],
        current=emitter["current"],
    )
    (case / "supported-registration.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(report, flush=True)


if __name__ == "__main__":
    bind(*(Path(v).resolve() for v in sys.argv[1:]))

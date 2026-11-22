"""Bind a locally equilibrated initial pose to the existing Metal owner and mass transfer."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from fit_dorsal_material import inputs
from material_context import sha
from torso_cell_geometry import geom


def coordinates(path):
    with path.open() as stream:
        size, dimension, attributes, marker = map(int, next(stream).split())
        values = np.loadtxt(stream)
    if (
        (dimension, attributes, marker) != (3, 0, 0)
        or values.shape != (size, 4)
        or not np.array_equal(values[:, 0], np.arange(1, size + 1))
    ):
        raise ValueError("Incomplete TetGen node set")
    return values[:, 1:]


def bind(run, parent, split, bed, case):
    report = json.loads((parent / "supported-registration.json").read_text())
    base = json.loads((parent / "supported-launch.json").read_text())
    fit = json.loads((case / "fit.json").read_text())
    reference, previous, tets, roles, ids = inputs(run)
    if (
        fit["inputs"] != ids
        or ids["installed_pose_sha256"] != report["current"]
        or ids["reference_sha256"] != report["reference"]
        or report["launch"] != sha(parent / "supported-launch.json")
    ):
        raise ValueError("Unmatched parent qualification")
    candidate = coordinates(case / "current.node")
    edited = np.load(case / "field.npz")
    rows = edited["nodes"]
    if (
        not np.array_equal(candidate, candidate.astype("f4"))
        or not np.array_equal(np.flatnonzero((candidate != previous).any(1)), rows)
        or not np.array_equal(edited["current"], candidate[rows])
        or not np.array_equal(edited["original"], previous[rows])
        or sha(case / "current.node") != fit["output_pose"]
    ):
        raise ValueError("Physical coordinate delta differs from fitted source")
    anchors = json.loads((bed / "loaded-registration.json").read_text())[
        "attachment_rows"
    ]
    anchor = [row["volume_row"] for row in anchors]
    support = json.loads((bed / "load.json").read_text())
    outer = np.load(split / "physical.npz")["Orows"]
    if (
        not np.array_equal(candidate[outer], previous[outer])
        or not np.array_equal(candidate[anchor], previous[anchor])
        or not np.array_equal(
            np.sort(outer[candidate[outer, 2] < support["band_m"]]),
            np.sort(support["nodes"]),
        )
    ):
        raise ValueError("Bed placement, skin boundary or bone attachment changed")
    massdir = parent / "fine-mass-routed"
    ledger = json.loads((massdir / "mass-transfer.json").read_text())
    if (
        sha(massdir / "current-density-closed-flow.npz") != fit["source_mass"]
        or fit["source_mass"] != ledger["flow_sha256"]
    ):
        raise ValueError("Physical mass mismatch")
    # Recheck every loaded cell; reference geometry and element indexing are hash-identical.
    reference = reference.astype(float)
    quality = {}
    stretch = [np.inf, 0.0]
    ratio = [np.inf, 0.0]
    for material in (15, 16):
        cells = tets[roles == material]
        invalid = inverted = 0
        for off in range(0, len(cells), 16384):
            tetra = cells[off : off + 16384]
            vol, q = geom(candidate, tetra)
            X, Y = reference[tetra], candidate[tetra]
            F = np.linalg.solve(X[:, 1:] - X[:, :1], Y[:, 1:] - Y[:, :1])
            J = np.linalg.det(F)
            s = np.linalg.svd(F, compute_uv=False)
            stretch = [min(stretch[0], s.min()), max(stretch[1], s.max())]
            ratio = [min(ratio[0], J.min()), max(ratio[1], J.max())]
            inverted += int(np.sum(vol <= 0.0))
            invalid += int(np.sum((vol <= 1e-18) | (q < 1e-5) | (J < 0.4) | (J > 2.5)))
        quality[material] = dict(cells=len(cells), invalid=invalid, inverted=inverted)
        if invalid or len(cells) != report["roles"][str(material)]["cells"]:
            raise ValueError(("Physical volume failure", quality))
    if stretch[0] < 2 / 3 or stretch[1] > 1.5:
        raise ValueError(("Physical stretch failure", stretch))
    old = parent / "native-owner"
    new = case / "native-owner"
    owner = json.loads((old / "identities.json").read_text())
    main = "numilab_human_myosim_visual_probe.mm"
    header = "NumiHumanLoadedSkinPoseData.inc"
    if (
        sha(old / "numi-human-native") != owner["binary_sha256"]
        or sha(old / main) != owner["source"]
        or sha(old / header) != owner["generated_header"]
        or owner["binary_sha256"] != report["native_binary"]
    ):
        raise ValueError("Parent native owner does not match registration")
    new.mkdir(exist_ok=False)
    for file in old.iterdir():
        if (
            file.is_file()
            and file.name != header
            and (
                file.name == main
                or file.suffix in (".inc", ".hpp", ".metal", ".metalinc", ".metallib")
            )
        ):
            (new / file.name).hardlink_to(file)
    (new / "metalrobo").symlink_to(
        (old / "metalrobo").resolve(), target_is_directory=True
    )
    content = (old / header).read_text()
    if content.count(report["current"]) != 1:
        raise ValueError("Compiled starting pose signature is ambiguous")
    (new / header).write_text(content.replace(report["current"], fit["output_pose"]))
    commands = [
        [arg.replace(str(old) + "/", str(new) + "/") for arg in cmd]
        for cmd in owner["commands"]
    ]
    for cmd in commands:
        subprocess.run(cmd, cwd=base["cwd"], check=True)
    owner.update(
        commands=commands,
        parent=owner["binary_sha256"],
        binary_sha256=sha(new / "numi-human-native"),
        binary=sha(new / "numi-human-native"),
        generated_header=sha(new / header),
        initial_coordinate_node=fit["output_pose"],
    )
    (new / "identities.json").write_text(json.dumps(owner, indent=2) + "\n")
    for name in ("fine-mass-routed", "reference.node"):
        (case / name).symlink_to((parent / name).resolve())
    key = "NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"
    if base["pins"].pop(base["environment"][key]) != report["current"]:
        raise ValueError("Unmatched launch pin")
    base["environment"][key] = str(case / "current.node")
    base["pins"][str(case / "current.node")] = fit["output_pose"]
    base["current_pose_sha256"] = fit["output_pose"]
    (case / "supported-launch.json").write_text(json.dumps(base, indent=2) + "\n")
    out = dict(
        roles=quality,
        principal_stretch=stretch,
        relative_volume=ratio,
        unchanged_contact_rows=len(support["nodes"]),
        unchanged_attachments=len(anchors),
        mass_reader=report["mass_reader"],
        source_mass=fit["source_mass"],
        reference=report["reference"],
        current=fit["output_pose"],
        native_binary=owner["binary_sha256"],
        launch=sha(case / "supported-launch.json"),
        parent_launch=report["launch"],
    )
    (case / "supported-registration.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    bind(*(Path(arg).resolve() for arg in sys.argv[1:]))

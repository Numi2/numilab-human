"""Replay a TetGen reference using the retained native volume reader and actual physical bounds."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from transfer_connected_volume import geometry
from audit_contacts import run as audit
from coupled_capacity import run as capacity
from prepare_contacts import run as prepare


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run(a):
    arm = json.loads(a.arms.read_text())
    if digest(a.elements) != arm["captured"]["elements"]:
        raise ValueError("Native element connectivity or physical region tags changed")
    cfg = json.loads((a.scene / "launch.json").read_text())["environment"]
    source = Path(
        cfg["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE"]
    )
    if digest(source) != arm["captured"]["nodes"]:
        raise ValueError("Reference nodes changed")
    native = a.output / "reference.volumes"
    subprocess.run([a.reader, a.nodes, a.elements, native], check=True)
    x = np.loadtxt(source, skiprows=1, usecols=(1, 2, 3), dtype=np.float32)
    y = np.loadtxt(a.nodes, skiprows=1, usecols=(1, 2, 3), dtype=np.float32)
    if x.shape != y.shape or len(np.unique(x, axis=0)) != len(x):
        raise ValueError("Reference material-node correspondence is ambiguous")
    table = np.loadtxt(a.elements, skiprows=1, usecols=(1, 2, 3, 4, 5), dtype=np.int32)
    t, tags = table[:, :4] - 1, table[:, 4]
    volumes, quality = geometry(y, t)
    v0, _ = geometry(x, t)
    independent = np.fromfile(native, dtype=np.float64)
    # Compare every element using its physical volume. A near-degenerate
    # sliver may have a larger determinant evaluation cancellation error.
    if independent.shape != volumes.shape or not np.allclose(
        independent, volumes, rtol=2e-7, atol=1e-24
    ):
        raise ValueError("Native and offline element volumes disagree")
    data = a.output / "material.npz"
    positions = a.output / "positions.npy"
    np.savez(data, tets=t, tags=tags)
    np.save(positions, y)
    pack = a.output / "boundaries.npz"
    prepare(a.scene, a.loader, a.pose, a.bones, a.arms, pack)
    expected = [s for group in arm["moved"] for s in group["banks"]]
    with np.load(pack) as p:
        actual = sorted(int(k[1:]) for k in p.files if k.startswith("b"))
        layers = [1, 0xFFFFFD11, 0xFFFFFD12]
        if sorted(expected) != actual or len(actual) != 62:
            raise ValueError("An affected skeletal bank is absent or ambiguous")
        bank_bed = {
            layer: float(y[np.unique(p[f"l{layer}"]), 2].min()) for layer in layers
        }
    pair_file = a.output / "crossing-pairs.json"
    audit(pack, positions, pair_file)
    rows = json.loads(pair_file.read_text())
    counts = {str(l): sum(len(p) for _, layer, p in rows if l == layer) for l in layers}
    capfile = a.output / "carrier-capacity.json"
    capacity(
        data, positions, a.allocation, a.flow, capfile, True, native_volumes=native
    )
    stat = json.loads(capfile.read_text())
    mats = {}
    for tag in (15, 16):
        take = tags == tag
        mats[str(tag)] = dict(
            cells=int(sum(take)),
            inverted=int(sum(independent[take] <= 0)),
            small_volume=int(sum(independent[take] <= 1e-18)),
            low_quality=int(sum(quality[take] <= 1e-5)),
            minimum_volume_m3=float(independent[take].min()),
            minimum_scaled_determinant=float(quality[take].min()),
            source_volume_m3=float(v0[take].sum()),
            candidate_volume_m3=float(independent[take].sum()),
        )
    return dict(
        scope="Static captured entrance, 62 moved bone banks and three physical layers",
        nodes=len(y),
        changed_nodes=int(np.any(y != x, axis=1).sum()),
        materials=mats,
        candidate_node_sha256=digest(a.nodes),
        native_volume_sha256=digest(native),
        native_reader_sha256=digest(a.reader),
        mass_solver_sha256=digest(a.flow),
        source=arm["captured"],
        reference_transform_sha256=digest(a.arms),
        pair_file_sha256=digest(pair_file),
        layer_bone_pairs=counts,
        minimum_layer_z_m=bank_bed,
        carrier_input_sha256=stat["input"],
        carrier_result=stat["after"],
        carrier_count=stat["carrier_count"],
        carrier_mass_kg=stat["carrier_mass_kg"],
        max_carrier_error_kg=stat["carrier_error_kg"],
        native_scene_admitted=False,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "nodes",
        "elements",
        "reader",
        "flow",
        "allocation",
        "scene",
        "loader",
        "pose",
        "bones",
        "arms",
        "output",
    ):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = run(args)
    (args.output / "receipt.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["layer_bone_pairs"]))

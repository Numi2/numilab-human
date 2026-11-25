"""Compile the larger registered gluteal operator through the current native Metal and host owners."""

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Unmatched source: " + old[:90])
    return text.replace(old, new)


def build(work, parent, owner, compiler, destination=None):
    install = destination or work / "expanded-owner"
    install.mkdir(exist_ok=True)
    cfg = json.loads((parent / "launch.json").read_bytes())
    env = cfg["environment"]
    base_gpu = json.loads((owner.parent / "linked/owner/build.json").read_bytes())
    base_host = json.loads((compiler / "build.json").read_bytes())
    old_matrix = Path(env["NUMI_HUMAN_GLUTEAL_MATRIX"])
    if sha(old_matrix) != cfg["pins"][str(old_matrix)]:
        raise ValueError("Different active stiffness")
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_"
    new_ref = Path(env[prefix + "NODE"])
    from scipy.spatial.transform import Rotation

    ports = work / "coupled/material"
    p = np.load(ports / "parents.npz")
    fat = p["fat"]
    donors = p["donors"]
    registration = json.loads((work / "clear/registration.json").read_bytes())
    proof = json.loads((work / "clear/source-occupancy.json").read_bytes())
    if registration["maximum_relative_area_error"] >= 1e-8 or any(
        v["summed_mm3"] != 0
        for frame in proof["source330_mm3"].values()
        for v in frame.values()
    ):
        raise ValueError("Unqualified reference compartment")
    if not np.array_equal(
        np.loadtxt(new_ref, skiprows=1, usecols=(1, 2, 3))[fat],
        p["reference"][: len(fat)],
    ):
        raise ValueError("Different stress-free skin attachment")
    if donors.tolist() != [128, 131, 145]:
        raise ValueError("Different bilateral muscle receiver")
    if (
        fat.ndim != 1
        or len(fat) == 0
        or np.any(fat < 0)
        or len(np.unique(fat)) != len(fat)
    ):
        raise ValueError("Invalid or repeated fat node")
    (span,) = set(
        re.findall(
            r"registered_material_span object=15 element_offset=\d+ element_count=\d+ state_offset=(\d+) state_count=\d+",
            (parent / "stderr.log").read_text(),
        )
    )
    rows = {
        b["body_index"]: b["current"]
        for b in json.loads((parent / "native-initial-bodies.json").read_bytes())[
            "all_bodies"
        ]
    }
    origins = np.array([rows[i]["position_world_m"] for i in donors])
    rot = np.array([rows[i]["orientation_world_xyzw"] for i in donors])
    if (
        np.max(abs(origins - p["origins"])) > 1e-8
        or np.max(abs(Rotation.from_quat(rot[0]).as_matrix() - p["pelvis_rotation"]))
        > 1e-7
    ):
        raise ValueError("Different skeletal rest frame")
    k = np.load(ports / "reference-tangent.npz")["matrix"]
    nfat = len(fat)
    points = nfat + len(donors)
    ndof = 3 * nfat + 6 * len(donors)
    if (
        points > 256
        or k.shape != (ndof, ndof)
        or not np.isfinite(k).all()
        or np.max(abs(k - k.T)) > 1e-8
    ):
        raise ValueError("Invalid condensed degrees of freedom")
    matrix = install / "material.f32"
    ((k + k.T) * 0.5).astype("<f4").tofile(matrix)
    src = install / "src"
    metal = src / "metal"
    metal.mkdir(parents=True, exist_ok=True)
    for folder, target, exclude in (
        (owner / "src", src, ("metal", "matter_kernels.metal")),
        (owner / "src/metal", metal, ("numi_human.metalinc",)),
    ):
        for f in folder.iterdir():
            if (
                f.name not in exclude
                and not (target / f.name).exists()
                and not (target / f.name).is_symlink()
            ):
                (target / f.name).symlink_to(f)
    shutil.copy2(owner / "src/matter_kernels.metal", src / "matter_kernels.metal")
    original = (owner / "src/metal/numi_human.metalinc").read_text()
    start = original.index("constant uint gcNodes[]")
    end = original.index(
        "kernel void nm_numi_human_assemble_fem_surface_attachment_loads", start
    )
    kernel = Path(__file__).with_name("gluteal_interface.metalinc").read_text()
    body = kernel.rstrip() + "\n\n"

    def vectors(name, x):
        n = x.shape[1]
        rows = [
            "float"
            + str(n)
            + "("
            + ",".join(f"{np.float32(value):.9e}f" for value in row)
            + ")"
            for row in x
        ]
        return f"constant float{n} {name}[]={{" + ",\n".join(rows) + "};\n"

    header = f"constant uint gcFat={nfat}u,gcPorts={points}u,gcTranslations={3 * nfat}u,gcDofs={ndof}u;\n"
    header += (
        "constant uint gcNodes[] = {" + ",".join(map(str, fat + int(span))) + "};\n"
    )
    header += "constant uint gcBodies[] = {" + ",".join(map(str, donors)) + "};\n"
    header += vectors(
        "gcPoints", np.vstack([p["reference"][:nfat], origins])
    ) + vectors("gcQuaternions", rot)
    gpu = metal / "numi_human.metalinc"
    gpu.write_text(original[:start] + header + body + original[end:])
    old_host = owner / "numi_human.mm"
    s = old_host.read_text()
    fingerprint = hashlib.sha256(
        (sha(matrix) + sha(gpu) + sha(ports / "parents.npz")).encode()
    ).hexdigest()
    s = once(s, env["NUMI_HUMAN_GLUTEAL_OPERATOR"], fingerprint)
    s = once(
        s,
        "matrix.length!=171u*171u*sizeof(float)",
        f"matrix.length!={ndof}u*{ndof}u*sizeof(float)",
    )

    def signature(path):
        return (
            "const unsigned char expected[]={"
            + ",".join(map(str, bytes.fromhex(sha(path))))
            + "};"
        )

    s = once(s, signature(old_matrix), signature(matrix))
    host = install / "numi_human.mm"
    host.write_text(s)
    commands = [list(base_gpu["commands"][i]) for i in (0, 1, 2)] + [
        list(base_host["commands"][2])
    ]
    lib = install / "libmetalrobo.dylib"
    for cmd in commands[:3]:
        for i, v in enumerate(cmd):
            if v.startswith(str(owner) + "/") and v in [
                str(owner / n)
                for n in (
                    "src/matter_kernels.metal",
                    "NumiMatter.air",
                    "NumiMatter.metallib",
                    "numi_human.mm",
                    "numi_human.mm.o",
                )
            ]:
                cmd[i] = str(install / Path(v).relative_to(owner))
    link = commands[3]
    obj = str(owner / "numi_human.mm.o")
    if link.count(obj) != 1:
        raise ValueError("Missing current host operator object")
    link[link.index(obj)] = str(install / "numi_human.mm.o")
    link[link.index("-o") + 1] = str(lib)
    for i, cmd in enumerate(commands):
        with (install / f"build-{i}.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    info = dict(
        reference=sha(new_ref),
        owner=sha(lib),
        metal=sha(install / "NumiMatter.metallib"),
        matrix=sha(matrix),
        operator=fingerprint,
        baseline_host=sha(old_host),
        baseline_gpu=sha(owner / "src/metal/numi_human.metalinc"),
        original_launch=sha(parent / "launch.json"),
        commands=commands,
        nodes=nfat,
        muscle_parents=len(p["muscle"]),
        donors=donors.tolist(),
        coordinates=ndof,
        source_span=int(span),
    )
    (install / "build.json").write_text(json.dumps(info, indent=2) + "\n")
    print("built", nfat, ndof, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "parent", "owner", "compiler"):
        parser.add_argument(name, type=Path)
    parser.add_argument(
        "--output", type=Path, help="directory for the compiled runtime"
    )
    args = parser.parse_args()
    build(
        args.work.resolve(),
        args.parent.resolve(),
        args.owner.resolve(),
        args.compiler.resolve(),
        args.output.resolve() if args.output else None,
    )

"""Reauthor only the fixed bed position and its registered source witnesses for a physical load comparison."""

import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from compile_connected_joint import digest
from update_witnesses import reproject


def build(case, vol, host, out):
    a = json.loads((case / "launch.json").read_bytes())
    args = a["argv"]
    old = Path(args[args.index("--support-contact-payload") + 1])
    p = Path(a["environment"]["NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"])
    code = (host / "NumiHumanLoadedSkinPoseData.inc").read_text()

    def scalar(name):
        return float.fromhex(re.search(r"\b" + name + r"\s*=\s*(0x[^;]+);", code)[1])

    packet = bytearray(old.read_bytes())
    if digest(old) != a["pins"][str(old)] or packet[:8] != b"NHCNT1\0\0":
        raise ValueError("Invalid registered bed")
    ox, oy, oz, *normal = struct.unpack_from("<6f", packet, 56)
    n = np.array(normal)
    expected = np.array(
        [
            float.fromhex(q)
            for q in re.search(r"bedNormal\[\]\s*=\s*\{([^}]+)", code)[1].split(",")
        ]
    )
    if (
        ox != 0
        or oy != 0
        or oz != scalar("fixedBedHeight")
        or not np.array_equal(n, expected)
    ):
        raise ValueError("Different support frame")
    if digest(p) != re.search(r'nodeSHA\[\]="([^"]+)"', code)[1]:
        raise ValueError("Different opening pose")
    x = np.loadtxt(p, skiprows=1, usecols=(1, 2, 3))
    outer = np.unique(np.load(vol / "material-walls.npz")["O"])

    def gaps(h):
        return x[:, 0] * n[0] + x[:, 1] * n[1] + (x[:, 2] - h) * n[2]

    g = gaps(oz)
    ids = np.array(
        [
            int(k.rstrip("u"))
            for k in re.search(r"registeredBedContactNodes\[\]\s*=\s*\{([^}]+)", code)[
                1
            ].split(",")
        ]
    )
    width = float.fromhex("0x1.4f8b580000000p-17")
    if abs(scalar("minimumBedGap") - g.min()) > 2e-18 or not np.array_equal(
        ids, outer[g[outer] < width]
    ):
        raise ValueError("Unmatched original contact history")
    new = float(np.float32(oz - 5.0e-6 / n[2]))
    g = gaps(new)
    first = outer[g[outer] < width]
    if not len(first) or g.min() <= 0:
        raise ValueError("Cannot retain real opening contact")
    out.mkdir(exist_ok=True)
    for name, val in [
        ("fixedBedHeight", new.hex()),
        ("minimumBedGap", g.min().item().hex()),
        (
            "registeredBedContactNodes[]",
            "{" + ",".join(str(i) + "u" for i in first) + "}",
        ),
    ]:
        code, k = re.subn(
            re.escape(name) + r"\s*=\s*[^;]*;", lambda _: name + " = " + val + ";", code
        )
        if k != 1:
            raise ValueError("Ambiguous bed registration")
    (out / "NumiHumanLoadedSkinPoseData.inc").write_text(code)
    shutil.copy2(
        host / "NumiHumanLoadedSkinPose.hpp", out / "NumiHumanLoadedSkinPose.hpp"
    )
    main = host / "numilab_human_myosim_visual_probe.mm"
    shutil.copy2(main, out / main.name)
    struct.pack_into("<6f", packet, 56, 0.0, 0.0, new, *normal)
    reproject(packet, new, n)
    edited = out / old.name
    edited.write_bytes(packet)
    spec = json.loads((host / "build.json").read_bytes())
    c = spec["commands"][2]
    l = spec["direction"]["commands"][3]
    obj = c[c.index("-o") + 1]
    newobj = str(out / "numi-human-native.o")
    c = [
        newobj if v == obj else str(out / main.name) if v == str(main) else v for v in c
    ]
    c[1:1] = ["-iquote", str(host)]
    if str(out / main.name) not in c:
        raise ValueError("Unknown native application source")
    l = [newobj if v == obj else v for v in l]
    l[l.index("-o") + 1] = str(out / "numi-human-native")
    for i, cmd in enumerate((c, l)):
        with (out / f"compile-{i}.log").open("w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    result = dict(
        control=digest(case / "launch.json"),
        physical_node_sha256=digest(p),
        original_bed=digest(old),
        new_bed=digest(edited),
        original_program=digest(host / "numi-human-native"),
        new_program=digest(out / "numi-human-native"),
        plane_normal=n.tolist(),
        normal_displacement_m=(new - oz) * n[2],
        prior_height_m=oz,
        new_height_m=new,
        first_contact_gap_m=float(g.min()),
        old_nodes=ids.tolist(),
        new_nodes=first.tolist(),
        commands=[c, l],
        header=digest(out / "NumiHumanLoadedSkinPoseData.inc"),
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("new bed nodes", first.tolist(), flush=True)


if __name__ == "__main__":
    build(*(Path(s).resolve() for s in sys.argv[1:]))

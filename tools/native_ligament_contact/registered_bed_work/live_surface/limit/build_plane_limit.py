"""Build the axis-exact feasibility bound in the existing registered plane shader."""

import json, sys, subprocess
from pathlib import Path
from material_context import sha
from patch_native import once


def build(root, base):
    current = root / "material-nodes"
    out = root / "plane-limit"
    build = out / "build"
    subprocess.run(["cp", "-cR", str(current / "build"), str(build)], check=True)
    kernel = build / "src/metal/contact.metalinc"
    s = kernel.read_text()
    anchor = "kernel void nm_contact_limit_rigid_line_search("
    s = once(
        s, anchor, (out / "fixed_plane_bound.metalinc").read_text() + "\n" + anchor
    )
    start = s.index(anchor)
    end = s.index("\nkernel void ", start + len(anchor))
    part = s[start:end]
    part = once(
        part,
        "                } else if (!certifiedOutward) {",
        """                } else if (contact.planeEnabled[0] != 0u) {
                    bound = min(bound, registeredPlaneFraction(separation,
                        feasibleSeparation, relativeDisplacement.z, boundaryRoundoff));
                } else if (!certifiedOutward) {""",
    )
    kernel.write_text(s[:start] + part + s[end:])
    cmd = json.loads((base / "velocity-consistent/identities.json").read_text())[
        "commands"
    ]

    def shader(src, dst, includes=()):
        air = dst.with_suffix(".air")
        c = cmd[0].copy()
        c[c.index("-c") + 1] = str(src)
        c[c.index("-o") + 1] = str(air)
        c[7:7] = includes
        subprocess.run(c, check=True)
        subprocess.run(
            ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(dst)],
            check=True,
        )

    shader(build / "src/matter_kernels.metal", build / "NumiMatter.metallib")
    shader(out / "check_bound.metal", out / "test.metallib", ["-I", str(build)])
    j = json.loads((build / "build-pins.json").read_text())
    j["NumiMatter.metallib"] = sha(build / "NumiMatter.metallib")
    (build / "build-pins.json").write_text(json.dumps(j, indent=2) + "\n")
    p = out / "parent"
    p.mkdir()
    cfg = json.loads((root / "gravity/parent/launch.json").read_text())
    env = cfg["environment"]
    k = "NUMI_MATTER_METALLIB_OVERRIDE"
    del cfg["pins"][env[k]]
    env[k] = str(build / "NumiMatter.metallib")
    cfg["pins"][env[k]] = j["NumiMatter.metallib"]
    (p / "launch.json").write_text(json.dumps(cfg, indent=2) + "\n")


if __name__ == "__main__":
    build(*(Path(v).resolve() for v in sys.argv[1:]))

"""Use the installed dense constitutive law's confined normal tangent in native bed columns."""

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from audit_cavity_response import elastic
from fit_posterior_equilibrium import materials
from material_context import sha


def read(p):
    return json.loads(p.read_text())


def save(p, data):
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def verify(props):
    # A unit tetrahedron with F=diag(1,1,s) in the same finite-strain material.
    ref = np.vstack((np.array([0.0, 0.0, 0.0]), np.eye(3)))
    t = np.array([[0, 1, 2, 3]])
    out = []
    for mu, lam in props:

        def axial(stretch):
            y = ref.copy()
            y[3, 2] = stretch
            _, force, _ = elastic(ref, y, t, np.arange(4), mu, lam)
            return -6 * force[3, 2]

        h = 2.0e-6
        slope = (axial(1 + h) - axial(1 - h)) / (2 * h)
        M = 2 * mu + lam
        test = abs(slope / M - 1.0)
        if test > 1e-7 or abs(axial(1)) > 1.0e-10:
            raise ValueError("Different normal tangent from tetrahedral material")
        # P_zz = mu*s + (lambda*log(s)-mu)/s for this particular law.
        curve = []
        for s in (1.0, 0.99, 0.95, 0.90, 0.80, 0.65):
            exact = mu * s + (lam * np.log(s) - mu) / s
            if abs(axial(s) - exact) > 1e-7 * max(1, abs(exact)):
                raise ValueError("Finite material response differs")
            curve.append(
                dict(
                    axial_stretch=s,
                    reference_normal_stress_pa=float(exact),
                    linearized_normal_stress_pa=float(M * (s - 1)),
                )
            )
        out.append(
            dict(
                mu_pa=float(mu),
                log_volume_coefficient_pa=float(lam),
                normal_tangent_pa=float(M),
                tetrahedral_relative_error=float(test),
                compression=curve,
            )
        )
    return out


def install(source, owner, dense, build, out):
    cfg = read(source / "launch.json")
    ids = read(owner / "identities.json")
    if any(sha(owner / n) != d for n, d in ids.items()):
        raise ValueError("Changed compact parent")
    dense_cfg = read(dense / "launch.json")
    dense_owner = Path(dense_cfg["argv"][0]).parent
    if (
        sha(dense_owner / "numi-human-native")
        != dense_cfg["pins"][dense_cfg["argv"][0]]
    ):
        raise ValueError("Dense material owner mismatch")
    props, pins = materials(dense_owner)
    checked = verify(props)
    if build.exists() or out.exists():
        raise FileExistsError("Use a new trial directory")
    subprocess.run(["cp", "-cR", owner, build], check=True)
    out.mkdir()
    file = build / "NumiHumanBedColumnQuadrature.hpp"
    h = file.read_text()
    for name, material in zip(("skinModulusPa", "fatModulusPa"), checked):
        h, n = re.subn(
            r"(static constexpr double " + name + r" = )([0-9.]+);",
            lambda m: m[1] + str(material["normal_tangent_pa"]) + ";",
            h,
        )
        if n != 1:
            raise ValueError("Different bed material interface")
    h = h.replace(
        "// Skin and fat act in series",
        "// Moduli are confined normal tangents at F=I from the pinned dense tissue deck.\n// Skin and fat act in series",
    )
    file.write_text(h)
    commands = [
        [a.replace(str(owner) + "/", str(build) + "/") for a in cmd]
        for cmd in read(owner / "build-commands.json")
    ]
    if (
        str(build / "numilab_human_myosim_visual_probe.mm") not in commands[2]
        or str(build / "numi-human-native") not in commands[3]
    ):
        raise ValueError("Wrong compile/link owner")
    runtime = Path(cfg["environment"]["DYLD_LIBRARY_PATH"]) / "libmetalrobo.dylib"
    if sha(runtime) != cfg["pins"][str(runtime)]:
        raise ValueError("Different native runtime")
    for cmd in commands[2:]:
        subprocess.run(cmd, cwd=runtime.parent.parent, check=True)
    save(build / "build-commands.json", commands)
    names = set(ids) | {file.name, "numilab_human_myosim_visual_probe.mm"}
    save(build / "identities.json", {n: sha(build / n) for n in sorted(names)})
    save(
        out / "material.json",
        dict(
            dense_launch=sha(dense / "launch.json"),
            material_pins=pins,
            materials=checked,
            changed_header=sha(file),
            parent_header=sha(owner / file.name),
            scope="Confined normal tangent at unit stretch; measured column thicknesses, tributary areas and damping are inherited. Finite-strain and lateral motion require separate checks.",
        ),
    )
    cfg["pins"].update(pins)
    cfg["parent_launch"] = sha(source / "launch.json")
    save(out / "launch.json", cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "owner", "dense", "build", "out"):
        p.add_argument(name, type=lambda s: Path(s).resolve())
    install(**vars(p.parse_args()))

"""Condense both native finite-strain materials at equal traction on the registered bed columns."""

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np
from fit_bed_material_to_dense import read, save
from material_context import sha


def stress(s, mu, lam):
    return mu * (1 / s - s) - lam * np.log(s) / s


def modulus(s, mu, lam):
    return mu * (1 + 1 / s**2) + lam * (1 - np.log(s)) / s**2


def column_force(d, area, skin, fat, props):
    # Independent columns in parallel. The two layers of each column have
    # equal normal traction and together supply exactly the imposed shortening.
    d = np.asarray(d)[:, None]
    lo = np.maximum(0.0, d - fat)
    hi = np.minimum(d, skin)
    for _ in range(50):
        u = (lo + hi) * 0.5
        ss = 1 - u / skin
        sf = 1 - (d - u) / fat
        higher = stress(ss, *props[0]) > stress(sf, *props[1])
        hi = np.where(higher, u, hi)
        lo = np.where(higher, lo, u)
    u = (lo + hi) * 0.5
    ss = 1 - u / skin
    sf = 1 - (d - u) / fat
    if np.min(ss) <= 0 or np.min(sf) <= 0:
        raise ValueError("Constitutive query exhausted stock")
    ts, tf = stress(ss, *props[0]), stress(sf, *props[1])
    if np.max(abs(ts - tf) / np.maximum(1.0, abs(tf))) > 2e-7:
        raise ValueError("Layer traction mismatch")
    return ts @ area, (
        1 / (skin / modulus(ss, *props[0]) + fat / modulus(sf, *props[1]))
    ) @ area


def hermite(v, w, h, u):
    u = u.astype(np.float32)
    df = w[:, 0] - v[:, 0]
    a = 3 * df - h * (2 * v[:, 1] + w[:, 1])
    b = h * (v[:, 1] + w[:, 1]) - 2 * df
    return v[:, 0] + u * (h * v[:, 1] + u * (a + u * b)), v[:, 1] + u * (
        2 * a + 3 * u * b
    ) / h


def build(source, owner, case, shader, build, out):
    cfg = read(source / "launch.json")
    material = read(source / "material.json")
    props = [
        (r["mu_pa"], r["log_volume_coefficient_pa"]) for r in material["materials"]
    ]
    if any(sha(Path(k)) != v for k, v in material["material_pins"].items()):
        raise ValueError("Mismatched source law")
    ids = read(owner / "identities.json")
    if any(sha(owner / k) != v for k, v in ids.items()):
        raise ValueError("Unmatched material parent")
    footprint = case / "resting-local-bed-footprint.csv"
    sections = list(csv.DictReader((case / "resting-local-bed-sections.csv").open()))
    points = list(
        csv.DictReader((case / "resting-support-selected-skin-witness.csv").open())
    )
    columns = list(csv.DictReader(footprint.open()))
    if len(columns) != 7282 or len(sections) != 38 or not points:
        raise ValueError("Wrong support registration")
    if len({int(r["vertex"]) for r in columns}) != len(columns):
        raise ValueError("Duplicate source column")
    curves = []
    metadata = []
    peak = [0.0, 0.0]
    n = 128
    for i, s in enumerate(sections):
        p = points[i]
        if (
            int(s["contact_index"]) != i
            or int(p["region"]) != i
            or int(p["point_query_index"]) != 628 + i
            or p["source_geometry_index"] != s["source_geometry_index"]
        ):
            raise ValueError("Incorrect material owner")
        rows = [r for r in columns if int(r["row"]) == i]
        area, skin, fat = [
            np.array([float(r[k]) for r in rows])
            for k in ("area_m2", "skin_m", "fat_m")
        ]
        if (
            not np.all(np.r_[area, skin, fat] > 0)
            or abs(area.sum() / float(s["projected_area_m2"]) - 1) > 1e-7
        ):
            raise ValueError("Incomplete physical footprint")
        # The table ends before any column reaches zero thickness. The GPU
        # reports an inadmissible compressed state outside this domain.
        limit = np.float32(0.95 * np.min(skin + fat))
        h = np.float32(limit / n)
        d = np.arange(n + 1) * h
        f, k = column_force(d, area, skin, fat, props)
        vals = np.stack((f, k), axis=1).astype("f4")
        curves.append(vals)
        for off in (0.13, 0.5, 0.87):
            u = np.full(n, off)
            direct = column_force((np.arange(n) + off) * h, area, skin, fat, props)
            pred = hermite(vals[:-1], vals[1:], h, u)
            for j in range(2):
                err = float(
                    np.max(abs(direct[j] - pred[j]) / np.maximum(1e-7, abs(direct[j])))
                )
                peak[j] = max(peak[j], err)
                if err > 0.002 or np.min(pred[j]) <= 0:
                    raise ValueError(("Curve precision", i, j, err))
        metadata.append(
            dict(
                index=i,
                source=int(s["source_geometry_index"]),
                point=628 + i,
                increment_m=float(h),
                maximum_m=float(limit),
                columns=len(rows),
                linear_compliance_m_per_N=float(1 / k[0]),
            )
        )
    if build.exists() or out.exists():
        raise FileExistsError("Use a new native cohort")
    subprocess.run(["cp", "-cR", owner, build], check=True)
    out.mkdir()

    def lit(v):
        return float(v).hex() + "f"

    tables = (
        "constant float2 forceAndModulus[] = {\n"
        + ",\n".join(
            "float2(" + lit(a) + "," + lit(b) + ")" for row in curves for a, b in row
        )
        + "\n};\n"
    )
    tables += (
        "constant float depthIncrement[] = {"
        + ",".join(lit(m["increment_m"]) for m in metadata)
        + "};\n"
    )
    (build / "NumiHumanBedForceCurves.metalinc").write_text(tables)
    stand = "NumiHumanStand.metal"
    solve = "NumiHumanStandSolve.metalinc"
    program = (shader / stand).read_text()
    marker = "inline StandContactNormalLaw standContactNormalLaw("
    if program.count(marker) != 1:
        raise ValueError("Mismatched GPU solver")
    program = program.replace(
        marker, '#include "NumiHumanBedForceCurves.metalinc"\n\n' + marker
    )
    old = """    return {support.boundaryNormalVelocity - gap / interval,
            compliance / (dt * interval)};"""
    new = """    if (gap >= 0.0f) {
        return {support.boundaryNormalVelocity - gap / interval, compliance / (dt * interval)};
    }
    if (support.pointQueryIndex < 628u || support.pointQueryIndex >= 666u)
        return {NAN, NAN};
    const uint row = support.pointQueryIndex - 628u;
    const float h = depthIncrement[row];
    const float position = -gap / h;
    if (!(position >= 0.0f && position <= 128.0f)) return {NAN, NAN};
    const uint index = min(uint(position), 127u);
    const float u = position - float(index);
    const float2 l = forceAndModulus[row*129u+index];
    const float2 r = forceAndModulus[row*129u+index+1u];
    const float df = r.x - l.x;
    const float a = 3.0f * df - h * (2.0f*l.y+r.y);
    const float b = h * (l.y+r.y) - 2.0f * df;
    const float force = l.x + u * (h*l.y + u*(a+u*b));
    const float tangent = l.y + u * (2.0f*a + 3.0f*u*b) / h;
    if (!(isfinite(force) && tangent > 0.0f && isfinite(tangent))) return {NAN, NAN};
    return {support.boundaryNormalVelocity + force/(tangent*interval),
            1.0f/(tangent*dt*interval)};"""
    if program.count(old) != 1:
        raise ValueError("Unmatched contact equation")
    (build / stand).write_text(program.replace(old, new))
    (build / solve).write_bytes((shader / solve).read_bytes())
    # Relink the unchanged shader objects first. This catches source/artifact
    # drift in every system sharing this Metal library, including Brain.
    old_commands = read(shader / "shader-build-commands.json")
    cwd = Path(cfg["environment"]["DYLD_LIBRARY_PATH"]).parent
    test = old_commands[1][:-1] + [str(build / "control.metallib")]
    subprocess.run(test, cwd=cwd, check=True)
    if sha(build / "control.metallib") != sha(owner / "metalrobo/MetalRobo.metallib"):
        raise ValueError("Nonmatching control GPU library")
    (build / "control.metallib").unlink()
    commands = [
        [a.replace(str(shader) + "/", str(build) + "/") for a in cmd]
        for cmd in old_commands
    ]
    if (
        str(build / stand) not in commands[0]
        or str(build / "metalrobo/MetalRobo.metallib") != commands[1][-1]
    ):
        raise ValueError("Mismatched GPU cohort")
    for cmd in commands:
        subprocess.run(cmd, cwd=cwd, check=True)
    # Validate the lookup ID and uncompressed compliance inside the actual
    # support builder. Area and thickness are anchored by the pinned source.
    host = "numilab_human_myosim_visual_probe.mm"
    hs = (build / host).read_text()
    anchor = "            row.planePoint.w = section.skinComplianceMPerN + section.fatComplianceMPerN;"
    if hs.count(anchor) != 1:
        raise ValueError("Unknown coupled host")
    guards = (
        "            constexpr double initialCompliance[] = {"
        + ",".join(float(m["linear_compliance_m_per_N"]).hex() for m in metadata)
        + "};\n"
    )
    guards += (
        "            constexpr unsigned sourceIds[] = {"
        + ",".join(str(m["source"]) for m in metadata)
        + "};\n"
    )
    guards += """            require(localBedSections.size()==38 && row.pointQueryIndex==628u+i &&
                    row.sourceGeometryIndex==sourceIds[i] &&
                    std::abs((section.skinComplianceMPerN+section.fatComplianceMPerN)/
                              initialCompliance[i]-1.0) < 1e-5,
                    "finite-strain bed table disagrees with registered source columns");
"""
    (build / host).write_text(hs.replace(anchor, guards + anchor))
    hostcmd = [
        [a.replace(str(owner) + "/", str(build) + "/") for a in cmd]
        for cmd in read(owner / "build-commands.json")
    ]
    for cmd in hostcmd[2:]:
        subprocess.run(cmd, cwd=cwd, check=True)
    save(build / "build-commands.json", hostcmd)
    save(build / "shader-build-commands.json", commands)
    names = set(ids) | {stand, solve, host, "NumiHumanBedForceCurves.metalinc"}
    save(build / "identities.json", {k: sha(build / k) for k in sorted(names)})
    cfg["pins"][str(footprint)] = sha(footprint)
    cfg["parent_launch"] = sha(source / "launch.json")
    save(out / "launch.json", cfg)
    save(
        out / "curves.json",
        dict(
            footprint=sha(footprint),
            material=material,
            sections=metadata,
            columns=len(columns),
            intervals=n,
            max_relative_heldout_force_error=peak[0],
            max_relative_heldout_tangent_error=peak[1],
            old_shader=sha(owner / "metalrobo/MetalRobo.metallib"),
            gpu=sha(build / "metalrobo/MetalRobo.metallib"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "owner", "case", "shader", "build", "out"):
        p.add_argument(name, type=lambda s: Path(s).resolve())
    build(**vars(p.parse_args()))

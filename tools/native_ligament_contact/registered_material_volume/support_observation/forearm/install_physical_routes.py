"""Rebind bilateral APL guides and their distributed tractions in the existing native payloads."""

import json
from pathlib import Path
import struct
import sys

import numpy as np
from numilab_human import model as owner
from compile_connected_joint import digest
from forearm_route import asset, read, route
from register_deltoid_origins import pack_envelope
from registration_stages import parts
from rebind_ligament_material import banks


def save(p, data):
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def check_intervals(side, metrics):
    attachments = {147: {"16": 0, "88": 6}, 148: {"17": 0, "107": 6}}[side]
    for k, m in metrics.items():
        if not m["invalid_segments"]:
            continue
        # Allow only an endpoint touching its own float32 cortical triangle.
        edge = attachments.get(k)
        if (
            m["invalid_mm"] > 0.001
            or edge is None
            or any(
                s != edge or (a != 0 if edge == 0 else b != 1)
                for s, a, b in m["invalid_segments"]
            )
        ):
            raise ValueError(
                ("Tendon segment intersects tissue", side, k, m["invalid_mm"])
            )


def install(control, donor, fitted, audit, out):
    c = read(control / "launch.json")
    ref = read(donor / "launch.json")
    a = c["argv"]
    m_path = Path(a[2])
    t_path = asset(c, "--tendon-payload")
    tm = t_path.with_suffix(".manifest.json")
    meta = read(tm)
    native = bytearray(m_path.read_bytes())
    tend = bytearray(t_path.read_bytes())
    h = struct.unpack_from("<8s9I32s", native)
    th = list(struct.unpack_from("<8s10I32s32s32s", tend))
    mesh = asset(c, "--soft-tissue-payload")
    bm = banks(Path(a[3]))
    settings = meta["admission"]
    matched = {str(p): digest(p) for p in [m_path, t_path, Path(a[3])]}
    if any(
        c["pins"].get(path) != sha or ref["pins"].get(path) != sha
        for path, sha in matched.items()
    ) or th[-2].hex() != digest(m_path):
        raise ValueError("Underlying muscle, bone or attachment revision differs")
    if len(tend) != 144 + 64 * th[5] + 288 * th[6] or th[0] != b"NHTEND3\0":
        raise ValueError("Unexpected tendon layout")
    ro = 76 + 16 * h[4] + 64 * h[5]
    used = {}
    for j in range(h[6]):
        kind, i, _, _ = struct.unpack_from("<4I", native, ro + 16 * j)
        if kind == 1:
            used[i] = used.get(i, 0) + 1
    reports = []
    record = read(audit)["rows"]
    for sid, mi in [(147, 259), (148, 322)]:
        fit = read(fitted / f"{sid}.json")
        samples = [r for r in record if r["side"] == sid]
        if (
            len(samples) != 3
            or [row["pack"] for row in samples] != fit["packs"]
            or fit["launch"] != digest(donor / "launch.json")
        ):
            raise ValueError("Evidence and fitted native poses differ")
        for sample in samples:
            check_intervals(sid, sample["metrics"]["candidate"])
        _, _, donors, _, aff = parts(mesh, sid)
        _, _, other, _, fa = parts(asset(ref, "--soft-tissue-payload"), sid)
        if donors != other or not all(
            np.array_equal(v, w) for aa, bb in zip(aff, fa) for v, w in zip(aa, bb)
        ):
            raise ValueError("Reference registration differs")
        ends = sorted(
            [
                (i, e)
                for i, e in enumerate(meta["endpoints"])
                if e["muscle_index"] == mi
            ],
            key=lambda x: x[1]["endpoint"] != "origin",
        )
        coords, which, sites = route(m_path, mi, donors, aff, [e for i, e in ends])
        if (
            sites != fit["sites"]
            or len(sites) != 8
            or min(used[k] for k in sites) != 1
            or max(used[k] for k in sites) != 1
            or len(ends) != 2
        ):
            raise ValueError("Route not private or changed")
        delta = np.asarray(fit["source_delta_mm"]) * 0.001
        for k, (site, index) in enumerate(zip(sites, which)):
            off = 76 + 16 * site
            body, *x = struct.unpack_from("<I3f", native, off)
            mat, t = aff[index]
            if body != donors[index] or np.max(abs(mat @ coords[k] + t - x)) > 1e-9:
                raise ValueError("Material frame mismatch")
            new = np.asarray(
                fit["endpoint_local_m"][int(k == 7)]
                if k in (0, 7)
                else x + mat @ delta[k - 1],
                dtype="<f4",
            ).astype(float)
            struct.pack_into("<I3f", native, off, body, *new)
            if k not in (0, 7):
                continue
            idx, ep = ends[int(k == 7)]
            row = list(struct.unpack_from("<8I8f", tend, 144 + 64 * idx))
            surf = ep["surface"]
            if row[:6] != [mi, int(k == 7), ep["route_node_index"], site, body, 2]:
                raise ValueError("Incorrect physical terminal")
            if (
                row[7] != surf["bone_stable_id"]
                or np.linalg.norm(np.array(x) - row[8:11]) > 1e-7
            ):
                raise ValueError("Terminal or cortex owner differs")
            env_off = 144 + 64 * th[5] + 288 * row[6]
            packet = struct.unpack_from("<4I68f", tend, env_off)
            if packet[:4] != (body, row[7], surf["source_triangle_index"], 4):
                raise ValueError("Invalid distributed load footprint")
            nodes = np.asarray(packet[4:20]).reshape(4, 4)[:, :3]
            ft, v, _ = bm[body, row[7]]
            point = (
                np.array(surf["nearest_barycentric"])
                @ v[ft[surf["source_triangle_index"]]]
            )
            if np.linalg.norm(new - point) > 1e-7:
                raise ValueError("Target moved off its original cortical triangle")
            rad = float(np.linalg.norm(nodes - new, axis=1).max())
            solved = owner._tendon_envelope_force_maps(
                new.tolist(), nodes.tolist(), rad
            )
            if solved is None:
                raise ValueError("Cannot conserve the terminal wrench")
            maps, metrics = solved
            if (
                rad > settings["maximum_patch_radius_m"]
                or metrics["sampled_total_force_amplification"]
                > settings["maximum_sampled_total_force_amplification"]
            ):
                raise ValueError("Footprint failed existing admission")
            env = dict(surf, **metrics)
            env.update(
                node_local_points_m=nodes.tolist(),
                nearest_local_point_m=point.tolist(),
                force_maps=maps,
                patch_radius_m=rad,
                surface_distance_m=float(np.linalg.norm(new - point)),
            )
            block = pack_envelope(env, body)
            if block[:80] != tend[env_off : env_off + 80]:
                raise ValueError("Material load sites moved")
            tend[env_off : env_off + 288] = block
            struct.pack_into(
                "<8I8f",
                tend,
                144 + 64 * idx,
                *row[:8],
                *new,
                env["surface_distance_m"],
                env["sampled_total_force_amplification"],
                rad,
                env["moment_residual_m"],
                row[-1],
            )
            mats = np.asarray(struct.unpack_from("<4I68f", block)[20:68]).reshape(
                4, 3, 4
            )[:, :, :3]
            skew = lambda r: np.asarray(
                [[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]]
            )
            ferr = float(np.linalg.norm(mats.sum(0) - np.eye(3), 2))
            merr = float(
                np.linalg.norm(sum(skew(q - new) @ A for q, A in zip(nodes, mats)), 2)
            )
            if ferr > 1e-5 or merr > 1e-6:
                raise ValueError("Float32 tendon map lost terminal wrench")
            ep.update(
                prior_source_local_point_m=x,
                source_local_point_m=new.tolist(),
                resolved_local_point_m=new.tolist(),
                surface={
                    name: value for name, value in env.items() if name != "force_maps"
                },
            )
            reports.append(
                dict(
                    muscle=mi,
                    site=site,
                    body=body,
                    endpoint=ep["endpoint"],
                    cortex=row[7],
                    footprint=row[6],
                    resultant_displacement_mm=1000 * np.linalg.norm(new - x),
                    cortical_gap_m=env["surface_distance_m"],
                    force_error=ferr,
                    moment_error_m=merr,
                )
            )
    out.mkdir()
    muscle = out / m_path.name
    muscle.write_bytes(native)
    th[-2] = bytes.fromhex(digest(muscle))
    struct.pack_into("<8s10I32s32s32s", tend, 0, *th)
    tendon = out / t_path.name
    tendon.write_bytes(tend)
    report = dict(
        control_launch=digest(control / "launch.json"),
        fit_source=digest(donor / "launch.json"),
        fits={s: digest(fitted / f"{s}.json") for s in [147, 148]},
        corridor_audit=digest(audit),
        endpoints=reports,
        parent_payloads=matched,
        updated_muscle=digest(muscle),
        updated_tendon=digest(tendon),
    )
    reg = out / "registration.json"
    save(reg, report)
    meta["payload"].update(sha256=digest(tendon), bytes=len(tend))
    meta["source"].update(
        myosim_muscle_payload_sha256=digest(muscle),
        forearm_traction_registration_sha256=digest(reg),
    )
    manifest = out / tm.name
    save(manifest, meta)
    a[2] = str(muscle)
    a[a.index("--tendon-payload") + 1] = str(tendon)
    c["pins"].update({str(p): digest(p) for p in [muscle, tendon, reg, manifest]})
    c["forearm_traction_control"] = report["control_launch"]
    save(out / "launch.json", c)


if __name__ == "__main__":
    install(*(Path(p).resolve() for p in sys.argv[1:]))

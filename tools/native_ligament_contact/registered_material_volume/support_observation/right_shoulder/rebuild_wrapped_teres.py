"""Rebuild bilateral scapular tendon routes with the registered cortical attachment and original inlet wrench.

MuJoCo evaluates static spatial-tendon geometry during authoring. The native
scene continues to own compliant muscle, tendon state and time integration.
"""

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def fmt(v):
    return " ".join(format(float(x), ".17g") for x in v)


class PathGeometry:
    def __init__(self, path, muscle, scapula, humerus):
        raw = path.read_bytes()
        header = struct.unpack_from("<8s9I32s", raw)
        wrap_base = 76 + 16 * header[4]
        route_base = wrap_base + 64 * header[5]
        muscle_base = route_base + 16 * header[6]
        _, first, count, _ = struct.unpack_from("<4I", raw, muscle_base + muscle * 164)
        program = [
            struct.unpack_from("<4I", raw, route_base + 16 * i)
            for i in range(first, first + count)
        ]
        if [kind for kind, *_ in program] != [1, 3, 1, 2, 1, 1]:
            raise ValueError("Changed teres major route")
        self.bodies = (scapula, humerus)
        sites = {}
        by_body = {body: [] for body in self.bodies}
        route = []
        for kind, idx, side, _ in program:
            site = idx if kind == 1 else side
            body, *pos = struct.unpack_from("<I3f", raw, 76 + 16 * site)
            if site not in sites:
                by_body[body].append(f'<site name="s{site}" pos="{fmt(pos)}"/>')
                sites[site] = (body, pos)
            if kind == 1:
                route.append(f'<site site="s{idx}"/>')
                continue
            body, wrap_type, radius, _, *coords = struct.unpack_from(
                "<2I14f", raw, wrap_base + 64 * idx
            )
            if wrap_type != kind:
                raise ValueError("Wrap geometry does not match its route element")
            orientation = Rotation.from_matrix(np.array(coords[3:]).reshape(3, 3))
            orient = fmt(np.roll(orientation.as_quat(), 1))
            params = (
                f'type="cylinder" size="{radius} 1"'
                if kind == 3
                else f'type="sphere" size="{radius}"'
            )
            by_body[body].append(
                f'<geom name="g{idx}" {params} pos="{fmt(coords[:3])}" quat="{orient}"/>'
            )
            route.append(f'<geom geom="g{idx}" sidesite="s{side}"/>')
        self.terminal_sites = [site for _, site, _, _ in program[-2:]]

        def body_link(b):
            return f'<body name="b{b}"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>{"".join(by_body[b])}</body>'

        xml = (
            "<mujoco><worldbody>"
            + "".join(map(body_link, self.bodies))
            + "</worldbody><tendon><spatial>"
            + "".join(route)
            + "</spatial></tendon></mujoco>"
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

    def measure(self, poses):
        for k, b in enumerate(self.bodies):
            p, r = poses[b]
            self.data.qpos[7 * k : 7 * k + 7] = np.r_[p, np.roll(r.as_quat(), 1)]
        mujoco.mj_forward(self.model, self.data)
        return float(self.data.ten_length[0])

    def site(self, which):
        return self.model.site("s" + str(self.terminal_sites[which])).pos


def poses(case, bs):
    data = json.loads((case / "native-initial-bodies.json").read_bytes())["all_bodies"]
    samples = {
        k: {
            b: (
                np.array(data[b][k]["position_world_m"]),
                Rotation.from_quat(data[b][k]["orientation_world_xyzw"]),
            )
            for b in bs
        }
        for k in ("rest", "current")
    }
    return samples


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare(parent, cortical, out):
    parent, cortical, out = map(Path, (parent, cortical, out))
    data = json.loads((cortical / "registration.json").read_bytes())
    cfg = json.loads((parent / "launch.json").read_bytes())
    a = cfg["argv"]
    m = Path(a[2])
    t = Path(a[a.index("--tendon-payload") + 1])
    cm = cortical / m.name
    ct = cortical / t.name
    if (
        sha(Path(a[3])) != data["source_bone"]
        or sha(m) != data["parent_muscle"]
        or sha(t) != data["parent_tendon"]
        or sha(cm) != data["muscle"]
        or sha(ct) != data["tendon"]
    ):
        raise ValueError("Inputs have different anatomical parents")
    muscle = bytearray(cm.read_bytes())
    tendon = bytearray(ct.read_bytes())
    manifest = json.loads(ct.with_suffix(".manifest.json").read_bytes())
    if (
        manifest["source"]["myosim_muscle_payload_sha256"] != data["muscle"]
        or manifest["payload"]["sha256"] != data["tendon"]
        or manifest["source"]["bodyparts3d_bone_payload"]["sha256"]
        != data["source_bone"]
        or sorted((p["body"], p["bank"]) for p in data["scapular_ports"])
        != [(34, 10), (84, 11)]
    ):
        raise ValueError("Cortical attachment metadata disagrees with the bound owners")
    fields = json.loads(json.dumps(data))
    results = []
    for rec in fields["scapular_ports"]:
        bs = rec["body"]
        mi, bh = {34: (217, 41), 84: (280, 91)}[bs]
        old = PathGeometry(m, mi, bs, bh)
        new = PathGeometry(cm, mi, bs, bh)
        sample = poses(parent, (bs, bh))
        frame = sample["current"]
        a0 = old.measure(frame)
        n = old.data.ten_wrapnum[0]
        points = old.data.wrap_xpos.reshape(-1, 3)[:n]
        p, R = frame[bs]
        local = R.inv().apply(points[-3:] - p)
        v0 = old.site(0).copy()
        s0 = old.site(1).copy()
        s1 = new.site(1).copy()
        u = local[1] - local[0]
        inlet_length = np.linalg.norm(u)
        if not np.isfinite(inlet_length) or inlet_length < 1e-8:
            raise ValueError("Degenerate inlet segment")
        u /= inlet_length
        if not np.allclose(local[1:], [v0, s0], atol=1e-8, rtol=0):
            raise ValueError("Wrap order changed")
        radius = np.linalg.norm(s0 - v0)
        dx = s1 - v0
        den = 2 * (radius - dx @ u)
        if abs(den) < 1e-8:
            raise ValueError("Singular terminal locus")
        # For displacement t along the incoming line u, require
        # t + ||new_terminal - old_via - t*u|| = old_terminal_length.
        # The root below retains the full length and incoming force direction.
        distance = (radius * radius - dx @ dx) / den
        if (
            radius - distance <= 0
            or np.linalg.norm(local[1] - local[0]) + distance <= 0
        ):
            raise ValueError("Changed wrapping branch")
        via = (v0 + distance * u).astype("<f4").astype(float)
        if not np.isfinite(via).all() or not np.array_equal(rec["new_terminal_m"], s1):
            raise ValueError("Invalid emitted terminal")
        if [rec["via_site"], rec["source_site"]] != new.terminal_sites:
            raise ValueError("Registration addresses different muscle sites")
        new.site(0)[:] = via
        if abs(new.measure(frame) - a0) > 1e-7:
            raise ValueError("Complete wrapped path length differs")
        bi, *xyz = struct.unpack_from("<I3f", muscle, 76 + 16 * rec["via_site"])
        if bi != bs or not np.array_equal(xyz, rec["new_via_m"]):
            raise ValueError("Unexpected private via")
        struct.pack_into("<3f", muscle, 80 + 16 * rec["via_site"], *via)
        rec["new_via_m"] = via.tolist()
        rec.pop("via_to_cortex_m")
        rec["terminal_length_change_m"] = float(np.linalg.norm(via - s1) - radius)
        results.append(
            {
                "muscle_index": mi,
                "wrapped_control_m": a0,
                "wrapped_emitted_m": new.measure(frame),
                "rest_difference_m": new.measure(sample["rest"])
                - old.measure(sample["rest"]),
            }
        )
    mm = out / m.name
    tt = out / t.name
    mm.write_bytes(muscle)
    h = list(struct.unpack_from("<8s10I32s32s32s", tendon))
    if h[0] != b"NHTEND3\0" or h[-2].hex() != data["muscle"]:
        raise ValueError("Cortical tendon has a different muscle owner")
    h[-2] = bytes.fromhex(sha(mm))
    struct.pack_into("<8s10I32s32s32s", tendon, 0, *h)
    tt.write_bytes(tendon)
    fields.update(
        muscle=sha(mm),
        tendon=sha(tt),
        cortical_registration_sha256=sha(cortical / "registration.json"),
        wrapped_path=results,
        scope="Registered four-point cortical attachment and shoulder route. The private same-body terminal segment redistributes geometric length to preserve complete wrapped muscle length and inlet line at the matched release pose. Compliant fibre and tendon constants are unchanged.",
    )
    manifest["source"]["myosim_muscle_payload_sha256"] = sha(mm)
    manifest["source"]["scapular_teres_registration"] = fields
    manifest["payload"].update(sha256=sha(tt), bytes=len(tendon))
    tt.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    (out / "registration.json").write_text(
        json.dumps(fields, indent=2, allow_nan=False) + "\n"
    )
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "parent_run",
        type=Path,
        help="matched covered shoulder native run with its initial rigid poses",
    )
    parser.add_argument(
        "cortical_inputs",
        type=Path,
        help="recovered scapular-ports-cortical payloads and registration",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="new output directory for the bound NHMYO/NHTENDON pair",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as stage:
        assembled = Path(stage) / "assembled"
        assembled.mkdir()
        fields = prepare(args.parent_run, args.cortical_inputs, assembled)
        os.rename(assembled, args.output)
    print(json.dumps(fields["wrapped_path"], indent=2))


if __name__ == "__main__":
    main()

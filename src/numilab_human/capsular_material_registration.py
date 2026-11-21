"""Bind the named superior glenohumeral entheses to emitted NHBONES1 faces."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from .composed_bone_body_frame import _decode_bones
from .model import _bodyparts_obj_triangles, sha256

LABEL = "superior_glenohumeral_capsuloligament"


def closest_material_attachment(raw_faces, local_faces, center, radius, target):
    """Minimize body-local distance over the part of each source face in the footprint.

    Both triangles use the same material coordinates. The registration may shear
    a face, so the objective and footprint are evaluated in their own frames.
    """
    center, target = np.asarray(center, float), np.asarray(target, float)
    if not np.isfinite(radius) or radius <= 0 or center.shape != (3,) or target.shape != (3,):
        raise ValueError("Invalid footprint")
    if (raw_faces.shape != local_faces.shape or raw_faces.shape[1:] != (3, 3) or
            not all(np.isfinite(a).all() for a in (center, target, raw_faces, local_faces))):
        raise ValueError("Invalid registered triangle arrays")

    offset = np.maximum(0, np.maximum(raw_faces.min(1) - center, center - raw_faces.max(1)))
    best = None
    # The native constructor requires strict interior barycentrics. Solve on
    # this slightly smaller triangle, including its edges, in both frames.
    inset = np.eye(3) * (1 - 3e-5) + 1e-5
    for face in np.flatnonzero(np.linalg.norm(offset, axis=1) <= radius):
        raw, local = inset @ raw_faces[face], inset @ local_faces[face]
        U, A = (raw[1:] - raw[0]).T, (local[1:] - local[0]).T
        M, H = U.T @ U, A.T @ A
        if np.linalg.det(M) <= 0 or np.linalg.det(H) <= 0:
            raise ValueError(f"Degenerate attachment face {face}")
        mid = np.linalg.solve(M, U.T @ (center - raw[0]))
        radial2 = radius**2 - np.linalg.norm(raw[0] + U @ mid - center)**2
        if radial2 < 0:
            continue
        g = A.T @ (target - local[0] - A @ mid)
        delta = np.linalg.solve(H, g)
        if delta @ M @ delta > radial2:
            if radial2 == 0:
                delta = np.zeros(2)
            else:
                def error(mu):
                    v = np.linalg.solve(H + mu * M, g)
                    return v @ M @ v - radial2

                upper = 1.
                while error(upper) > 0:
                    upper *= 2
                mu = brentq(error, 0., upper, xtol=1e-14)
                delta = np.linalg.solve(H + mu * M, g)
        uv = mid + delta
        weights = np.r_[1 - uv.sum(), uv]
        proposals = [weights @ inset] if weights.min() >= 0 else []
        for i in range(3):
            j = (i + 1) % 3
            u, a = raw[j] - raw[i], local[j] - local[i]
            middle = (center - raw[i]) @ u / (u @ u)
            square = radius**2 - np.linalg.norm(raw[i] + middle * u - center)**2
            if square < 0:
                continue
            half = np.sqrt(square / (u @ u))
            low, high = max(0., middle - half), min(1., middle + half)
            if low <= high:
                t = np.clip((target - local[i]) @ a / (a @ a), low, high)
                proposals.append((1 - t) * inset[i] + t * inset[j])
        for w in proposals:
            distance = np.linalg.norm(w @ local_faces[face] - target)
            if best is None or distance < best[0]:
                best = distance, int(face), w
    if best is None:
        raise ValueError("No registered material face intersects the attachment footprint")
    return best


def read_material_banks(path):
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    raw = path.read_bytes()
    pin = hashlib.sha256(raw).hexdigest()
    if manifest["payload"]["file"] != path.name or manifest["payload"]["sha256"] != pin:
        raise ValueError("Bone binary differs from its manifest")
    header, records, vo, io = _decode_bones(raw)
    records_by_id = {r[5]: r for r in records}
    if len(records_by_id) != len(records):
        raise ValueError("Duplicate stable bone identity")
    data = np.frombuffer(raw, "<f4", header[3] * 6, offset=vo).reshape(-1, 6)
    indices = np.frombuffer(raw, "<u4", header[4], offset=io)
    return manifest, pin, records_by_id, data, indices


def emitted_body_vertices(attributes, record):
    """Match bakeRegisteredBoneSurface, including its binary32 scale product."""
    xyz = np.multiply(attributes[:, :3], np.float32(record[13]), dtype=np.float32).astype(float)
    q = np.asarray(record[9:13], float)
    uv = np.cross(q[:3], xyz)
    return np.asarray(record[6:9]) + (xyz + (2*q[3]*uv + 2*np.cross(q[:3], uv)))


def material_bank(bank, sid, body):
    manifest, _, records, data, indices = bank
    row = records[sid]
    b, first, count, offset, size = row[:5]
    if b != body or size % 3 or first+count > len(data) or offset+size > len(indices):
        raise ValueError(f"Invalid physical owner or range on bank {sid}")
    attrs = data[first:first+count]
    f = indices[offset:offset+size].astype(np.int64).reshape(-1, 3) - first
    if f.min() < 0 or f.max() >= count or not np.isfinite(attrs).all():
        raise ValueError(f"Invalid material faces on bank {sid}")
    anchor = manifest["source"]["anchors"][sid-1]
    if (anchor["core_body_index"], anchor["source_record_index"]) != (body, row[-1]):
        raise ValueError(f"Source identity changed for bank {sid}")
    return anchor, row, attrs, f


def register(bone, fit_reference, atlas, routes, correspondence, regions):
    candidate = read_material_banks(bone)
    reference = read_material_banks(fit_reference)
    manifest, pin = candidate[:2]
    source = manifest["source"]
    if bone.resolve() != fit_reference.resolve() and source["connected_girdle_parent"]["bone_sha256"] != reference[1]:
        raise ValueError("Registration reference differs from the declared parent")
    if sha256(atlas) != source["bodyparts"]["archives"][0]["sha256"]:
        raise ValueError("Atlas identity differs from its manifest")
    fits = {v["sid"]: v for v in json.loads(correspondence.read_text())}
    footprints = {v["side"]: v for v in json.loads(regions.read_text())["regions"] if v["name"] == LABEL}
    fibers = [r for r in json.loads(routes.read_text())["routes"] if r["name"] in
              ("s_glenohum_ligament", "s_glenohum_ligament_l")]
    if sorted(r["side"] for r in fibers) != ["l", "r"]:
        raise ValueError("Expected one named superior glenohumeral source on each side")
    rows = []
    with zipfile.ZipFile(atlas) as archive:
        members = {Path(k).name: k for k in archive.namelist() if k.endswith(".obj")}
        for fiber in fibers:
            side = int(fiber["side"] == "r")
            region = footprints[side]
            attachments = []
            for i, site in enumerate(reversed(fiber["sites"])):
                sid, body = site["stable_id"], site["body"]
                anchor, record, attrs, faces = material_bank(candidate, sid, body)
                ref_anchor, ref_record, ref_attrs, ref_faces = material_bank(reference, sid, body)
                if (anchor != ref_anchor or record != ref_record or not np.array_equal(attrs, ref_attrs)
                        or not np.array_equal(faces, ref_faces)):
                    raise ValueError(f"Recompute the donor correspondence for changed shoulder bank {sid}")
                expected = ("scapula" if i == 0 else "humerus") + "_" + fiber["side"]
                fit = fits[sid]
                if (fit["body"] != body or fit["bone"] != expected or anchor["myosim_body"] != expected
                        or anchor["core_body_index"] != body or not np.isfinite(fit["scale"]) or fit["scale"] <= 0):
                    raise ValueError(f"Attachment owner mismatch for {site['site']}")
                member = members[anchor["member_id"] + ".obj"]
                obj = archive.read(member)
                if hashlib.sha256(obj).hexdigest() != anchor["member_sha256"]:
                    raise ValueError(f"Atlas member identity changed for {member}")
                raw, obj_faces = _bodyparts_obj_triangles(obj, member)
                if not np.array_equal(faces, obj_faces):
                    raise ValueError(f"Material triangle ancestry changed for {member}")
                raw_faces = (np.asarray(raw) * .001)[faces]
                vertices = emitted_body_vertices(attrs, record)
                local_faces = vertices[faces]
                target = Rotation.from_quat(fit["rotation_xyzw"]).inv().apply(
                    np.asarray(site["point_local_m"]) - fit["translation_m"]) / fit["scale"]
                center, radius = np.asarray(region["centers_atlas_m"][i]), region["bounds_m"][i]
                # Leave 0.1 micrometre for Float32 emission at the spherical boundary.
                distance, face, w = closest_material_attachment(raw_faces, local_faces, center, radius - 1e-7, target)
                attachments.append(dict(stable=sid, body=body, triangle=face, barycentric=w.tolist(),
                    source_site=site["site"], member=anchor["member_id"], member_sha256=anchor["member_sha256"],
                    source_correspondence_residual_m=distance, footprint_radius_m=radius,
                    raw_footprint_distance_m=float(np.linalg.norm(w @ raw_faces[face] - center)),
                    local_m=(w @ local_faces[face]).tolist()))
            rows.append(dict(name=LABEL, side=side, attachments=attachments, source=
                "Bilateral MyoSim named superior glenohumeral sites registered to material faces within the independent superior labral and lesser tubercle proxy regions."))
    return dict(candidate_sha256=pin, reference_sha256=reference[1], atlas_sha256=sha256(atlas), routes_sha256=sha256(routes),
                correspondence_sha256=sha256(correspondence), regions_sha256=sha256(regions), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bone", "fit_reference", "atlas", "routes", "correspondence", "regions", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    doc = register(args.bone, args.fit_reference, args.atlas, args.routes, args.correspondence, args.regions)
    args.output.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

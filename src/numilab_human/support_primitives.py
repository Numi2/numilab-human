"""Author source capsule/sphere/ellipsoid geometry for native current-pose plane contact."""
from __future__ import annotations
import hashlib
import math
import struct
from .model import ImportError


def compile_support_primitives(exported: dict, manifest: dict) -> tuple[dict, bytes]:
    """NHCNT2: 84-byte header, 96-byte primitive records; no physics stepping."""
    if exported.get("source") != manifest["source"]:
        raise ImportError("source export and native manifest provenance disagree")
    source = exported["support_contact"]
    ground = source["ground"]
    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 3.4028234663852886e38:
            raise ImportError("support primitive requires finite source numbers")
        return float(value)
    def vector(value, count):
        if not isinstance(value, list) or len(value) != count:
            raise ImportError("support primitive vector dimension mismatch")
        return [number(x) for x in value]
    point = vector(ground["point_world_m"], 3)
    normal = vector(ground["normal_world"], 3)
    if abs(sum(x*x for x in normal)-1) > 1e-6:
        raise ImportError("support primitive plane must have a unit normal")
    friction = number(ground["friction_tangential"])
    if friction < 0:
        raise ImportError("support primitive ground friction is negative")
    # Resolve through the already compiled native body map, never source IDs as native IDs.
    names = {row["name"]: row for row in manifest["support_contact"]["support_geometries"]}
    records, rows, seen, geometry_ids = [], [], set(), set()
    for geom in sorted(source["geometries"], key=lambda x: x["name"]):
        name = geom["name"]
        if name not in names or name in seen or geom["id"] in geometry_ids:
            raise ImportError("support primitive identity is unresolved or duplicated")
        seen.add(name); geometry_ids.add(geom["id"])
        binding = names[name]
        if type(geom["id"]) is not int or not 0 <= geom["id"] < 2**32 or \
                type(binding["core_body_index"]) is not int or not 0 <= binding["core_body_index"] < manifest["core_tree"]["engine_body_count"]:
            raise ImportError("support primitive index is outside the native model")
        if binding["source_body_id"] != geom["body"] or binding["source_geom_id"] != geom["id"]:
            raise ImportError("support primitive native/source identity disagrees")
        primitive = geom.get("primitive")
        if not isinstance(primitive, dict) or primitive.get("kind") not in {"sphere", "capsule", "ellipsoid"}:
            raise ImportError("NHCNT2 requires authored sphere/capsule/ellipsoid primitives")
        kind = {"sphere": 1, "capsule": 2, "ellipsoid": 3}[primitive["kind"]]
        endpoints = primitive["endpoints_local_com_m"]
        if not isinstance(endpoints, list) or len(endpoints) != 2:
            raise ImportError("support primitive endpoint count mismatch")
        a, b = [vector(p, 3) for p in endpoints]
        radius = number(primitive["radius_m"])
        gaps = vector(primitive["endpoint_plane_gaps_m"], 2)
        mu = number(geom["friction_tangential"])
        radii = vector(primitive.get("radii_m", [0,0,0]),3)
        orientation = vector(primitive.get("orientation_xyzw", [0,0,0,0]),4)
        if kind == 3:
            if radius != 0 or any(r < 1.1754943508222875e-38 for r in radii) or abs(sum(x*x for x in orientation)-1)>1e-6:
                raise ImportError("support ellipsoid has invalid axes/orientation")
        elif any(radii) or any(orientation):
            raise ImportError("sphere/capsule has unsupported ellipsoid fields")
        if (kind != 3 and radius < 1.1754943508222875e-38) or mu < 0 or (kind != 2 and (a != b or gaps[0] != gaps[1])):
            raise ImportError("support primitive has invalid radius/friction/sphere endpoints")
        records.append(struct.pack("<4I20f", binding["core_body_index"], geom["id"], kind, 0,
                                   *a, radius, *b, mu, *gaps, 0, 0, *orientation, *radii, 0))
        for end in range(2 if kind == 2 else 1):
            rows.append({"source_name": name, "source_geom_id": geom["id"], "endpoint": end,
                         "witness_index": len(rows)})
    if seen != set(names) or not records or len(rows) > 32:
        raise ImportError("support primitive inventory is incomplete")
    # Reuse the archive binding from the verified legacy payload compilation.
    source_hash = manifest["source"]["archive_sha256"]
    payload = struct.pack("<8s4I32s7f", b"NHCNT2\0\0", 2,
                          manifest["core_tree"]["engine_body_count"], len(records), 0,
                          bytes.fromhex(source_hash), *point, *normal, friction) + b"".join(records)
    return {"file": "myosim-fullbody-support-primitives.nhcnt", "payload_abi": 2,
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            "primitive_count": len(records), "expanded_contact_count": len(rows), "rows": rows}, payload


def main():
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-export", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exported = json.loads(args.source_export.read_text())
    manifest = json.loads(args.source_manifest.read_text())
    # The incremental compiler reuses an existing native map, with its rigid
    # bytes bound here instead of recalibrating all 416 muscle architectures.
    rigid = manifest["payloads"]["rigid"]
    rigid_path = args.source_manifest.parent / rigid["file"]
    raw = rigid_path.read_bytes()
    if len(raw) != rigid["bytes"] or hashlib.sha256(raw).hexdigest() != rigid["sha256"]:
        raise ImportError("source manifest native rigid payload drifted")
    metadata, payload = compile_support_primitives(exported, manifest)
    metadata["source_manifest_sha256"] = hashlib.sha256(args.source_manifest.read_bytes()).hexdigest()
    metadata["source_export_sha256"] = hashlib.sha256(args.source_export.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / metadata["file"]).write_bytes(payload)
    (args.output / "support-primitives.manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({k:v for k,v in metadata.items() if k != "rows"}))


if __name__ == "__main__":
    main()

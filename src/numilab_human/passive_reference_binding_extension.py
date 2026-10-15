"""Append existing registered supports to passive reference inspection rows.

This is asset preparation using NHTISS4 ABI5. Existing vertices, local faces,
physical muscle routes, original support records, and source identities survive
unchanged. Geometry composition and native motion admission remain separate.
"""
from pathlib import Path
import hashlib
import json
import struct

from . import passive_attachment_composition as pac


def extend_reference_bindings(source: Path, output: Path, extensions: dict[int, int]) -> dict:
    """Append a donor row's missing supports; never alter or invent a transform."""
    import numpy as np

    source, output = Path(source).resolve(), Path(output).resolve()
    pac.require(bool(extensions) and not output.exists(), "empty extension or existing output")
    data = pac._read_nhtiss4(source)
    mp, manifest = pac._payload_manifest(source)
    pac._validate_passive_payload_metadata(data, manifest)
    pins = {str(p): pac._surface_sha256(p) for p in (source, mp)}
    rows = {int(r[6]): r for r in data["records"]}
    pac.require(all(type(s) is int and type(d) is int and s != d
                    and s in rows and d in rows for s, d in extensions.items()),
                "binding extension requires distinct existing typed rows")
    new_records = data["records"].copy()
    parts, proofs, first = [], [], 0
    for j, row in enumerate(data["records"]):
        sid = int(row[6])
        local = pac._row_slices(data, row)
        binding = local["binding_bytes"]
        count = local["binding_count"]
        if sid in extensions:
            donor_id = extensions[sid]
            donor = pac._row_slices(data, rows[donor_id])
            pac.require(local["layer"] == donor["layer"] == 1,
                        "binding extension only prepares passive muscle inspection rows")
            identity = pac._manifest_row(manifest, sid)
            donor_identity = pac._manifest_row(manifest, donor_id)
            pac.require(len(identity["body_bindings"]) == count
                        and len(donor_identity["body_bindings"]) == donor["binding_count"],
                        "binding extension metadata count differs")
            by_core = {struct.unpack_from("<I", binding, k*36)[0]: binding[k*36:(k+1)*36]
                       for k in range(count)}
            pac.require(len(by_core) == count, "duplicate target support Core")
            added = []
            for k, support in enumerate(donor_identity["body_bindings"]):
                record = donor["binding_bytes"][k*36:(k+1)*36]
                core = struct.unpack_from("<I", record)[0]
                pac.require(core == support["core_body_index"], "donor support identity differs")
                if core in by_core:
                    pac.require(by_core[core] == record, "one Core has inconsistent registered transforms")
                    continue
                binding += record
                by_core[core] = record
                identity["body_bindings"].append(dict(support))
                added.append(dict(support))
                count += 1
            pac.require(bool(added), "donor supplies no new support")
            identity["body_weight_count"] = count
            proof = {"stable_id": sid, "donor_stable_id": donor_id,
                     "added_body_bindings": added,
                     "original_binding_records_and_vertex_weights_byte_exact": True,
                     "physical_routes_mass_tendon_and_activation_unchanged": True,
                     "scope": "Inferred passive common-interface support, copied exactly from an existing registered anatomical row. Original named physical attachments remain in matched_muscles."}
            identity.setdefault("passive_reference_binding_extensions", []).append(proof)
            proofs.append(proof)
        new_records[j, 0], new_records[j, 1] = first, count
        first += count
        parts.append(binding)
    raw = (struct.pack("<8s6I32s", b"NHTISS4\0", 5, data["surface_count"], first,
                       data["vertex_count"], data["index_count"],
                       data["fingerprint"], data["source_digest"])
           + new_records.astype("<u4").tobytes() + b"".join(parts)
           + data["raw"][data["vertex_start"]:])
    operation = {"input_sha256": pins, "rows": proofs,
                 "all_original_vertices_local_faces_and_support_records_preserved": True,
                 "native_admission": False, "anatomical_admission": False}
    manifest["source"].setdefault("passive_reference_binding_extensions", []).append(operation)
    manifest["payload"].update(binding_count=first, bytes=len(raw),
                               sha256=hashlib.sha256(raw).hexdigest())
    pac.require(pins == {p: pac._surface_sha256(p) for p in pins}, "binding source changed")
    output.mkdir()
    target = output/source.name
    target.write_bytes(raw)
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    result = pac._read_nhtiss4(target)
    pac._validate_passive_payload_metadata(result, manifest)
    for before, after in zip(data["records"], result["records"]):
        old, new = pac._row_slices(data, before), pac._row_slices(result, after)
        pac.require(old["vertex_bytes"] == new["vertex_bytes"]
                    and old["local_faces"] == new["local_faces"]
                    and new["binding_bytes"].startswith(old["binding_bytes"]),
                    "binding extension changed existing geometry or supports")
        if int(before[6]) not in extensions:
            pac.require(old["binding_bytes"] == new["binding_bytes"],
                        "unselected binding row changed")
    operation["payload_sha256"] = pac._surface_sha256(target)
    (output/"report.json").write_text(json.dumps(operation, indent=2, sort_keys=True)+"\n")
    return operation

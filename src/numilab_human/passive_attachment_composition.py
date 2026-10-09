"""Compose source-derived passive attachments in the existing NHTISS4 ABI5.

Only stable IDs 7, 8 and 23 are replaceable. This is an asset preparation step;
the native simulation retains ownership of mechanics, mass and tendon state.
Composition validates identity and wire layout, not anatomical admission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import errno
from pathlib import Path
import struct

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError('passive attachment composition: ' + message)

def compose(source: Path, output: Path, replacements: list[tuple[int, Path, Path]]) -> dict:
    import numpy as np
    (source, output) = (Path(source).resolve(), Path(output).resolve())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    T = source
    M = T.with_suffix('.manifest.json')
    inputs = {str(T): sha(T), str(M): sha(M)}
    repairs = {}
    require(bool(replacements), 'at least one passive attachment repair is required')
    for (sid, z, report) in replacements:
        sid = int(sid)
        if sid not in (7, 8, 23) or sid in repairs:
            raise ValueError('Only distinct passive attachment rows7,8,23 are allowed')
        (z, report) = (Path(z), Path(report))
        inputs.update({str(z): sha(z), str(report): sha(report)})
        repairs[sid] = (z, report)
    raw = T.read_bytes()
    require(len(raw) >= 64, 'truncated NHTISS4 header')
    (magic, abi, nr, nb, nv, ni, fp, source_digest) = struct.unpack_from('<8s6I32s', raw)
    require(magic == b'NHTISS4\x00' and abi == 5 and (nr == 150), 'unsupported NHTISS4 ABI or source inventory')
    require(len(raw) == 64 + nr * 32 + nb * 36 + nv * 56 + ni * 4, 'NHTISS4 byte ranges')
    declared = json.loads(M.read_text())
    expected = declared['payload']
    require(declared.get('schema') == 'numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1', 'manifest schema')
    require(expected['file'] == T.name and expected['sha256'] == sha(T) and (expected['bytes'] == len(raw)), 'source payload identity')
    require(expected['surface_count'] == nr and expected['binding_count'] == nb and (expected['vertex_count'] == nv) and (expected['index_count'] == ni), 'manifest count identity')
    require(expected['registration_fingerprint32'] == f'{fp:08x}' and declared['source']['myosim_source_archive_sha256'] == source_digest.hex(), 'source registration identity')
    rec = np.frombuffer(raw, dtype='<u4', count=nr * 8, offset=64).reshape(nr, 8)
    vo = 64 + nr * 32 + nb * 36
    io = vo + nv * 56
    idx = np.frombuffer(raw, dtype='<u4', count=ni, offset=io)
    require(len(raw) == io + ni * 4, 'candidate composition invariant')
    require(set(repairs).issubset(set((int(r[6]) for r in rec))), 'missing source stable ID')
    require(len(set((int(r[6]) for r in rec))) == nr, 'duplicate source stable ID')
    require([int(r[6]) for r in rec] == [int(r['stable_id']) for r in declared['source']['surfaces']], 'source row identity')
    for r in rec:
        require(int(r[0]) + int(r[1]) <= nb and int(r[2]) + int(r[3]) <= nv and (int(r[4]) + int(r[5]) <= ni) and (int(r[5]) % 3 == 0), 'source row byte range')
        ri = idx[int(r[4]):int(r[4] + r[5])]
        require(len(ri) > 0 and int(ri.min()) >= int(r[2]) and (int(ri.max()) < int(r[2]) + int(r[3])), 'source row allocation index range')
    newrec = rec.copy()
    vparts = []
    fparts = []
    vcur = icur = 0
    proofs = []
    unchanged = []
    for (j, r) in enumerate(rec):
        sid = int(r[6])
        oldbytes = raw[vo + int(r[2]) * 56:vo + int(r[2] + r[3]) * 56]
        oldfaces = (idx[int(r[4]):int(r[4] + r[5])] - r[2]).reshape(-1, 3)
        if sid in repairs:
            (zpath, rpath) = repairs[sid]
            with np.load(zpath, allow_pickle=False) as z:
                v = np.asarray(z['vertices6'], dtype='<f4')
                bi_raw = np.asarray(z['binding_indices'])
                require(np.issubdtype(bi_raw.dtype, np.integer) and (bi_raw >= 0).all() and (bi_raw <= 4294967295).all(), 'candidate binding indices must be unsigned integer values')
                bi = bi_raw.astype('<u4')
                w = np.asarray(z['weights'], dtype='<f4')
                f = np.asarray(z['faces'])
            n = len(v)
            require(n > 0 and v.shape == (n, 6) and (bi.shape == w.shape == (n, 4)), 'candidate vertex field shape')
            require(np.isfinite(v).all() and np.isfinite(w).all() and (w >= 0).all(), 'candidate nonfinite or negative fields')
            require(np.max(np.abs(w.sum(axis=1, dtype=np.float64) - 1)) < 1e-05, 'candidate composition invariant')
            require(f.ndim == 2 and f.shape[1] == 3 and (len(f) > 0) and np.issubdtype(f.dtype, np.integer), 'candidate face shape/type')
            require(f.min() >= 0 and f.max() < n, 'candidate composition invariant')
            valid = w > 0
            require((bi[valid] < r[1]).all(), 'record-local binding index out of range')
            faces = f.astype('<u4')
            vb = bytearray(n * 56)
            np.ndarray((n, 6), dtype='<f4', buffer=vb, offset=0, strides=(56, 4))[:] = v
            np.ndarray((n, 4), dtype='<u4', buffer=vb, offset=24, strides=(56, 4))[:] = bi
            np.ndarray((n, 4), dtype='<f4', buffer=vb, offset=40, strides=(56, 4))[:] = w
            vb = bytes(vb)
            proofs.append({'stable_id': sid, 'vertex_count_before': int(r[3]), 'vertex_count_after': n, 'triangle_count_before': len(oldfaces), 'triangle_count_after': len(faces), 'candidate': str(zpath), 'candidate_sha256': sha(zpath), 'report': str(rpath), 'report_sha256': sha(rpath), 'original_member_identity_retained': True})
        else:
            n = int(r[3])
            vb = oldbytes
            faces = oldfaces
            unchanged.append(sid)
        newrec[j, 2] = vcur
        newrec[j, 3] = n
        newrec[j, 4] = icur
        newrec[j, 5] = faces.size
        vparts.append(vb)
        fparts.append((faces + vcur).astype('<u4').tobytes())
        require(len(vb) == n * 56, 'candidate composition invariant')
        if sid not in repairs:
            require(vb == oldbytes and np.array_equal(faces, oldfaces), 'unchanged row drift')
        vcur += n
        icur += faces.size
    out = bytearray(raw[:vo])
    out[64:64 + nr * 32] = newrec.astype('<u4').tobytes()
    struct.pack_into('<II', out, 20, vcur, icur)
    out.extend(b''.join(vparts))
    out.extend(b''.join(fparts))
    require(raw[64 + nr * 32:vo] == out[64 + nr * 32:vo], 'candidate composition invariant')
    require(inputs == {p: sha(p) for p in inputs}, 'input changed during composition')
    output.mkdir(exist_ok=False)
    P = output / T.name
    P.write_bytes(out)
    manifest = json.loads(M.read_text())
    manifest['source']['upstream_runtime_binding_description'] = manifest['runtime_binding']
    manifest['runtime_binding'] = 'Unchanged passive rows retain their existing BodyParts3D route-body surface binding. Stable IDs ' + ','.join(map(str, sorted(repairs))) + ' use explicitly inferred source-derived reference attachment surfaces described per row. They retain the existing named MyoSim body-binding table and do not replace the authored physical force routes or compliant tendon state.'
    manifest['evidence_boundary'] = 'This mixed-source passive inspection package follows named articulated bodies. The reconstructed attachment rows are reference inferences, not measured-person source topology. This package does not create a force-transmitting continuum, new constitutive law, collision response, or clinical registration.'
    manifest['coverage']['upstream_preparation_counts_not_recomputed_after_reference_reconstruction'] = ['cancelled_opposite_face_pair_count', 'opposite_face_pair_cancellation_surface_count']
    manifest['payload'].update(file=P.name, sha256=sha(P), bytes=len(out), vertex_count=vcur, index_count=icur)
    for row in manifest['source']['surfaces']:
        sid = row['stable_id']
        if sid not in repairs:
            continue
        proof = next((p for p in proofs if p['stable_id'] == sid))
        row['vertex_count'] = proof['vertex_count_after']
        row['triangle_count'] = proof['triangle_count_after']
        row['reference_attachment_reconstruction'] = {**proof, 'scope': 'Passive source-derived reference inspection surface. Original anatomical identity/laterality, named attachment relationships, MyoSim force route and compliant tendon state remain their original owners. Not a measured-person reconstruction.', 'prior_registration_metadata': 'Retained above as upstream provenance; this reconstruction supersedes the listed source surface geometry.'}
    manifest['source']['reference_attachment_composition'] = {'source_payload_sha256': inputs[str(T)], 'changed_stable_ids': sorted(repairs), 'unchanged_row_vertex_bytes_and_local_faces': unchanged, 'binding_table_byte_exact': True, 'physical_route_mass_and_force_state_unchanged': True}
    PM = output / M.name
    PM.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    require(inputs == {p: sha(p) for p in inputs}, 'candidate composition invariant')
    proof = {'scope': 'Existing NHTISS4 ABI5 passive geometry composition; final native and anatomical admission separate.', 'input_sha256': inputs, 'inputs_unchanged': True, 'changed_rows': proofs, 'unchanged_row_vertex_bytes_and_local_faces': unchanged, 'binding_table_byte_exact': True, 'payload_sha256': sha(P), 'manifest_sha256': sha(PM), 'vertex_count': vcur, 'index_count': icur}
    proof['composer_source_sha256'] = sha(Path(__file__))
    (output / 'report.json').write_text(json.dumps(proof, indent=2) + '\n')
    return proof


def _surface_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_nhtiss4(path: Path) -> dict:
    import numpy as np
    raw = Path(path).read_bytes()
    require(len(raw) >= 64, "truncated NHTISS4 header")
    magic, abi, nr, nb, nv, ni, fingerprint, source_digest = struct.unpack_from("<8s6I32s", raw)
    require(magic == b"NHTISS4\x00" and abi == 5, "unsupported NHTISS4 ABI")
    require(len(raw) == 64 + nr*32 + nb*36 + nv*56 + ni*4, "NHTISS4 byte ranges")
    bs, vs, ix = 64 + nr*32, 64 + nr*32 + nb*36, 64 + nr*32 + nb*36 + nv*56
    records = np.frombuffer(raw, "<u4", nr*8, 64).reshape(nr, 8)
    indices = np.frombuffer(raw, "<u4", ni, ix)
    ids = [int(row[6]) for row in records]
    require(len(set(ids)) == nr, "duplicate NHTISS stable ID")
    for row in records:
        fb, bc, fv, vc, fi, ic = (int(v) for v in row[:6])
        require(fb+bc <= nb and fv+vc <= nv and fi+ic <= ni and ic > 0 and ic%3 == 0,
                "NHTISS row byte range")
        ri = indices[fi:fi+ic]
        require(len(ri) == ic and int(ri.min()) >= fv and int(ri.max()) < fv+vc,
                "NHTISS row index allocation")
    return {"path": Path(path).resolve(), "raw": raw, "sha256": hashlib.sha256(raw).hexdigest(),
            "surface_count": nr, "binding_count": nb, "vertex_count": nv, "index_count": ni,
            "fingerprint": fingerprint, "source_digest": source_digest, "records": records,
            "indices": indices, "binding_start": bs, "vertex_start": vs, "index_start": ix}


def _manifest_row(manifest: dict, stable_id: int) -> dict:
    surfaces = manifest.get("source", {}).get("surfaces")
    require(isinstance(surfaces, list), "manifest has no source surface rows")
    found = [row for row in surfaces if isinstance(row, dict) and row.get("stable_id") == stable_id]
    require(len(found) == 1, "manifest does not contain exactly one selected stable ID")
    return found[0]


def _payload_manifest(payload: Path) -> tuple[Path, dict]:
    path = Path(payload).with_suffix(".manifest.json")
    manifest = json.loads(path.read_text())
    require(manifest.get("schema") ==
            "numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1",
            "manifest schema")
    record = manifest.get("payload")
    require(isinstance(record, dict) and record.get("file") == Path(payload).name
            and record.get("sha256") == _surface_sha256(payload)
            and record.get("bytes") == Path(payload).stat().st_size, "manifest payload identity")
    return path.resolve(), manifest


def _row_slices(data: dict, row) -> dict:
    fb, bc, fv, vc, fi, ic, sid, layer = (int(value) for value in row)
    vb = data["raw"][data["vertex_start"]+fv*56:data["vertex_start"]+(fv+vc)*56]
    bb = data["raw"][data["binding_start"]+fb*36:data["binding_start"]+(fb+bc)*36]
    local = (data["indices"][fi:fi+ic].astype("<i8")-fv).reshape(-1, 3)
    require(len(vb) == vc*56 and len(bb) == bc*36 and len(local)*3 == ic,
            "selected NHTISS row truncated")
    return {"first_binding":fb,"binding_count":bc,"first_vertex":fv,"vertex_count":vc,
            "first_index":fi,"index_count":ic,"stable_id":sid,"layer":layer,
            "binding_bytes":bb,"vertex_bytes":vb,"local_faces":local.tolist()}


def _producer_source_identity(source: dict) -> dict:
    keys = ("bodyparts", "registration", "myosim_manifest", "myosim_source_archive_sha256", "surface_map")
    return {key: source.get(key) for key in keys}


def _replay_conforming_edge(source: Path, context_manifest_path: Path, stable_id: int,
                            requested_edge: tuple[int, int]) -> tuple[bytes, dict, dict]:
    import math
    import numpy as np
    from . import model

    source = Path(source).resolve()
    context_manifest_path = Path(context_manifest_path).resolve()
    source_manifest_path = source.with_suffix(".manifest.json")
    start_hashes = {str(source): _surface_sha256(source),
                    str(source_manifest_path): _surface_sha256(source_manifest_path),
                    str(context_manifest_path): _surface_sha256(context_manifest_path)}
    source_manifest_path, source_manifest = _payload_manifest(source)
    context = json.loads(context_manifest_path.read_text())
    require(context.get("schema") == source_manifest.get("schema"), "producer context manifest schema")
    context_payload_path = (context_manifest_path.parent / context["payload"]["file"]).resolve()
    _, checked_context = _payload_manifest(context_payload_path)
    require(checked_context == context, "producer context manifest changed during read")
    base, candidate = _read_nhtiss4(source), _read_nhtiss4(context_payload_path)
    helper_path = Path(model.__file__).resolve()
    start_hashes[str(context_payload_path)] = _surface_sha256(context_payload_path)
    start_hashes[str(helper_path)] = _surface_sha256(helper_path)
    for data, manifest_record in ((base, source_manifest["payload"]),
                                  (candidate, context["payload"])):
        require(
            manifest_record.get("surface_count") == data["surface_count"]
            and manifest_record.get("binding_count") == data["binding_count"]
            and manifest_record.get("vertex_count") == data["vertex_count"]
            and manifest_record.get("index_count") == data["index_count"]
            and manifest_record.get("registration_fingerprint32") == f'{data["fingerprint"]:08x}',
            "NHTISS payload metadata counts/fingerprint",
        )
    require(source_manifest["payload"]["sha256"] == base["sha256"], "accepted039 source payload identity")
    require(source_manifest["source"]["myosim_source_archive_sha256"] == base["source_digest"].hex()
            and context["source"]["myosim_source_archive_sha256"] == base["source_digest"].hex(),
            "NHTISS MyoSim archive identity")
    require(base["fingerprint"] == int(source_manifest["payload"]["registration_fingerprint32"], 16)
            and candidate["fingerprint"] == base["fingerprint"]
            and candidate["source_digest"] == base["source_digest"], "registration fingerprint/source digest")
    require(_producer_source_identity(context["source"]) == _producer_source_identity(source_manifest["source"]),
            "producer registration or source identity differs from accepted039")
    require([int(r[6]) for r in base["records"]] ==
            [int(r["stable_id"]) for r in source_manifest["source"]["surfaces"]], "accepted039 row order")
    require([int(r[6]) for r in candidate["records"]] ==
            [int(r["stable_id"]) for r in context["source"]["surfaces"]], "producer context row order")
    br = next((r for r in base["records"] if int(r[6]) == stable_id), None)
    cr = next((r for r in candidate["records"] if int(r[6]) == stable_id), None)
    require(br is not None and cr is not None, "selected stable ID missing")
    old, emitted = _row_slices(base, br), _row_slices(candidate, cr)
    old_surface, new_surface = _manifest_row(source_manifest, stable_id), _manifest_row(context, stable_id)
    op = new_surface.get("conforming_edge_refinement")
    require(isinstance(op, dict), "producer context has no conforming refinement operation")
    identity_fields = ("stable_id","member_id","member","member_sha256","label","layer","endpoint_source",
                       "body_bindings","matched_muscles","body_weight_count")
    for key in identity_fields:
        require(old_surface.get(key) == new_surface.get(key), f"selected source identity changed: {key}")
    binding_names = [x["myosim_body"] for x in old_surface["body_bindings"]]
    require(op.get("stable_surface_id") == stable_id and
            op.get("edge_local_vertex_ids") == list(requested_edge), "producer operation selection mismatch")
    require(op.get("route_body_binding_order") == binding_names, "route binding order differs from accepted039")
    require(old["binding_bytes"] == emitted["binding_bytes"], "producer changed selected binding records")
    require(old["layer"] == emitted["layer"], "producer changed selected row layer")
    require(old["vertex_count"] == op.get("old_vertex_count") and
            emitted["vertex_count"] == op.get("new_vertex_count") and
            len(old["local_faces"]) == op.get("old_face_count") and
            len(emitted["local_faces"]) == op.get("new_face_count"), "producer refinement counts")
    require(hashlib.sha256(old["vertex_bytes"]).hexdigest() == op.get("source_row_vertex_bytes_sha256")
            and len(old["vertex_bytes"]) == op.get("source_row_vertex_byte_count"),
            "producer original vertex-record hash")
    old_local_bytes = np.asarray(old["local_faces"], dtype="<u4").reshape(-1).tobytes()
    require(hashlib.sha256(old_local_bytes).hexdigest() == op.get("source_row_local_faces_sha256")
            and len(old_local_bytes) == op.get("source_row_local_face_index_byte_count"),
            "producer original local-face hash")
    require(emitted["vertex_bytes"][:len(old["vertex_bytes"])] == old["vertex_bytes"],
            "producer changed original selected vertex records")

    vertices, normals, weights = [], [], []
    for offset in range(0, len(old["vertex_bytes"]), 56):
        f = struct.unpack_from("<6f4I4f", old["vertex_bytes"], offset)
        vertices.append(list(f[:3])); normals.append(list(f[3:6]))
        dense, seen = [0.0]*old["binding_count"], set()
        for index, value in zip(f[6:10], f[10:14], strict=True):
            if value > 0.0:
                require(index < old["binding_count"] and index not in seen, "accepted039 sparse binding index")
                dense[index] = value; seen.add(index)
            else:
                require(index == 0xffffffff or index < old["binding_count"], "zero-weight sparse binding index")
        require(all(math.isfinite(x) and x >= 0 for x in dense) and abs(sum(dense)-1.0) <= 1e-6,
                "accepted039 source route weights")
        weights.append(dense)
    matrix, routes = op.get("global_source_mm_to_myosim_world_m"), op.get("authored_route_points")
    require(isinstance(matrix,list) and isinstance(routes,list), "producer source route context")
    global_vertices = [[sum(matrix[r][c]*v[c]*1000.0 for c in range(3))+matrix[r][3]
                        for r in range(3)] for v in vertices]
    replay = model._bodyparts_refine_conforming_route_edge(
        stable_id, requested_edge, vertices, normals, global_vertices, weights,
        old["local_faces"], matrix, binding_names, routes)
    new_v, new_n, _, new_w, new_faces, expected_op = replay
    for key,value in expected_op.items():
        require(op.get(key) == value, f"producer operation replay mismatch: {key}")
    require(op.get("member_id") == old_surface.get("member_id")
            and op.get("member") == old_surface.get("member")
            and op.get("source_obj_sha256") == old_surface.get("member_sha256"),
            "producer OBJ/member identity")
    require(emitted["local_faces"] == new_faces, "producer refined faces differ from helper replay")
    require(new_surface.get("vertex_count") == len(new_v) and new_surface.get("triangle_count") == len(new_faces),
            "producer manifest refined row counts")
    active = sorted(((value,index) for index,value in enumerate(new_w[-1]) if value > 1e-8), reverse=True)[:4]
    total = sum(value for value,_ in active)
    require(active and math.isfinite(total) and total > 0, "midpoint sparse weights")
    sparse_i = [index for _,index in active] + [0xffffffff]*(4-len(active))
    sparse_w = [value/total for value,_ in active] + [0.0]*(4-len(active))
    midpoint = struct.pack("<6f4I4f", *new_v[-1], *new_n[-1], *sparse_i, *sparse_w)
    expected_row_v = old["vertex_bytes"] + midpoint
    require(emitted["vertex_bytes"] == expected_row_v, "producer midpoint/vertex bytes differ from helper replay")

    new_records = base["records"].copy()
    vertex_parts, face_parts, unchanged = [], [], []
    vc = ic = 0
    for i,record in enumerate(base["records"]):
        sid = int(record[6]); row = _row_slices(base, record)
        if sid == stable_id:
            vertex_bytes, faces = expected_row_v, np.asarray(new_faces, dtype="<u4")
        else:
            vertex_bytes, faces = row["vertex_bytes"], np.asarray(row["local_faces"], dtype="<u4")
            unchanged.append(sid)
        new_records[i,2] = vc; new_records[i,3] = len(vertex_bytes)//56
        new_records[i,4] = ic; new_records[i,5] = faces.size
        vertex_parts.append(vertex_bytes); face_parts.append((faces+vc).astype("<u4").tobytes())
        vc += len(vertex_bytes)//56; ic += faces.size
    header = bytearray(base["raw"][:base["vertex_start"]])
    header[64:64+base["surface_count"]*32] = new_records.astype("<u4").tobytes()
    struct.pack_into("<II", header,20,vc,ic)
    output_raw = bytes(header)+b"".join(vertex_parts)+b"".join(face_parts)
    require(vc == base["vertex_count"] + 1 and ic == base["index_count"] + 6,
            "one-edge refinement must add exactly one vertex and two faces")
    require(len(output_raw)==64+base["surface_count"]*32+base["binding_count"]*36+vc*56+ic*4,
            "refined NHTISS byte ranges")
    require(base["raw"][base["binding_start"]:base["vertex_start"]]==
            output_raw[base["binding_start"]:base["vertex_start"]], "accepted039 full binding table changed")
    manifest = json.loads(json.dumps(source_manifest))
    manifest["payload"].update(file=source.name,sha256=hashlib.sha256(output_raw).hexdigest(),
                               bytes=len(output_raw),vertex_count=vc,index_count=ic)
    out_surface = _manifest_row(manifest,stable_id)
    out_surface["vertex_count"],out_surface["triangle_count"]=len(new_v),len(new_faces)
    out_surface["conforming_edge_refinement"]=op
    rb=out_surface.get("route_binding")
    if isinstance(rb,dict):
        rb["maximum_vertex_influences"]=max(int(rb.get("maximum_vertex_influences",0)),
                                             int(op["new_vertex_active_route_influence_count"]))
        rb["maximum_nearest_route_distance_m"]=max(float(rb.get("maximum_nearest_route_distance_m",0)),
                                                   float(op["new_vertex_max_nearest_route_distance_m"]))
        rb["conforming_refinement_midpoint_count"]=1
    manifest["source"]["conforming_edge_refinement_composition"]={
        "source_payload_path":str(source),"source_payload_sha256":base["sha256"],
        "source_manifest_path":str(source_manifest_path),"source_manifest_sha256":_surface_sha256(source_manifest_path),
        "context_manifest_path":str(context_manifest_path),"context_manifest_sha256":_surface_sha256(context_manifest_path),
        "context_payload_path":str(context_payload_path),"context_payload_sha256":candidate["sha256"],
        "stable_id":stable_id,"edge_local_vertex_ids":list(requested_edge),
        "helper_source_path":str(Path(model.__file__).resolve()),
        "helper_source_sha256":_surface_sha256(Path(model.__file__)),
        "unchanged_row_vertex_bytes_and_local_faces":unchanged,
        "binding_table_byte_exact":True,"physical_route_mass_and_force_state_unchanged":True}
    require({path: _surface_sha256(Path(path)) for path in start_hashes} == start_hashes,
            "source, producer context, or helper changed during composition")
    proof={"scope":"Explicit source-derived conforming edge refinement of one existing NHTISS row; native and anatomical admission remain separate.",
           "input_sha256":start_hashes,"inputs_unchanged":True,
           "stable_id":stable_id,"edge_local_vertex_ids":list(requested_edge),
           "unchanged_row_vertex_bytes_and_local_faces":unchanged,"binding_table_byte_exact":True,
           "payload_sha256":hashlib.sha256(output_raw).hexdigest(),"vertex_count":vc,"index_count":ic}
    return output_raw,manifest,proof


def compose_conforming_edge(source: Path, output: Path, context_manifest: Path,
                            stable_id: int, edge_vertex_ids: tuple[int,int]) -> dict:
    source, output=Path(source).resolve(),Path(output).resolve()
    require(stable_id>0 and len(edge_vertex_ids)==2 and edge_vertex_ids[0]!=edge_vertex_ids[1],
            "invalid conforming edge selection")
    output_raw,manifest,proof=_replay_conforming_edge(source,context_manifest,stable_id,edge_vertex_ids)
    output.mkdir(exist_ok=False)
    payload=output/source.name; payload.write_bytes(output_raw)
    manifest_path=output/source.with_suffix(".manifest.json").name
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    require(proof.get("inputs_unchanged") is True and
            all(_surface_sha256(Path(path)) == digest for path,digest in proof["input_sha256"].items()),
            "source, producer context, or helper changed before output was sealed")
    require(_surface_sha256(payload) == proof["payload_sha256"], "composed payload changed during write")
    proof["manifest_sha256"]=_surface_sha256(manifest_path)
    proof["composer_source_sha256"]=_surface_sha256(Path(__file__))
    (output/"report.json").write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n")
    return proof

def bind_anatomy_receipt(source_receipt: Path, payload: Path, output_receipt: Path) -> dict:
    """Bind this composition to the existing anatomy receipt for native launch."""
    source_receipt, payload = Path(source_receipt).resolve(), Path(payload).resolve()
    output_receipt = Path(output_receipt).resolve()
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    receipt = json.loads(source_receipt.read_text())
    require(receipt.get("schema") == "numi.human.resting-anatomy-receipt.v1", "anatomy receipt schema")
    owner = receipt["provenance"]["native_muscle_surfaces"]
    old = Path(owner["payload_path"])
    if not old.is_absolute():
        old = source_receipt.parent / old
    require(sha(old) == owner["sha256"], "prior anatomical muscle owner changed")
    manifest_path = payload.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    record = manifest["payload"]
    edge_composition = manifest["source"].get("conforming_edge_refinement_composition")
    if edge_composition is not None:
        require(edge_composition["source_payload_sha256"] == owner["sha256"],
                "conforming refinement source differs from prior anatomical muscle owner")
        source_path = Path(edge_composition["source_payload_path"]).resolve()
        context_manifest_path = Path(edge_composition["context_manifest_path"]).resolve()
        require(source_path == old.resolve() and sha(source_path) == owner["sha256"],
                "conforming refinement is not directly bound to accepted039 payload")
        accepted_manifest_path = source_path.with_suffix(".manifest.json")
        owner_manifest_path = Path(owner.get("manifest_path", accepted_manifest_path))
        if not owner_manifest_path.is_absolute():
            owner_manifest_path = source_receipt.parent / owner_manifest_path
        require(owner_manifest_path.resolve() == accepted_manifest_path.resolve()
                and owner.get("manifest_sha256") == sha(accepted_manifest_path),
                "accepted039 source manifest binding changed")
        require(sha(context_manifest_path) == edge_composition["context_manifest_sha256"],
                "conforming producer context changed")
        expected_raw, expected_manifest, replay_proof = _replay_conforming_edge(
            source_path, context_manifest_path, int(edge_composition["stable_id"]),
            tuple(int(x) for x in edge_composition["edge_local_vertex_ids"]),
        )
        require(payload.read_bytes() == expected_raw,
                "composed conforming payload differs from exact source/helper replay")
        require(manifest == expected_manifest,
                "composed conforming manifest differs from exact source/helper replay")
        composition = edge_composition
        changed_stable_ids = [int(edge_composition["stable_id"])]
        require(edge_composition["binding_table_byte_exact"] is True and
                edge_composition["physical_route_mass_and_force_state_unchanged"] is True,
                "conforming refinement changed physical ownership")
    else:
        composition = manifest["source"]["reference_attachment_composition"]
        require(composition["source_payload_sha256"] == owner["sha256"],
                "composition source differs from prior anatomical muscle owner")
        require(composition["binding_table_byte_exact"] is True and
                composition["physical_route_mass_and_force_state_unchanged"] is True,
                "composition changed physical ownership")
        changed_stable_ids = composition["changed_stable_ids"]
        require(set(changed_stable_ids).issubset({7, 8, 23}),
                "unsupported anatomical attachment replacements")
    require(record["sha256"] == sha(payload) and record["file"] == payload.name
            and record["bytes"] == payload.stat().st_size, "composed payload identity")
    old_raw, new_raw = old.read_bytes(), payload.read_bytes()
    old_header = struct.unpack_from("<8s6I32s", old_raw)
    new_header = struct.unpack_from("<8s6I32s", new_raw)
    require(old_header[:4] == new_header[:4] and old_header[6:] == new_header[6:],
            "anatomical binding or source identity changed")
    nr, nb = old_header[2:4]
    binding_start, binding_end = 64 + nr * 32, 64 + nr * 32 + nb * 36
    require(old_raw[binding_start:binding_end] == new_raw[binding_start:binding_end],
            "anatomical binding table changed")
    require(record["registration_fingerprint32"] == owner["registration_fingerprint32"],
            "anatomical registration changed")
    owner.update(payload_path=str(payload), sha256=record["sha256"],
                 manifest_path=str(manifest_path), manifest_sha256=sha(manifest_path),
                 surface_count=record["surface_count"], body_binding_count=record["binding_count"],
                 vertex_count=record["vertex_count"], index_count=record["index_count"])
    # Native cardiac descriptors must stay inside their receipt directory.
    # Reuse immutable bytes locally while retaining each exact content identity.
    shared_inputs = []
    common = receipt["provenance"].get("cardiac_geometry_binding", {}).get("common_field")
    if common:
        for key in ("map", "polynomials", "domain_boxes"):
            asset = Path(common[key]["path"])
            if not asset.is_absolute():
                asset = source_receipt.parent / asset
            require(sha(asset) == common[key]["sha256"], "cardiac input identity changed")
            destination = output_receipt.parent / asset.name
            require(not destination.exists(), "cardiac destination already exists")
            shared_inputs.append((asset.resolve(), destination))
            common[key]["path"] = asset.name
    anatomical_payload = Path(receipt["payload"]["path"])
    if not anatomical_payload.is_absolute():
        anatomical_payload = source_receipt.parent / anatomical_payload
    require(sha(anatomical_payload) == receipt["payload"]["sha256"], "anatomical payload changed")
    receipt["payload"]["path"] = str(anatomical_payload.resolve())
    binding_key = ("conforming_surface_refinement_binding" if edge_composition is not None
                   else "passive_attachment_composition_binding")
    receipt["provenance"][binding_key] = {
        "prior_receipt_path": str(source_receipt), "prior_receipt_sha256": sha(source_receipt),
        "composition_manifest_sha256": sha(manifest_path),
        "changed_stable_ids": changed_stable_ids,
        "scope": ("Explicit source-derived conforming surface refinement only; existing physical owners, "
                  "mass, forces, and tendon state are unchanged."
                  if edge_composition is not None else
                  "Passive attachment source binding only; physical owners and anatomical acceptance are unchanged.")
    }
    require(not output_receipt.exists(), "output anatomy receipt already exists")
    for source, destination in shared_inputs:
        try:
            os.link(source, destination)
        except OSError as error:
            if error.errno not in {errno.EXDEV, errno.EPERM}:
                raise
            shutil.copyfile(source, destination)
        require(sha(source) == sha(destination), "relocated cardiac input changed")
    with output_receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--row', nargs=3, action='append', metavar=('STABLE_ID', 'NPZ', 'REPORT'))
    parser.add_argument("--conforming-edge-context", type=Path,
                        help="opt-in producer context manifest for one route-bound source edge refinement")
    parser.add_argument("--conforming-edge", nargs=3, type=int,
                        metavar=("STABLE_ID", "VERTEX_A", "VERTEX_B"),
                        help="single explicit source edge to replay from the producer context")
    parser.add_argument("--anatomy-receipt", type=Path,
                        help="bind the composed surface rows in a new native anatomy launch receipt")
    args = parser.parse_args(argv)
    edge_mode = args.conforming_edge_context is not None or args.conforming_edge is not None
    if edge_mode:
        if args.row or args.conforming_edge_context is None or args.conforming_edge is None:
            parser.error("conforming edge mode requires only --conforming-edge-context and --conforming-edge")
        if args.conforming_edge[0] <= 0 or min(args.conforming_edge[1:]) < 0 or args.conforming_edge[1] == args.conforming_edge[2]:
            parser.error("conforming edge IDs are invalid")
    elif not args.row:
        parser.error("provide at least one --row replacement")
    try:
        if edge_mode:
            report = compose_conforming_edge(args.source, args.output, args.conforming_edge_context,
                                             args.conforming_edge[0], tuple(args.conforming_edge[1:]))
        else:
            report = compose(args.source, args.output, args.row)
        if args.anatomy_receipt is not None:
            bind_anatomy_receipt(args.anatomy_receipt, args.output / args.source.name,
                                 args.output / "resting-anatomy-receipt.json")
    except (OSError, ValueError, KeyError, struct.error) as error:
        parser.exit(2, f'{error}\n')
    print(json.dumps({key: report[key] for key in ('payload_sha256', 'manifest_sha256', 'vertex_count', 'index_count', 'binding_table_byte_exact')}))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

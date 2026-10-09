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

# This opt-in is intentionally pinned to the single reviewed biceps patch and
# the current 1216 anatomical parent. A different source or parent requires a
# new reviewed bridge rather than a broader accepted stable-ID set.
_BICEPS_SOURCE_CORRECTION_EXPECTED_PINS = {
    "source_base_payload": {
        "path": "/Users/n/numi-human-resting-evidence-20261005/passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue",
        "sha256": "b3d0381f73e05b04ed7ea1ba23eb6559073ad20aa0d9e2ad0f97a18fab3359bd",
    },
    "source_base_manifest": {
        "path": "/Users/n/numi-human-resting-evidence-20261005/passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json",
        "sha256": "82cdd937e0f2daf0a8704fb21246353c38148527f8602d674ac865f935264dde",
    },
    "source_candidate_payload": {
        "path": "/Users/n/numi-human-retained-delivery-20261009/passive-biceps-micro-overlap-1225/attempt-002/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue",
        "sha256": "73272780f92e113cccd2dc66b3005384cbc68afc86b4f63458e697bf75a6f375",
    },
    "source_candidate_manifest": {
        "path": "/Users/n/numi-human-retained-delivery-20261009/passive-biceps-micro-overlap-1225/attempt-002/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json",
        "sha256": "38b7be04d77ef2a5479cf705970450d6097d6b69e4e2f215f351c0c3d17a7b09",
    },
    "source_candidate_report": {
        "path": "/Users/n/numi-human-retained-delivery-20261009/passive-biceps-micro-overlap-1225/attempt-002/source-candidate-audit.json",
        "sha256": "f610927408d87224fbe10dfd518de4c02fab87cd95e4accac0b547932b2fe57e",
    },
    "source_generator_script": {
        "path": "/Users/n/numi-human-retained-delivery-20261009/passive-biceps-micro-overlap-1225/attempt-002/run_inferred_opening.py",
        "sha256": "0aa6e136e9d6a0b322326b08f3d05564ad6164e94c94a53b5c70f256844d71bd",
    },
    "direct_parent_payload": {
        "path": "/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue",
        "sha256": "1cd0c3d5bd1ff6d163f6544be4d0fa169e8bf696bf6c729ef0b18879b0856c48",
    },
    "direct_parent_manifest": {
        "path": "/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json",
        "sha256": "f052eff736f040f2f6ca1a1a99d2939ca0ed5c0bead055dfb39d3141f05a3cac",
    },
}

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError('passive attachment composition: ' + message)

def compose(source: Path, output: Path, replacements: list[tuple[int, Path, Path]], *,
            biceps_source_correction: dict | None = None) -> dict:
    import numpy as np
    (source, output) = (Path(source).resolve(), Path(output).resolve())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    T = source
    M = T.with_suffix('.manifest.json')
    inputs = {str(T): sha(T), str(M): sha(M)}
    repairs = {}
    require(bool(replacements), 'at least one passive attachment repair is required')
    replacement_ids = [int(item[0]) for item in replacements]
    biceps_correction = None
    if any(sid not in (7, 8, 23) for sid in replacement_ids):
        require(set(replacement_ids) == {103, 104} and len(replacement_ids) == 2
                and biceps_source_correction is not None,
                'only distinct passive attachment rows 7, 8, and 23 are allowed, except a proof-bound paired biceps 103/104 source correction')
        biceps_correction = _verify_biceps_source_correction(T, biceps_source_correction)
        inputs.update(biceps_correction['input_sha256'])
    else:
        require(biceps_source_correction is None,
                'biceps source correction proof may only accompany the paired 103/104 rows')
    for (sid, z, report) in replacements:
        sid = int(sid)
        if (sid not in (7, 8, 23) and not (biceps_correction is not None and sid in (103, 104))) or sid in repairs:
            raise ValueError('Only distinct passive attachment rows7,8,23 or a proof-bound biceps pair103,104 are allowed')
        (z, report) = (Path(z), Path(report))
        if biceps_correction is not None:
            expected_patch = biceps_correction["row_patch_npz"].get(str(sid))
            require(isinstance(expected_patch, dict)
                    and z.resolve() == Path(expected_patch["path"]).resolve()
                    and sha(z) == expected_patch["sha256"],
                    "replacement row does not match its proof-bound biceps NPZ")
            require(report.resolve() == Path(biceps_correction["source_candidate_report_path"]).resolve()
                    and sha(report) == biceps_correction["source_candidate_report_sha256"],
                    "replacement report does not match the proof-bound biceps source report")
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
    if biceps_correction is None:
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
        if biceps_correction is None:
            row['reference_attachment_reconstruction'] = {**proof, 'scope': 'Passive source-derived reference inspection surface. Original anatomical identity/laterality, named attachment relationships, MyoSim force route and compliant tendon state remain their original owners. Not a measured-person reconstruction.', 'prior_registration_metadata': 'Retained above as upstream provenance; this reconstruction supersedes the listed source surface geometry.'}
    if biceps_correction is None:
        manifest['source']['reference_attachment_composition'] = {'source_payload_sha256': inputs[str(T)], 'changed_stable_ids': sorted(repairs), 'unchanged_row_vertex_bytes_and_local_faces': unchanged, 'binding_table_byte_exact': True, 'physical_route_mass_and_force_state_unchanged': True}
    else:
        # This is an additive operation on the current receipted payload. Keep
        # the older 7/8 attachment composition and 64/23 operation lineage intact.
        biceps_correction['composed_parent_payload_sha256'] = inputs[str(T)]
        biceps_correction['composed_parent_manifest_path'] = str(M)
        biceps_correction['composed_parent_manifest_sha256'] = inputs[str(M)]
        biceps_correction['changed_stable_ids'] = [103, 104]
        biceps_correction['unchanged_row_vertex_bytes_and_local_faces'] = unchanged
        manifest['source']['biceps_source_preserving_correction'] = biceps_correction
        for surface in manifest['source']['surfaces']:
            sid = int(surface['stable_id'])
            if sid in (103, 104):
                surface['biceps_microcomponent_source_correction'] = {
                    'source_experiment_report_sha256': biceps_correction['source_candidate_report_sha256'],
                    'source_candidate_payload_sha256': biceps_correction['source_candidate_payload_sha256'],
                    'direct_parent_row_identity_verified': True,
                    'scope': 'Bounded inferred source-geometry correction only; unchanged bindings, weights, faces, and force routes.'}
    PM = output / M.name
    PM.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    require(inputs == {p: sha(p) for p in inputs}, 'candidate composition invariant')
    proof = {'scope': 'Existing NHTISS4 ABI5 passive geometry composition; final native and anatomical admission separate.', 'input_sha256': inputs, 'inputs_unchanged': True, 'changed_rows': proofs, 'unchanged_row_vertex_bytes_and_local_faces': unchanged, 'binding_table_byte_exact': True, 'payload_sha256': sha(P), 'manifest_sha256': sha(PM), 'vertex_count': vcur, 'index_count': icur}
    if biceps_correction is not None:
        proof['biceps_source_preserving_correction'] = biceps_correction
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


def _row_local_equal(a: dict, b: dict) -> bool:
    """Compare row-local identity while allowing parent arrays to be repacked."""
    keys = ("binding_count", "vertex_count", "index_count", "stable_id", "layer",
            "binding_bytes", "vertex_bytes", "local_faces")
    return all(a[key] == b[key] for key in keys)


def _biceps_row_arrays(row: dict) -> dict:
    import numpy as np
    vb = row["vertex_bytes"]
    n = int(row["vertex_count"])
    v6 = np.ndarray((n, 6), dtype="<f4", buffer=vb, offset=0, strides=(56, 4)).copy()
    bi = np.ndarray((n, 4), dtype="<u4", buffer=vb, offset=24, strides=(56, 4)).copy()
    weights = np.ndarray((n, 4), dtype="<f4", buffer=vb, offset=40, strides=(56, 4)).copy()
    return {"vertices6": v6, "binding_indices": bi, "weights": weights,
            "faces": np.asarray(row["local_faces"], dtype="<u4")}


def _verify_biceps_source_correction(parent_payload: Path, correction: dict) -> dict:
    """Verify the one permitted 103/104 patch is disjoint from current-parent edits.

    This is intentionally narrower than the existing 7/8/23 attachment path:
    the biceps patch must replay from the pinned 039 source rows, which must be
    byte-identical to rows 103/104 in the immediate parent. The source report is
    evidence for the geometry experiment; it does not qualify accepted poses.
    """
    import numpy as np
    parent_payload = Path(parent_payload).resolve()
    require(isinstance(correction, dict)
            and correction.get("schema") == "numi.human.biceps-source-preserving-row-correction.v1",
            "biceps source-correction proof schema")
    names = ("source_base_payload_path", "source_candidate_payload_path",
             "source_candidate_report_path", "source_generator_script_path")
    paths = {key: Path(correction[key]).resolve() for key in names}
    declared_base_manifest_path = correction.get("source_base_manifest_path")
    declared_candidate_manifest_path = correction.get("source_candidate_manifest_path")
    parent_manifest_path, parent_manifest = _payload_manifest(parent_payload)
    base_manifest_path, base_manifest = _payload_manifest(paths["source_base_payload_path"])
    candidate_manifest_path, candidate_manifest = _payload_manifest(paths["source_candidate_payload_path"])
    require(declared_base_manifest_path is None
            or Path(declared_base_manifest_path).resolve() == base_manifest_path.resolve(),
            "declared source-base manifest path differs from payload manifest")
    require(declared_candidate_manifest_path is None
            or Path(declared_candidate_manifest_path).resolve() == candidate_manifest_path.resolve(),
            "declared source-candidate manifest path differs from payload manifest")
    report_path = paths["source_candidate_report_path"]
    script_path = paths["source_generator_script_path"]
    expected_pins = correction.get("expected_pins")
    require(expected_pins == _BICEPS_SOURCE_CORRECTION_EXPECTED_PINS,
            "biceps source/report/script/parent pins are not the reviewed exact set")
    actual_pin_paths = {
        "source_base_payload": paths["source_base_payload_path"],
        "source_base_manifest": base_manifest_path,
        "source_candidate_payload": paths["source_candidate_payload_path"],
        "source_candidate_manifest": candidate_manifest_path,
        "source_candidate_report": report_path,
        "source_generator_script": script_path,
        "direct_parent_payload": parent_payload,
        "direct_parent_manifest": parent_manifest_path,
    }
    for key, path in actual_pin_paths.items():
        pin = expected_pins[key]
        require(Path(pin["path"]).resolve() == path.resolve()
                and _surface_sha256(path) == pin["sha256"],
                "biceps expected pin mismatch: " + key)
    report = json.loads(report_path.read_text())
    require(report.get("schema") == "numi.human.passive-biceps-microcomponent-inferred-opening-experiment.v1"
            and report.get("qualification") == "bounded_source_geometry_experiment_only_unadmitted",
            "biceps source experiment is not the retained bounded candidate")
    require(report.get("accepted_pose_forward_verification", {}).get("status") == "not_run",
            "biceps source experiment report changed scope")
    require(report.get("method", {}).get("selected_smallest_passing_opening_um") == 0.5
            and report.get("method", {}).get("component_geometry_inference_not_measured") is True
            and report.get("method", {}).get("preserve_anatomy_ids_bindings_weights_faces_and_components") is True,
            "biceps source operation is not the pinned 0.5um inferred patch")

    parent_sha = _surface_sha256(parent_payload)
    base_sha = _surface_sha256(paths["source_base_payload_path"])
    candidate_sha = _surface_sha256(paths["source_candidate_payload_path"])
    report_sha = _surface_sha256(report_path)
    script_sha = _surface_sha256(script_path)
    require(report["candidate"].get("payload_path") == str(paths["source_candidate_payload_path"])
            and report["candidate"].get("payload_sha256") == candidate_sha
            and report["candidate"].get("manifest_path") == str(candidate_manifest_path)
            and report["candidate"].get("manifest_sha256") == _surface_sha256(candidate_manifest_path),
            "biceps source report candidate payload binding")
    pins_before = report.get("input_pins_before")
    pins_after = report.get("input_pins_after")
    require(isinstance(pins_before, dict) and pins_before == pins_after,
            "biceps source report input pins changed during its experiment")
    require(pins_before.get(str(paths["source_base_payload_path"])) == base_sha
            and pins_before.get(str(base_manifest_path)) == _surface_sha256(base_manifest_path),
            "biceps source report does not bind its source payload and manifest")
    for path, expected_sha in pins_before.items():
        pinned_path = Path(path)
        require(pinned_path.is_file() and _surface_sha256(pinned_path) == expected_sha,
                "biceps source experiment input pin changed: " + path)

    current_lineage = parent_manifest.get("source", {}).get("conforming_edge_refinement_composition")
    require(isinstance(current_lineage, dict)
            and current_lineage.get("source_payload_sha256") == base_sha,
            "current parent does not descend from the biceps patch source")
    require(parent_manifest.get("source", {}).get("biceps_source_preserving_correction") is None,
            "biceps source patch already exists in immediate parent")
    require(base_manifest.get("payload", {}).get("sha256") == base_sha
            and candidate_manifest.get("payload", {}).get("sha256") == candidate_sha,
            "biceps base/candidate manifest payload identity")
    require(base_manifest.get("source", {}).get("myosim_source_archive_sha256")
            == candidate_manifest.get("source", {}).get("myosim_source_archive_sha256"),
            "biceps source archive identity changed")
    candidate_method = candidate_manifest.get("source", {}).get("bounded_inferred_biceps_microcomponent_opening")
    require(isinstance(candidate_method, dict)
            and candidate_method.get("candidate_is_unadmitted") is True
            and candidate_method.get("nhtiss_records_binding_table_faces_and_vertex_weights_preserved") is True
            and candidate_method.get("changed_stable_ids") == [103, 104],
            "biceps candidate manifest does not bind the two-row inferred operation")
    require(report["candidate"].get("NHTISS_binding_table_byte_exact") is True
            and report["candidate"].get("non-target_stable_rows_byte_exact") is True
            and report["candidate"].get("face_rows_added_removed_reordered") is False
            and report["candidate"].get("other_bytes_byte_exact") is True,
            "biceps source experiment candidate scope changed")

    base = _read_nhtiss4(paths["source_base_payload_path"])
    candidate = _read_nhtiss4(paths["source_candidate_payload_path"])
    parent = _read_nhtiss4(parent_payload)
    require(base["surface_count"] == candidate["surface_count"] == parent["surface_count"] == 150
            and base["binding_count"] == candidate["binding_count"] == parent["binding_count"]
            and base["vertex_count"] == candidate["vertex_count"]
            and base["index_count"] == candidate["index_count"]
            and base["fingerprint"] == candidate["fingerprint"] == parent["fingerprint"]
            and base["source_digest"] == candidate["source_digest"] == parent["source_digest"],
            "biceps composition NHTISS identity mismatch")

    base_rows = {_row_slices(base, row)["stable_id"]: _row_slices(base, row) for row in base["records"]}
    patch_rows = {_row_slices(candidate, row)["stable_id"]: _row_slices(candidate, row) for row in candidate["records"]}
    parent_rows = {_row_slices(parent, row)["stable_id"]: _row_slices(parent, row) for row in parent["records"]}
    require(set(base_rows) == set(patch_rows) == set(parent_rows)
            and len(base_rows) == 150, "biceps source stable-row inventory changed")
    operations = report.get("selected_operation_rows")
    require(isinstance(operations, list) and {int(op.get("stable_id", -1)) for op in operations} == {103, 104}
            and len(operations) == 2, "biceps source operation rows")
    operation_by_id = {int(op["stable_id"]): op for op in operations}
    require(candidate_method.get("operation_rows") == operations,
            "biceps source candidate manifest/report operation rows differ")
    expected_members = {103: "FJ1478", 104: "FJ1478M"}
    for sid in base_rows:
        br, cr = base_rows[sid], patch_rows[sid]
        if sid not in (103, 104):
            require(_row_local_equal(br, cr),
                    "biceps source experiment changed an unrelated stable row")
            continue
        pr = parent_rows[sid]
        require(_row_local_equal(br, pr),
                "current parent biceps row differs from its pre-64 source row")
        op = operation_by_id[sid]
        expected_member = expected_members[sid]
        base_surface = _manifest_row(base_manifest, sid)
        candidate_surface = _manifest_row(candidate_manifest, sid)
        require(base_surface.get("member_id") == expected_member
                and candidate_surface.get("member_id") == expected_member
                and base_surface.get("member_sha256") == op.get("source_member_sha256")
                and candidate_surface.get("member_sha256") == op.get("source_member_sha256"),
                "biceps source member identity changed")
        require(op.get("member_id") == expected_members[sid]
                and op.get("apex_signed_source_axis_displacements_requested_um") == [-0.5, 0.5]
                and op.get("apex_signed_source_axis_displacements_mm") == [-0.0005, 0.0005]
                and op.get("bindings_and_weights_changed") is False
                and op.get("triangles_added_removed_reindexed") is False
                and op.get("vertex_rows_added_removed_reindexed") is False
                and op.get("component_count_before_after") == [2, 2],
                "biceps source row operation details changed")
        require(br["binding_count"] == cr["binding_count"]
                and br["vertex_count"] == cr["vertex_count"]
                and br["index_count"] == cr["index_count"]
                and br["binding_bytes"] == cr["binding_bytes"]
                and br["local_faces"] == cr["local_faces"],
                "biceps source correction changed bindings, weights, or face topology")
        base_fields, patch_fields = _biceps_row_arrays(br), _biceps_row_arrays(cr)
        require(base_fields["binding_indices"].tobytes(order="C")
                == patch_fields["binding_indices"].tobytes(order="C")
                and base_fields["weights"].tobytes(order="C")
                == patch_fields["weights"].tobytes(order="C"),
                "biceps source correction changed vertex route bindings or weights")
        expected_delta = {int(k): float(v) for k, v in op.get("actual_float32_apex_displacements_m", {}).items()}
        groups = op.get("opposite_apex_source_vertex_groups")
        require(isinstance(groups, list) and len(groups) == 2
                and set(int(x) for group in groups for x in group) == set(expected_delta),
                "biceps source correction changed its declared apex vertices")
        require(set(expected_delta) and all(0 <= i < br["vertex_count"] for i in expected_delta),
                "biceps source correction apex index range")
        delta = patch_fields["vertices6"][:, :3].astype(np.float64) - base_fields["vertices6"][:, :3].astype(np.float64)
        changed_positions = set(int(i) for i in np.flatnonzero(np.any(delta != 0.0, axis=1)))
        require(changed_positions == set(expected_delta),
                "biceps source correction moved undeclared positions")
        unchanged_positions = sorted(set(range(br["vertex_count"])) - changed_positions)
        if unchanged_positions:
            require(base_fields["vertices6"][unchanged_positions, :3].tobytes(order="C")
                    == patch_fields["vertices6"][unchanged_positions, :3].tobytes(order="C"),
                    "biceps source correction changed undeclared position bits")
        for group_i, group in enumerate(groups):
            sign = -1.0 if group_i == 0 else 1.0
            for vertex in group:
                vertex = int(vertex)
                require(base_fields["vertices6"][vertex, :2].tobytes()
                        == patch_fields["vertices6"][vertex, :2].tobytes()
                        and delta[vertex, 2] == expected_delta[vertex]
                        and np.sign(delta[vertex, 2]) == sign
                        and abs(delta[vertex, 2]) <= 0.5e-6,
                        "biceps source correction apex displacement differs from its bounded operation")
        closed = set(int(v) for v in op.get("closed_microcomponent_vertex_ids", []))
        require(closed and closed.issubset(set(range(br["vertex_count"])))
                and np.isfinite(patch_fields["vertices6"]).all(),
                "biceps source correction microcomponent or vertex values invalid")
        base_norm = base_fields["vertices6"][:, 3:]
        patch_norm = patch_fields["vertices6"][:, 3:]
        changed_normals = set(int(i) for i in np.flatnonzero(np.any(base_norm != patch_norm, axis=1)))
        require(changed_normals.issubset(closed),
                "biceps source correction changed normals outside its microcomponent")

    patch_npz = correction.get("row_patch_npz")
    require(isinstance(patch_npz, dict) and set(patch_npz) == {"103", "104"},
            "biceps source correction requires exactly two row patch arrays")
    input_sha256 = {str(paths[key]): _surface_sha256(paths[key]) for key in names}
    input_sha256[str(base_manifest_path)] = _surface_sha256(base_manifest_path)
    input_sha256[str(candidate_manifest_path)] = _surface_sha256(candidate_manifest_path)
    input_sha256.update({str(Path(path).resolve()): expected_sha
                         for path, expected_sha in pins_before.items()})
    for sid_text, entry in patch_npz.items():
        path = Path(entry["path"]).resolve()
        digest = _surface_sha256(path)
        require(digest == entry.get("sha256"), "biceps row patch array hash changed")
        input_sha256[str(path)] = digest
        sid = int(sid_text)
        with np.load(path, allow_pickle=False) as stored:
            arrays = {key: np.asarray(stored[key]) for key in ("vertices6", "binding_indices", "weights", "faces")}
        expected = _biceps_row_arrays(patch_rows[sid])
        require(all(arrays[key].dtype == expected[key].dtype
                    and arrays[key].shape == expected[key].shape
                    and arrays[key].tobytes(order="C") == expected[key].tobytes(order="C")
                    for key in arrays),
                "biceps row patch arrays do not exactly match the pinned full candidate")
    input_sha256[str(paths["source_base_payload_path"])] = base_sha
    input_sha256[str(parent_manifest_path)] = _surface_sha256(parent_manifest_path)
    return {
        "schema": "numi.human.biceps-source-preserving-row-correction.v1",
        "source_base_payload_path": str(paths["source_base_payload_path"]),
        "source_base_payload_sha256": base_sha,
        "source_base_manifest_path": str(base_manifest_path),
        "source_base_manifest_sha256": _surface_sha256(base_manifest_path),
        "source_candidate_payload_path": str(paths["source_candidate_payload_path"]),
        "source_candidate_payload_sha256": candidate_sha,
        "source_candidate_manifest_path": str(candidate_manifest_path),
        "source_candidate_manifest_sha256": _surface_sha256(candidate_manifest_path),
        "source_candidate_report_path": str(report_path),
        "source_candidate_report_sha256": report_sha,
        "source_generator_script_path": str(script_path),
        "source_generator_script_sha256": script_sha,
        "expected_pins": expected_pins,
        "source_experiment_accepted_pose_status": "not_run",
        "direct_parent_payload_path": str(parent_payload),
        "direct_parent_payload_sha256": parent_sha,
        "direct_parent_manifest_path": str(parent_manifest_path),
        "direct_parent_manifest_sha256": _surface_sha256(parent_manifest_path),
        "row_patch_npz": {sid: {"path": str(Path(entry["path"]).resolve()),
                                 "sha256": input_sha256[str(Path(entry["path"]).resolve())]}
                          for sid, entry in patch_npz.items()},
        "changed_stable_ids": [103, 104],
        "parent_rows_103_104_byte_exact_to_source": True,
        "candidate_other_rows_byte_exact_to_source": True,
        "candidate_binding_weights_faces_preserved": True,
        "input_sha256": input_sha256,
    }


def _verify_biceps_composed_child(parent_payload: Path, child_payload: Path, correction: dict) -> None:
    import numpy as np
    parent = _read_nhtiss4(parent_payload)
    child = _read_nhtiss4(child_payload)
    require(parent["raw"][:64] == child["raw"][:64]
            and parent["surface_count"] == child["surface_count"]
            and parent["binding_count"] == child["binding_count"]
            and parent["vertex_count"] == child["vertex_count"]
            and parent["index_count"] == child["index_count"],
            "biceps child changed NHTISS header or inventory")
    require(parent["raw"][64:parent["vertex_start"]] == child["raw"][64:child["vertex_start"]]
            and parent["raw"][parent["index_start"]:] == child["raw"][child["index_start"]:],
            "biceps child changed records, body bindings, or indexed topology")
    parent_rows = {_row_slices(parent, row)["stable_id"]: _row_slices(parent, row) for row in parent["records"]}
    child_rows = {_row_slices(child, row)["stable_id"]: _row_slices(child, row) for row in child["records"]}
    require(set(parent_rows) == set(child_rows), "biceps child row inventory changed")
    for sid in parent_rows:
        if sid not in (103, 104):
            require(parent_rows[sid]["vertex_bytes"] == child_rows[sid]["vertex_bytes"],
                    "biceps child changed a non-target row")
            continue
        patch = correction["row_patch_npz"][str(sid)]
        with np.load(patch["path"], allow_pickle=False) as stored:
            arrays = {key: np.asarray(stored[key]) for key in ("vertices6", "binding_indices", "weights")}
        n = len(arrays["vertices6"])
        expected = bytearray(n * 56)
        np.ndarray((n, 6), dtype="<f4", buffer=expected, offset=0, strides=(56, 4))[:] = arrays["vertices6"]
        np.ndarray((n, 4), dtype="<u4", buffer=expected, offset=24, strides=(56, 4))[:] = arrays["binding_indices"]
        np.ndarray((n, 4), dtype="<f4", buffer=expected, offset=40, strides=(56, 4))[:] = arrays["weights"]
        require(child_rows[sid]["vertex_bytes"] == bytes(expected)
                and parent_rows[sid]["binding_bytes"] == child_rows[sid]["binding_bytes"]
                and parent_rows[sid]["local_faces"] == child_rows[sid]["local_faces"],
                "biceps child target row does not equal its exact source patch")


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
    biceps_composition = manifest["source"].get("biceps_source_preserving_correction")
    if biceps_composition is not None:
        require(biceps_composition.get("changed_stable_ids") == [103, 104]
                and biceps_composition.get("composed_parent_payload_sha256") == owner["sha256"],
                "biceps child is not bound to the immediate anatomical parent")
        source_path = old.resolve()
        require(Path(biceps_composition["direct_parent_payload_path"]).resolve() == source_path
                and sha(source_path) == owner["sha256"],
                "biceps direct parent payload identity changed")
        accepted_manifest_path = Path(owner.get("manifest_path", source_path.with_suffix(".manifest.json")))
        if not accepted_manifest_path.is_absolute():
            accepted_manifest_path = source_receipt.parent / accepted_manifest_path
        require(accepted_manifest_path.resolve() == Path(biceps_composition["composed_parent_manifest_path"]).resolve()
                and owner.get("manifest_sha256") == biceps_composition["composed_parent_manifest_sha256"]
                and sha(accepted_manifest_path) == owner.get("manifest_sha256"),
                "biceps direct parent manifest identity changed")
        replay = _verify_biceps_source_correction(source_path, biceps_composition)
        _verify_biceps_composed_child(source_path, payload, biceps_composition)
        require(biceps_composition.get("direct_parent_payload_sha256") == owner["sha256"]
                and biceps_composition.get("source_candidate_report_sha256") == replay["source_candidate_report_sha256"]
                and biceps_composition.get("source_candidate_payload_sha256") == replay["source_candidate_payload_sha256"],
                "biceps direct-child replay identity changed")
        composition = biceps_composition
        changed_stable_ids = [103, 104]
        require(biceps_composition.get("parent_rows_103_104_byte_exact_to_source") is True
                and biceps_composition.get("candidate_other_rows_byte_exact_to_source") is True
                and biceps_composition.get("candidate_binding_weights_faces_preserved") is True,
                "biceps source-preserving correction proof incomplete")
    elif edge_composition is not None:
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
    binding_key = ("biceps_source_preserving_correction_binding" if biceps_composition is not None
                   else "conforming_surface_refinement_binding" if edge_composition is not None
                   else "passive_attachment_composition_binding")
    if biceps_composition is not None:
        require(binding_key not in receipt["provenance"],
                "anatomy receipt already contains this biceps correction provenance binding")
    receipt["provenance"][binding_key] = {
        "prior_receipt_path": str(source_receipt), "prior_receipt_sha256": sha(source_receipt),
        "composition_manifest_sha256": sha(manifest_path),
        "changed_stable_ids": changed_stable_ids,
        "scope": ("Explicit source-derived conforming surface refinement only; existing physical owners, "
                  "mass, forces, and tendon state are unchanged."
                  if edge_composition is not None and biceps_composition is None else
                  "Proof-bound biceps source-geometry patch composed as a direct child of this receipt; anatomy acceptance remains separate."
                  if biceps_composition is not None else
                  "Passive attachment source binding only; physical owners and anatomical acceptance are unchanged.")
    }
    if biceps_composition is not None:
        receipt["provenance"][binding_key]["source_preserving_correction"] = {
            "schema": biceps_composition["schema"],
            "direct_parent_payload_sha256": biceps_composition["direct_parent_payload_sha256"],
            "source_base_payload_sha256": biceps_composition["source_base_payload_sha256"],
            "source_candidate_payload_sha256": biceps_composition["source_candidate_payload_sha256"],
            "source_candidate_report_sha256": biceps_composition["source_candidate_report_sha256"],
            "source_generator_script_sha256": biceps_composition["source_generator_script_sha256"],
            "accepted_pose_forward_status": biceps_composition["source_experiment_accepted_pose_status"],
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

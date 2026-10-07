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
    composition = manifest["source"]["reference_attachment_composition"]
    require(composition["source_payload_sha256"] == owner["sha256"],
            "composition source differs from prior anatomical muscle owner")
    require(record["sha256"] == sha(payload) and record["file"] == payload.name
            and record["bytes"] == payload.stat().st_size, "composed payload identity")
    require(composition["binding_table_byte_exact"] is True and
            composition["physical_route_mass_and_force_state_unchanged"] is True,
            "composition changed physical ownership")
    require(set(composition["changed_stable_ids"]).issubset({7, 8, 23}),
            "unsupported anatomical attachment replacements")
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
    receipt["provenance"]["passive_attachment_composition_binding"] = {
        "prior_receipt_path": str(source_receipt), "prior_receipt_sha256": sha(source_receipt),
        "composition_manifest_sha256": sha(manifest_path),
        "changed_stable_ids": composition["changed_stable_ids"],
        "scope": "Passive attachment source binding only; physical owners and anatomical acceptance are unchanged."
    }
    require(not output_receipt.exists(), "output anatomy receipt already exists")
    for source, destination in shared_inputs:
        try:
            os.link(source, destination)
        except OSError as error:
            if error.errno != errno.EXDEV:
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
    parser.add_argument('--row', nargs=3, action='append', metavar=('STABLE_ID', 'NPZ', 'REPORT'), required=True)
    parser.add_argument("--anatomy-receipt", type=Path,
                        help="bind the composed passive rows in a new native anatomy launch receipt")
    args = parser.parse_args(argv)
    try:
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

"""Compose source-derived passive attachments in the existing NHTISS4 ABI5.

The generic replacement path is limited to stable IDs 7, 8 and 23. Separate
proof-bound branches admit only the reviewed 103/104 biceps and 27/28 FHL row
pairs. This is asset preparation; the native simulation retains mechanics,
mass and tendon state. Composition does not establish anatomical admission.
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


# This proof-bound branch admits only the regenerated FHL source-seam rows 27/28
# as a direct child of the current 7b23 payload. It does not widen the generic
# passive-attachment allowlist and does not imply native/anatomical admission.
_FHL_SOURCE_SEAM_CORRECTION_EXPECTED_PINS = {
    "direct_parent_payload": {"path": "/Users/n/numi-human-retained-delivery-20261009/passive-biceps-micro-overlap-1225/compose-current-1cd-attempt003/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue", "sha256": "7b23d0daf2eb73221944389716d01c8d9f2a9ebfb86dd815f605c8a45e54bbc9"},
    "direct_parent_manifest": {"path": "/Users/n/numi-human-retained-delivery-20261009/passive-biceps-micro-overlap-1225/compose-current-1cd-attempt003/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json", "sha256": "945d53ddf24c652af4b55e91f21e27e65754e89c5d6c027008e4196c9f2a3f08"},
    "source_subset_payload": {"path": "/Users/n/numi-human-retained-delivery-20261009/source-seam-connectivity-1247/regen-4rows-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue", "sha256": "e13ac1065e70b63fb1dea68493d9023810507a7a13182d340142f67be4d8634a"},
    "source_subset_manifest": {"path": "/Users/n/numi-human-retained-delivery-20261009/source-seam-connectivity-1247/regen-4rows-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json", "sha256": "7eeed2dba8b43b6cfa4bd412da4c9e3fc2ffc4652d6adeb2453da801298e2a9e"},
    "row_comparison_report": {"path": "/Users/n/numi-human-retained-delivery-20261009/source-seam-connectivity-1247/regen-4rows-001/regenerated-row-comparison-004.json", "sha256": "91c10218da0d1b0384ae5cc09a754a89ecb4eccab40cf1c5485bf2b0046861b0"},
    "self_audit_report": {"path": "/Users/n/numi-human-retained-delivery-20261009/source-seam-connectivity-1247/regen-4rows-001/regenerated-row-exact-f32-self-audit-001.json", "sha256": "b9d21241947453de05855e215f92e11f966a670ac38100cc6ac4c3f87b9b62f8"},
    "source_topology_report": {"path": "/Users/n/numi-human-retained-delivery-20261009/source-seam-connectivity-1247/source-selection-comparison.json", "sha256": "4a62be2006a8879aea849553f619e793992ba72078c86f2edb4e85be1fb6412b"},
    "row_patch_preparation_report": {"path": "/Users/n/numi-human-retained-delivery-20261009/source-seam-connectivity-1247/regen-4rows-001/fhl-row-patches-001/row-patch-preparation.json", "sha256": "b3e8980a79108a1d119b72b44ebc600054466b129a53d9a1a757cce4ceafc293"},
    "source_declaration": {"path": "/Users/n/numi-human-retained-delivery-20261009/source-seam-connectivity-1247/regen-4rows-001/declaration.json", "sha256": "33be926987729d2a9e816b06c5012a1e7470697f19a324e246caacd35a3182a3"},
    "row_patch_27": {"path": "/Users/n/numi-human-retained-delivery-20261009/source-seam-connectivity-1247/regen-4rows-001/fhl-row-patches-001/row-27.npz", "sha256": "94b6fcd93aa494cb583a2d8452830c5af51d9bd0a9b479646e5ea4a3a46ebae7"},
    "row_patch_28": {"path": "/Users/n/numi-human-retained-delivery-20261009/source-seam-connectivity-1247/regen-4rows-001/fhl-row-patches-001/row-28.npz", "sha256": "dc4763af332f0aaea9c1e48f05c7da97887d10a6fbbbf6e7992409fe9f31154e"},
    "registration": {"path": "/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Docs/media/skin-patella-rebase-source-preflight-20261003/registration.json", "sha256": "b1b410ad6d4ac8c0c95fd0c3e10f655b24c890d5767a5c377cf78e28ef598f8e"},
}
_FHL_SOURCE_DECLARATION_INPUTS = {
    "/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid": "2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4",
    "/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-muscle-reference.nhmyo": "e5bb8a8168706bb3b23cf42b1d849659569ba2e6027bcc85729c3110fdbb8c4b",
    "/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-reference.manifest.json": "844d05330104a43f6c45867020f2abd493adec35d6e2f8f90fc636ee9bec04e7",
    "/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Docs/media/skin-patella-rebase-source-preflight-20261003/registration.json": "b1b410ad6d4ac8c0c95fd0c3e10f655b24c890d5767a5c377cf78e28ef598f8e",
    "/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/nhtiss-source-prep-20261005/project/Sources/isa_BP3D_4.0_obj_99.zip": "40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e",
    "/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/nhtiss-source-prep-20261005/project/Sources/isa_element_parts.txt": "a3de74423f943b0d724ae8f59b3a817f87c423a544f8db98113b1980817cbeaf",
    "/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/nhtiss-source-prep-20261005/project/Sources/isa_inclusion_relation_list.txt": "26e7d818e03a8c909fe09c561f38d0d513423c87681f9450a803bc38f5b07564",
    "/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/nhtiss-source-prep-20261005/project/Sources/isa_parts_list_e.txt": "ab7796deedd49205e77f3609a1cb8c53e2bbee14ecb5c9a6ca05227469780513",
    "/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/nhtiss-source-prep-20261005/project/Sources/myosim/source-overlays/myosim-left-knee-translation2-range.v1.json": "05f6b7698e571c83a62bdbc7055ff24322800a36e9b53e455a18b451e7b9dad4",
    "/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/nhtiss-source-prep-20261005/project/Sources/partof_BP3D_4.0_obj_99.zip": "9fbc713fffeee924a5a657d9813d84d7eb957bded63adb854931dd5e3eb61c97",
    "/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/nhtiss-source-prep-20261005/project/Sources/partof_element_parts.txt": "3f5f6df1028eb122b30de77c711597b6bb8e5541658e5985859fd228adbf88ea",
    "/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/nhtiss-source-prep-20261005/project/Sources/partof_inclusion_relation_list.txt": "1b40738270931e3c1d955ce34e0fce0d8d10d8c5ad543463e40b4b4c0243007c",
    "/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/nhtiss-source-prep-20261005/project/Sources/partof_parts_list_e.txt": "9224080557053e6f1322f1e13ab27f0ecde0db19bb3b505f0631afad230eeebd",
    "/Users/n/numi-human-source-seam-connectivity-1247/config/anatomy-classification.v1.json": "d1f832062adebfab361f86692e5ee49b57278d5ac58690f778eca08c64dff084",
    "/Users/n/numi-human-source-seam-connectivity-1247/config/bodyparts3d-myosim-surface-map.v1.json": "bd08a3d604a754065fd693028c1a307487006f7d13acd40153bda87ed7e423f9",
    "/Users/n/numi-human-source-seam-connectivity-1247/src/numilab_human/cli.py": "930f4854dabdd611628b0947ad953c9fa98943bfe2735ad78d6a4652affa5c5c",
    "/Users/n/numi-human-source-seam-connectivity-1247/src/numilab_human/model.py": "23f6f02a93f7eb052cd46e890661519101f1274cfd69c64a44c40b2403edaebc",
}

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError('passive attachment composition: ' + message)


def _reconcile_serialized_triangle_counts(manifest: dict, records) -> list[dict]:
    # Historical source-selection and precision-repair counts remain provenance;
    # this reconciles only the current row count to serialized index_count / 3.
    source = manifest.get("source")
    surfaces = source.get("surfaces") if isinstance(source, dict) else None
    require(isinstance(surfaces, list) and len(surfaces) == len(records),
            "manifest and serialized surface row counts differ")
    changes = []
    for row_index, (surface, record) in enumerate(zip(surfaces, records)):
        require(isinstance(surface, dict)
                and isinstance(surface.get("stable_id"), int)
                and not isinstance(surface.get("stable_id"), bool)
                and surface["stable_id"] == int(record[6]),
                "manifest and serialized stable-row order differ")
        previous_count = surface.get("triangle_count")
        require(isinstance(previous_count, int) and not isinstance(previous_count, bool)
                and previous_count > 0,
                "manifest surface triangle_count is missing or invalid")
        serialized_index_count = int(record[5])
        require(serialized_index_count > 0 and serialized_index_count % 3 == 0,
                "serialized row index count is not a positive triangle multiple")
        serialized_triangle_count = serialized_index_count // 3
        if previous_count != serialized_triangle_count:
            change = {
                "stable_id": int(record[6]),
                "manifest_triangle_count_before": previous_count,
                "serialized_triangle_count": serialized_triangle_count,
                "serialized_index_count": serialized_index_count,
                "record_index": row_index,
                "reason": "inherited manifest count reconciled to the current NHTISS4 ABI5 row record",
            }
            selection = surface.get("source_component_selection")
            precision = surface.get("source_precision_repair")
            if isinstance(selection, dict) and "retained_triangle_count" in selection:
                change["preserved_source_component_selection_retained_triangle_count"] = selection["retained_triangle_count"]
            if isinstance(precision, dict) and "removed_triangle_count" in precision:
                change["preserved_source_precision_repair_removed_triangle_count"] = precision["removed_triangle_count"]
            changes.append(change)
            surface["triangle_count"] = serialized_triangle_count
        else:
            surface["triangle_count"] = serialized_triangle_count
    if changes:
        key = "serialized_triangle_count_reconciliation"
        prior = source.get(key)
        operation = {
            "method": "nhtiss4_abi5_surface_record_index_count_div_3",
            "scope": "Manifest triangle-count metadata only; serialized vertex/index bytes and historical source-selection/precision-repair lineage are unchanged.",
            "rows": changes,
        }
        if prior is None:
            source[key] = {"schema": "numi.human.serialized-triangle-count-reconciliation.v1",
                           "operations": [operation]}
        else:
            require(isinstance(prior, dict)
                    and prior.get("schema") == "numi.human.serialized-triangle-count-reconciliation.v1"
                    and isinstance(prior.get("operations"), list),
                    "unrecognized prior serialized-count reconciliation lineage")
            prior["operations"].append(operation)
    return changes

def compose(source: Path, output: Path, replacements: list[tuple[int, Path, Path]], *,
            biceps_source_correction: dict | None = None,
            fhl_source_seam_correction: dict | None = None) -> dict:
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
    fhl_correction = None
    if any(sid not in (7, 8, 23) for sid in replacement_ids):
        if set(replacement_ids) == {103, 104} and len(replacement_ids) == 2:
            require(biceps_source_correction is not None and fhl_source_seam_correction is None,
                    'proof-bound paired biceps correction is required without an FHL correction')
            biceps_correction = _verify_biceps_source_correction(T, biceps_source_correction)
            inputs.update(biceps_correction['input_sha256'])
        elif set(replacement_ids) == {27, 28} and len(replacement_ids) == 2:
            require(fhl_source_seam_correction is not None and biceps_source_correction is None,
                    'the paired FHL rows require only their proof-bound source-seam correction')
            fhl_correction = _verify_fhl_source_seam_correction(T, fhl_source_seam_correction)
            inputs.update(fhl_correction['input_sha256'])
        else:
            require(False, 'only distinct passive attachment rows 7, 8, and 23 are allowed, except the proof-bound biceps 103/104 or FHL 27/28 pairs')
    else:
        require(biceps_source_correction is None and fhl_source_seam_correction is None,
                'source correction proofs may only accompany their exact paired rows')
    for (sid, z, report) in replacements:
        sid = int(sid)
        allowed_pair = ((biceps_correction is not None and sid in (103, 104))
                        or (fhl_correction is not None and sid in (27, 28)))
        if (sid not in (7, 8, 23) and not allowed_pair) or sid in repairs:
            raise ValueError('Only distinct passive attachment rows 7, 8, 23 or exact proof-bound biceps 103/104 and FHL 27/28 pairs are allowed')
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
        if fhl_correction is not None:
            expected_patch = fhl_correction["row_patch_npz"].get(str(sid))
            require(isinstance(expected_patch, dict)
                    and z.resolve() == Path(expected_patch["path"]).resolve()
                    and sha(z) == expected_patch["sha256"],
                    "replacement row does not match its proof-bound FHL NPZ")
            require(report.resolve() == Path(fhl_correction["row_patch_preparation_report_path"]).resolve()
                    and sha(report) == fhl_correction["row_patch_preparation_report_sha256"],
                    "replacement report does not match the proof-bound FHL row-patch report")
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
    if biceps_correction is None and fhl_correction is None:
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
        if biceps_correction is None and fhl_correction is None:
            row['reference_attachment_reconstruction'] = {**proof, 'scope': 'Passive source-derived reference inspection surface. Original anatomical identity/laterality, named attachment relationships, MyoSim force route and compliant tendon state remain their original owners. Not a measured-person reconstruction.', 'prior_registration_metadata': 'Retained above as upstream provenance; this reconstruction supersedes the listed source surface geometry.'}
        elif fhl_correction is not None:
            row['fhl_source_seam_correction'] = {
                'row_patch_preparation_report_sha256': fhl_correction['row_patch_preparation_report_sha256'],
                'source_subset_payload_sha256': fhl_correction['source_subset_payload_sha256'],
                'stable_id': sid,
                'scope': 'Exact source-face seam restoration for the named FHL row; no vertex positions, normals, bindings, weights, or physical force routes changed.'}
    if biceps_correction is None and fhl_correction is None:
        manifest['source']['reference_attachment_composition'] = {'source_payload_sha256': inputs[str(T)], 'changed_stable_ids': sorted(repairs), 'unchanged_row_vertex_bytes_and_local_faces': unchanged, 'binding_table_byte_exact': True, 'physical_route_mass_and_force_state_unchanged': True}
    elif fhl_correction is not None:
        fhl_correction['composed_parent_payload_sha256'] = inputs[str(T)]
        fhl_correction['composed_parent_manifest_path'] = str(M)
        fhl_correction['composed_parent_manifest_sha256'] = inputs[str(M)]
        fhl_correction['changed_stable_ids'] = [27, 28]
        fhl_correction['unchanged_row_vertex_bytes_and_local_faces'] = unchanged
        manifest['source']['fhl_source_seam_correction'] = fhl_correction
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
    serialized_triangle_count_reconciliation = _reconcile_serialized_triangle_counts(
        manifest, newrec)
    PM = output / M.name
    PM.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    require(inputs == {p: sha(p) for p in inputs}, 'candidate composition invariant')
    proof = {'scope': 'Existing NHTISS4 ABI5 passive geometry composition; final native and anatomical admission separate.', 'input_sha256': inputs, 'inputs_unchanged': True, 'changed_rows': proofs, 'unchanged_row_vertex_bytes_and_local_faces': unchanged, 'binding_table_byte_exact': True, 'payload_sha256': sha(P), 'manifest_sha256': sha(PM), 'vertex_count': vcur, 'index_count': icur, 'serialized_triangle_count_reconciliation': serialized_triangle_count_reconciliation}
    if biceps_correction is not None:
        proof['biceps_source_preserving_correction'] = biceps_correction
    if fhl_correction is not None:
        proof['fhl_source_seam_correction'] = fhl_correction
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



def _verify_fhl_source_seam_correction(parent_payload: Path, correction: dict) -> dict:
    """Verify the exact two-row FHL source-sheet restoration against its evidence."""
    import collections
    import numpy as np

    parent_payload = Path(parent_payload).resolve()
    require(isinstance(correction, dict)
            and correction.get("schema") == "numi.human.fhl-source-seam-correction.v1",
            "FHL source-seam proof schema")
    path_fields = {
        "direct_parent_payload": parent_payload,
        "direct_parent_manifest": parent_payload.with_suffix(".manifest.json"),
        "source_subset_payload": correction.get("source_subset_payload_path"),
        "source_subset_manifest": correction.get("source_subset_manifest_path"),
        "row_comparison_report": correction.get("row_comparison_report_path"),
        "self_audit_report": correction.get("self_audit_report_path"),
        "source_topology_report": correction.get("source_topology_report_path"),
        "row_patch_preparation_report": correction.get("row_patch_preparation_report_path"),
        "source_declaration": correction.get("source_declaration_path"),
        "row_patch_27": correction.get("row_patch_npz", {}).get("27", {}).get("path"),
        "row_patch_28": correction.get("row_patch_npz", {}).get("28", {}).get("path"),
        "registration": correction.get("registration_path"),
    }
    require(all(value is not None for value in path_fields.values()),
            "FHL proof is missing a pinned source, report, registration, or row patch")
    path_fields = {key: Path(value).resolve() for key, value in path_fields.items()}
    expected = correction.get("expected_pins")
    require(expected == _FHL_SOURCE_SEAM_CORRECTION_EXPECTED_PINS
            and set(expected) == set(path_fields),
            "FHL source/report/registration/row-patch pins are not the reviewed exact set")
    actual_inputs = {}
    for key, path in path_fields.items():
        pin = expected[key]
        require(path == Path(pin["path"]).resolve()
                and path.is_file() and _surface_sha256(path) == pin["sha256"],
                "FHL expected pin mismatch: " + key)
        actual_inputs[str(path)] = pin["sha256"]

    parent_manifest_path, parent_manifest = _payload_manifest(parent_payload)
    subset_payload = path_fields["source_subset_payload"]
    subset_manifest_path, subset_manifest = _payload_manifest(subset_payload)
    require(parent_manifest_path == path_fields["direct_parent_manifest"]
            and subset_manifest_path == path_fields["source_subset_manifest"],
            "FHL payload manifest path identity")
    require(parent_manifest.get("source", {}).get("fhl_source_seam_correction") is None,
            "FHL source correction is already present in the direct parent")
    require(_producer_source_identity(parent_manifest["source"])
            == _producer_source_identity(subset_manifest["source"]),
            "FHL regenerated subset changed source, registration, or runtime producer identity")
    registration_sha = expected["registration"]["sha256"]
    for label, manifest in (("direct parent", parent_manifest), ("regenerated subset", subset_manifest)):
        source = manifest["source"]
        require(source.get("registration", {}).get("sha256") == registration_sha
                and source.get("myosim_manifest", {}).get("sha256") == "844d05330104a43f6c45867020f2abd493adec35d6e2f8f90fc636ee9bec04e7"
                and source.get("myosim_source_archive_sha256") == "280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975",
                "FHL " + label + " source registration identity")
    require(subset_manifest.get("coverage", {}).get("selected_stable_ids") == [23, 24, 27, 28],
            "FHL subset source selection differs from the reviewed four-row generation")

    declaration = json.loads(path_fields["source_declaration"].read_text())
    declaration_argv = declaration.get("argv")
    require(isinstance(declaration_argv, list) and "--registration" in declaration_argv
            and declaration_argv.index("--registration") + 1 < len(declaration_argv),
            "FHL source regeneration declaration has no registration argument")
    require(declaration.get("schema") == "numi.human.source-seam-regeneration-declaration.v1"
            and declaration.get("expected_current_parent_payload_sha256") == expected["direct_parent_payload"]["sha256"]
            and declaration.get("expected_current_parent_manifest_sha256") == expected["direct_parent_manifest"]["sha256"]
            and declaration.get("expected_registration_sha256") == registration_sha
            and declaration_argv[declaration_argv.index("--registration") + 1]
                == str(path_fields["registration"]),
            "FHL source regeneration declaration/registration binding")
    declaration_inputs = declaration.get("input_sha256")
    require(isinstance(declaration_inputs, dict)
            and declaration_inputs == _FHL_SOURCE_DECLARATION_INPUTS,
            "FHL source regeneration input set changed")
    for raw_path, digest in declaration_inputs.items():
        source_path = Path(raw_path)
        require(source_path.is_file() and _surface_sha256(source_path) == digest,
                "FHL regeneration source input changed: " + raw_path)
        actual_inputs[str(source_path.resolve())] = digest

    def verify_inputs(report: dict, field: str, label: str) -> None:
        pins = report.get(field)
        require(isinstance(pins, dict) and pins, label + " has no input hash set")
        for raw_path, digest in pins.items():
            source_path = Path(raw_path)
            require(source_path.is_file() and _surface_sha256(source_path) == digest,
                    label + " input changed: " + raw_path)
            actual_inputs[str(source_path.resolve())] = digest

    row_comparison = json.loads(path_fields["row_comparison_report"].read_text())
    require(row_comparison.get("schema") == "numi.human.source-seam-regenerated-row-comparison.v2"
            and row_comparison.get("source_topology_comparison_sha256") == expected["source_topology_report"]["sha256"],
            "FHL regenerated-row comparison source topology binding")
    verify_inputs(row_comparison, "pins_sha256", "FHL row comparison")
    require(row_comparison["pins_sha256"].get(str(parent_payload)) == expected["direct_parent_payload"]["sha256"]
            and row_comparison["pins_sha256"].get(str(parent_manifest_path)) == expected["direct_parent_manifest"]["sha256"]
            and row_comparison["pins_sha256"].get(str(subset_payload)) == expected["source_subset_payload"]["sha256"]
            and row_comparison["pins_sha256"].get(str(subset_manifest_path)) == expected["source_subset_manifest"]["sha256"],
            "FHL row comparison does not bind the direct parent and regenerated subset")
    comparison_rows = {int(row["stable_id"]): row for row in row_comparison.get("row_comparisons", [])}
    require(set(comparison_rows).issuperset({27, 28}), "FHL rows are absent from exact coordinate comparison")

    self_report = json.loads(path_fields["self_audit_report"].read_text())
    require(self_report.get("schema") == "numi.human.regenerated-muscle-exact-f32-self-audit.v1",
            "FHL exact-F32 self-audit schema")
    self_input_map = self_report.get("inputs")
    require(isinstance(self_input_map, dict), "FHL exact-F32 self audit input map")
    for input_name, entry in self_input_map.items():
        require(isinstance(entry, dict) and entry.get("path") and entry.get("sha256"),
                "FHL exact-F32 self audit input entry")
        source_path = Path(entry["path"])
        require(source_path.is_file() and _surface_sha256(source_path) == entry["sha256"],
                "FHL exact-F32 self audit input changed: " + str(input_name))
        actual_inputs[str(source_path.resolve())] = entry["sha256"]
    require(self_input_map.get("candidate_payload", {}).get("sha256") == expected["source_subset_payload"]["sha256"]
            and self_input_map.get("candidate_manifest", {}).get("sha256") == expected["source_subset_manifest"]["sha256"]
            and self_input_map.get("parent_payload", {}).get("sha256") == expected["direct_parent_payload"]["sha256"]
            and self_input_map.get("parent_manifest", {}).get("sha256") == expected["direct_parent_manifest"]["sha256"],
            "FHL exact-F32 self audit does not bind the parent and regenerated subset")
    predicate = self_input_map.get("predicate", {})
    package_init = self_input_map.get("package_init", {})
    require(predicate.get("sha256") == self_report.get("predicate_source_expected_sha256", {}).get("predicate")
            and package_init.get("sha256") == self_report.get("predicate_source_expected_sha256", {}).get("package_init"),
            "FHL exact-F32 predicate source pin mismatch")
    self_rows = {int(row["stable_id"]): row for row in self_report.get("regenerated_rows", [])}
    require(set(self_rows).issuperset({27, 28}), "FHL rows absent from exact-F32 self audit")

    topology = json.loads(path_fields["source_topology_report"].read_text())
    require(topology.get("input_pins_unchanged") is True,
            "FHL source topology comparison did not retain stable input pins")
    verify_inputs(topology, "inputs", "FHL source topology comparison")
    surfaces = topology.get("surfaces")
    if isinstance(surfaces, dict):
        topology_rows = {int(row.get("stable_id", key)): row for key, row in surfaces.items()}
    else:
        topology_rows = {int(row["stable_id"]): row for row in surfaces or []}
    require(set(topology_rows).issuperset({27, 28}), "FHL source topology rows missing")

    patch_report = json.loads(path_fields["row_patch_preparation_report"].read_text())
    require(patch_report.get("schema") == "numi.human.fhl-source-row-patch-preparation.v1"
            and patch_report.get("source_subset_payload_sha256") == expected["source_subset_payload"]["sha256"]
            and patch_report.get("source_subset_manifest_sha256") == expected["source_subset_manifest"]["sha256"]
            and patch_report.get("parent_payload_sha256") == expected["direct_parent_payload"]["sha256"]
            and patch_report.get("parent_manifest_sha256") == expected["direct_parent_manifest"]["sha256"]
            and patch_report.get("self_audit_report_sha256") == expected["self_audit_report"]["sha256"],
            "FHL row-patch preparation does not bind the source, parent, and exact self audit")
    verify_inputs(patch_report, "input_sha256", "FHL row-patch preparation")
    patch_rows = patch_report.get("rows")
    require(isinstance(patch_rows, dict) and set(patch_rows) == {"27", "28"},
            "FHL row-patch preparation must contain exactly rows 27 and 28")
    patch_paths = {sid: path_fields["row_patch_" + sid] for sid in ("27", "28")}
    declared_npz = correction.get("row_patch_npz")
    require(isinstance(declared_npz, dict) and set(declared_npz) == {"27", "28"},
            "FHL correction requires exactly the 27/28 row patch arrays")

    parent = _read_nhtiss4(parent_payload)
    subset = _read_nhtiss4(subset_payload)
    require(parent["surface_count"] == 150 and subset["surface_count"] == 4
            and parent["fingerprint"] == subset["fingerprint"]
            and parent["source_digest"] == subset["source_digest"]
            and f'{parent["fingerprint"]:08x}' == "b1b410ad",
            "FHL NHTISS inventory, registration, or source identity changed")
    parent_rows = {int(row[6]): _row_slices(parent, row) for row in parent["records"]}
    subset_rows = {int(row[6]): _row_slices(subset, row) for row in subset["records"]}
    require(set(subset_rows) == {23, 24, 27, 28} and {27, 28}.issubset(parent_rows),
            "FHL NHTISS row inventory changed")
    expected_rows = {
        27: {"member_id": "FJ1415", "member_sha256": "b00ad979e807617650ab158b609e9c2c1032a9483be5e9c9ec5a4796455c809c",
             "bindings": [{"core_body_index": 136, "myosim_body": "tibia_r"}, {"core_body_index": 138, "myosim_body": "calcn_r"}, {"core_body_index": 139, "myosim_body": "toes_r"}], "route": "fhl_r"},
        28: {"member_id": "FJ1415M", "member_sha256": "2cbe6f64ab578445284e188c89f454feb8a3a986e5460639720384e8269e6421",
             "bindings": [{"core_body_index": 150, "myosim_body": "tibia_l"}, {"core_body_index": 152, "myosim_body": "calcn_l"}, {"core_body_index": 153, "myosim_body": "toes_l"}], "route": "fhl_l"},
    }
    for sid in (27, 28):
        patch_path = patch_paths[str(sid)]
        patch_entry = declared_npz[str(sid)]
        require(Path(patch_entry.get("path", "")).resolve() == patch_path
                and patch_entry.get("sha256") == expected["row_patch_" + str(sid)]["sha256"]
                and _surface_sha256(patch_path) == patch_entry.get("sha256"),
                "FHL row patch path/hash differs from reviewed source row")
        rp = patch_rows[str(sid)]
        expected_row = expected_rows[sid]
        candidate_surface = _manifest_row(subset_manifest, sid)
        parent_surface = _manifest_row(parent_manifest, sid)
        for surface in (candidate_surface, parent_surface):
            require(surface.get("member_id") == expected_row["member_id"]
                    and surface.get("member_sha256") == expected_row["member_sha256"]
                    and surface.get("layer") == "muscle"
                    and surface.get("body_bindings") == expected_row["bindings"]
                    and len(surface.get("matched_muscles", [])) == 1
                    and surface["matched_muscles"][0].get("name") == expected_row["route"],
                    "FHL named source member or body binding identity changed")
        require(candidate_surface.get("body_bindings") == parent_surface.get("body_bindings")
                and candidate_surface.get("matched_muscles") == parent_surface.get("matched_muscles")
                and candidate_surface.get("endpoint_source") == parent_surface.get("endpoint_source"),
                "FHL regenerated row changed its existing runtime route")
        parent_row, candidate_row = parent_rows[sid], subset_rows[sid]
        require(parent_row["binding_bytes"] == candidate_row["binding_bytes"]
                and parent_row["layer"] == candidate_row["layer"] == 1,
                "FHL regenerated row changed binding bytes or layer")
        comparison = comparison_rows[sid]
        require(comparison.get("binding_bytes_exact") is True
                and comparison.get("body_binding_and_route_manifest_fields_exact") is True
                and comparison.get("old_xyz_multiset_preserved") is True
                and comparison.get("old_non_normal_attributes_preserved") is True
                and comparison.get("old_oriented_f32_face_coordinates_missing") == 0
                and comparison.get("old_normal_records_recomputed_or_changed_at_same_xyz") == 0
                and comparison.get("old_normal_records_retained_at_same_xyz") == comparison.get("current_vertex_records")
                and comparison.get("regenerated_face_rows") > comparison.get("current_face_rows")
                and comparison.get("regenerated_oriented_f32_face_coordinates_added")
                    == comparison.get("regenerated_face_rows") - comparison.get("current_face_rows"),
                "FHL regenerated row did not preserve old positions, weights, normals, and oriented faces")
        # Independently replay the row comparison's exact packed-coordinate
        # multiplicity claims before permitting the source row-patch extraction.
        def vertex_records(row):
            return [row["vertex_bytes"][i:i+56] for i in range(0, len(row["vertex_bytes"]), 56)]
        old_vertices, new_vertices = vertex_records(parent_row), vertex_records(candidate_row)
        old_pos = collections.Counter(v[:12] for v in old_vertices)
        new_pos = collections.Counter(v[:12] for v in new_vertices)
        require(not (old_pos - new_pos), "FHL regeneration dropped an existing exact position record")
        old_attr = collections.Counter((v[:12], v[24:56]) for v in old_vertices)
        new_attr = collections.Counter((v[:12], v[24:56]) for v in new_vertices)
        require(not (old_attr - new_attr), "FHL regeneration changed existing binding/weight attributes")
        old_norm = collections.Counter((v[:12], v[12:24]) for v in old_vertices)
        new_norm = collections.Counter((v[:12], v[12:24]) for v in new_vertices)
        require(not (old_norm - new_norm), "FHL regeneration changed existing normals")
        def oriented_face_counter(row):
            vertices = vertex_records(row)
            counts = collections.Counter()
            for face in row["local_faces"]:
                xyz = tuple(vertices[int(index)][:12] for index in face)
                cyclic = min(xyz, (xyz[1], xyz[2], xyz[0]), (xyz[2], xyz[0], xyz[1]))
                counts[cyclic] += 1
            return counts
        require(not (oriented_face_counter(parent_row) - oriented_face_counter(candidate_row)),
                "FHL regeneration removed/reoriented an existing face-coordinate record")
        require(candidate_row["index_count"] - parent_row["index_count"]
                == (comparison.get("regenerated_face_rows") - comparison.get("current_face_rows")) * 3,
                "FHL source seam restoration changed an unexpected face count")

        topo = topology_rows[sid]
        after = topo.get("after", {})
        selection = after.get("selection", {})
        closed = after.get("topology", {})
        cancellation = topo.get("after_existing_opposite_pair_cancellation", {})
        cancellation_stats = cancellation.get("cancellation", {})
        cancellation_topology = cancellation.get("topology", {})
        require(topo.get("stable_id", sid) == sid
                and topo.get("source_points_moved") is False
                and topo.get("new_inferred_faces") == 0
                and topo.get("prior_retained_oriented_source_support_preserved") is True
                and selection.get("connectivity_basis") == "exact_source_coordinate_edges_without_vertex_welding"
                and selection.get("retained_triangle_count") == candidate_row["index_count"] // 3
                and closed.get("closed_oriented_manifold_candidate") is True
                and closed.get("boundary_edge_count") == 0
                and closed.get("nonmanifold_edge_count") == 0
                and closed.get("orientation_mismatch_edge_count") == 0
                and closed.get("duplicate_face_row_count") == 0
                and closed.get("degenerate_face_rows") == []
                and cancellation_stats.get("source_coordinate_support_preserved") is True
                and cancellation_stats.get("oriented_source_chain_preserved") is True
                and cancellation_stats.get("vertices_moved") is False
                and cancellation_stats.get("new_faces_added") is False
                and cancellation_stats.get("cancelled_opposite_face_pairs") == []
                and cancellation_topology.get("closed_oriented_manifold_candidate") is True
                and cancellation_topology.get("boundary_edge_count") == 0,
                "FHL source selection is not the pinned closed oriented source sheet")

        audit = self_rows[sid]
        audit_topology = audit.get("topology_after_exact_f32_coordinate_quotient", {})
        exact = audit.get("exact_predicate", {})
        require(audit.get("face_rows") == candidate_row["index_count"] // 3
                and audit.get("vertex_records") == candidate_row["vertex_count"]
                and audit_topology.get("boundary_edges") == 0
                and audit_topology.get("nonmanifold_edges") == 0
                and audit_topology.get("orientation_mismatch_edges") == 0
                and audit_topology.get("duplicate_face_rows") == 0
                and audit_topology.get("repeated-index_degenerate_faces") == 0
                and exact.get("unallowed_self_intersection_pair_count") == 0
                and exact.get("unallowed_pair_ids") == [],
                "FHL regenerated row fails exact-F32 closure/orientation/self audit")
        row_audit = rp
        require(row_audit.get("member_id") == expected_row["member_id"]
                and row_audit.get("layer") == "muscle"
                and row_audit.get("body_bindings") == expected_row["bindings"]
                and row_audit.get("exact_f32_quotient_boundary_edges") == 0
                and row_audit.get("exact_f32_self_unallowed_pairs") == 0
                and row_audit.get("vertices6_shape") == [candidate_row["vertex_count"], 6]
                and row_audit.get("faces_shape") == [candidate_row["index_count"] // 3, 3]
                and row_audit.get("weights_shape") == [candidate_row["vertex_count"], 4],
                "FHL row-patch preparation lacks exact closed/self/route proof")
        with np.load(patch_path, allow_pickle=False) as stored:
            arrays = {key: np.asarray(stored[key]) for key in ("vertices6", "binding_indices", "weights", "faces")}
        expected_arrays = _biceps_row_arrays(candidate_row)
        require(all(arrays[key].dtype == expected_arrays[key].dtype
                    and arrays[key].shape == expected_arrays[key].shape
                    and arrays[key].tobytes(order="C") == expected_arrays[key].tobytes(order="C")
                    for key in expected_arrays),
                "FHL row patch arrays differ from the independently pinned regenerated source row")
        require(np.isfinite(arrays["vertices6"]).all() and np.isfinite(arrays["weights"]).all()
                and (arrays["weights"] >= 0).all()
                and np.max(np.abs(arrays["weights"].sum(axis=1, dtype=np.float64) - 1.0)) < 1e-5,
                "FHL row patch contains invalid position/normal/weight values")

    # Expose a normalized proof record while preserving all source evidence pins.
    return {
        "schema": "numi.human.fhl-source-seam-correction.v1",
        "source_subset_payload_path": str(subset_payload),
        "source_subset_payload_sha256": expected["source_subset_payload"]["sha256"],
        "source_subset_manifest_path": str(subset_manifest_path),
        "source_subset_manifest_sha256": expected["source_subset_manifest"]["sha256"],
        "row_comparison_report_path": str(path_fields["row_comparison_report"]),
        "row_comparison_report_sha256": expected["row_comparison_report"]["sha256"],
        "self_audit_report_path": str(path_fields["self_audit_report"]),
        "self_audit_report_sha256": expected["self_audit_report"]["sha256"],
        "source_topology_report_path": str(path_fields["source_topology_report"]),
        "source_topology_report_sha256": expected["source_topology_report"]["sha256"],
        "row_patch_preparation_report_path": str(path_fields["row_patch_preparation_report"]),
        "row_patch_preparation_report_sha256": expected["row_patch_preparation_report"]["sha256"],
        "source_declaration_path": str(path_fields["source_declaration"]),
        "source_declaration_sha256": expected["source_declaration"]["sha256"],
        "registration_path": str(path_fields["registration"]),
        "registration_sha256": expected["registration"]["sha256"],
        "expected_pins": expected,
        "row_patch_npz": {sid: {"path": str(patch_paths[sid]),
                                  "sha256": expected["row_patch_" + sid]["sha256"]}
                           for sid in ("27", "28")},
        "changed_stable_ids": [27, 28],
        "source_experiment_accepted_pose_status": "not_run",
        "direct_parent_payload_path": str(parent_payload),
        "direct_parent_payload_sha256": expected["direct_parent_payload"]["sha256"],
        "direct_parent_manifest_path": str(parent_manifest_path),
        "direct_parent_manifest_sha256": expected["direct_parent_manifest"]["sha256"],
        "binding_weights_and_original_geometry_preserved": True,
        "source_sheet_closed_oriented": True,
        "exact_f32_self_unallowed_pairs": {"27": 0, "28": 0},
        "input_sha256": actual_inputs,
    }


def _verify_fhl_composed_child(parent_payload: Path, child_payload: Path, correction: dict) -> None:
    """Prove the NHTISS child and its manifest differ only by the pinned FHL rows."""
    import copy
    import numpy as np

    parent_payload, child_payload = Path(parent_payload).resolve(), Path(child_payload).resolve()
    parent, child = _read_nhtiss4(parent_payload), _read_nhtiss4(child_payload)
    require(parent["raw"][:20] == child["raw"][:20]
            and parent["raw"][28:64] == child["raw"][28:64]
            and parent["surface_count"] == child["surface_count"]
            and parent["binding_count"] == child["binding_count"],
            "FHL child changed NHTISS source, registration, or inventory")
    require(parent["raw"][parent["binding_start"]:parent["vertex_start"]]
            == child["raw"][child["binding_start"]:child["vertex_start"]],
            "FHL child changed the global body binding table")

    parent_record_ids = [int(row[6]) for row in parent["records"]]
    child_record_ids = [int(row[6]) for row in child["records"]]
    require(parent_record_ids == child_record_ids
            and set(correction.get("changed_stable_ids", [])) == {27, 28},
            "FHL child changed stable-row ordering or selected rows")
    parent_rows = {int(row[6]): _row_slices(parent, row) for row in parent["records"]}
    child_rows = {int(row[6]): _row_slices(child, row) for row in child["records"]}
    require(all(parent_rows[sid]["layer"] == child_rows[sid]["layer"] for sid in parent_record_ids),
            "FHL child changed a NHTISS row layer")
    require(all(child_rows[sid]["layer"] == 1 for sid in (27, 28)),
            "FHL target rows are not in the native muscle layer")

    for sid, old_row in parent_rows.items():
        new_row = child_rows[sid]
        if sid not in (27, 28):
            require(old_row["vertex_bytes"] == new_row["vertex_bytes"]
                    and old_row["binding_bytes"] == new_row["binding_bytes"]
                    and old_row["local_faces"] == new_row["local_faces"],
                    "FHL child changed a non-target row")
            continue
        patch_path = Path(correction["row_patch_npz"][str(sid)]["path"])
        with np.load(patch_path, allow_pickle=False) as stored:
            arrays = {key: np.asarray(stored[key]) for key in ("vertices6", "binding_indices", "weights", "faces")}
        n = len(arrays["vertices6"])
        encoded = bytearray(n * 56)
        np.ndarray((n, 6), dtype="<f4", buffer=encoded, offset=0, strides=(56, 4))[:] = arrays["vertices6"]
        np.ndarray((n, 4), dtype="<u4", buffer=encoded, offset=24, strides=(56, 4))[:] = arrays["binding_indices"]
        np.ndarray((n, 4), dtype="<f4", buffer=encoded, offset=40, strides=(56, 4))[:] = arrays["weights"]
        require(new_row["vertex_bytes"] == bytes(encoded)
                and new_row["binding_bytes"] == old_row["binding_bytes"]
                and new_row["local_faces"] == arrays["faces"].tolist(),
                "FHL child row does not equal the exact pinned source patch or changed bindings")

    parent_manifest_path, parent_manifest = _payload_manifest(parent_payload)
    child_manifest_path, child_manifest = _payload_manifest(child_payload)
    parent_surfaces = parent_manifest.get("source", {}).get("surfaces")
    child_surfaces = child_manifest.get("source", {}).get("surfaces")
    require(isinstance(parent_surfaces, list) and isinstance(child_surfaces, list),
            "FHL manifests are missing source surface rows")
    parent_surface_ids = [row.get("stable_id") if isinstance(row, dict) else None for row in parent_surfaces]
    child_surface_ids = [row.get("stable_id") if isinstance(row, dict) else None for row in child_surfaces]
    require(parent_surface_ids == child_surface_ids == parent_record_ids,
            "FHL manifest source row order/identity differs from NHTISS")
    require(parent_manifest["source"].get("fhl_source_seam_correction") is None
            and child_manifest["source"].get("fhl_source_seam_correction") == correction,
            "FHL manifest correction lineage differs from its direct parent or proof")

    expected_manifest = copy.deepcopy(parent_manifest)
    expected_manifest["payload"].update({
        "bytes": child_payload.stat().st_size,
        "sha256": child["sha256"],
        "vertex_count": child["vertex_count"],
        "index_count": child["index_count"],
    })
    expected_surfaces = expected_manifest["source"]["surfaces"]
    expected_surface_by_id = {int(row["stable_id"]): row for row in expected_surfaces}
    for sid in (27, 28):
        source_row = expected_surface_by_id[sid]
        require(source_row.get("stable_id") == sid and source_row.get("layer") == "muscle",
                "FHL manifest target row identity or source layer changed")
        source_row["vertex_count"] = child_rows[sid]["vertex_count"]
        source_row["triangle_count"] = len(child_rows[sid]["local_faces"])
        source_row["fhl_source_seam_correction"] = {
            "row_patch_preparation_report_sha256": correction["row_patch_preparation_report_sha256"],
            "source_subset_payload_sha256": correction["source_subset_payload_sha256"],
            "stable_id": sid,
            "scope": ("Exact source-face seam restoration for the named FHL row; no vertex positions, "
                      "normals, bindings, weights, or physical force routes changed."),
        }
    _reconcile_serialized_triangle_counts(expected_manifest, child["records"])
    expected_manifest["source"]["fhl_source_seam_correction"] = correction
    require(expected_manifest == child_manifest,
            "FHL manifest changed source row metadata/order or fields outside the exact row-count/proof additions")

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
    fhl_composition = manifest["source"].get("fhl_source_seam_correction")
    if fhl_composition is not None:
        require(fhl_composition.get("changed_stable_ids") == [27, 28]
                and fhl_composition.get("composed_parent_payload_sha256") == owner["sha256"],
                "FHL child is not bound to the immediate anatomical parent")
        source_path = old.resolve()
        require(Path(fhl_composition["direct_parent_payload_path"]).resolve() == source_path
                and sha(source_path) == owner["sha256"],
                "FHL direct parent payload identity changed")
        accepted_manifest_path = Path(owner.get("manifest_path", source_path.with_suffix(".manifest.json")))
        if not accepted_manifest_path.is_absolute():
            accepted_manifest_path = source_receipt.parent / accepted_manifest_path
        require(accepted_manifest_path.resolve() == Path(fhl_composition["composed_parent_manifest_path"]).resolve()
                and owner.get("manifest_sha256") == fhl_composition["composed_parent_manifest_sha256"]
                and sha(accepted_manifest_path) == owner.get("manifest_sha256"),
                "FHL direct parent manifest identity changed")
        replay = _verify_fhl_source_seam_correction(source_path, fhl_composition)
        _verify_fhl_composed_child(source_path, payload, fhl_composition)
        require(fhl_composition.get("direct_parent_payload_sha256") == owner["sha256"]
                and fhl_composition.get("source_subset_payload_sha256") == replay["source_subset_payload_sha256"]
                and fhl_composition.get("row_patch_preparation_report_sha256") == replay["row_patch_preparation_report_sha256"],
                "FHL direct-child source proof identity changed")
        require(fhl_composition.get("binding_weights_and_original_geometry_preserved") is True
                and fhl_composition.get("source_sheet_closed_oriented") is True
                and fhl_composition.get("exact_f32_self_unallowed_pairs") == {"27": 0, "28": 0},
                "FHL source-seam proof is incomplete")
        composition = fhl_composition
        changed_stable_ids = [27, 28]
    elif biceps_composition is not None:
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
    binding_key = ("fhl_source_seam_correction_binding" if fhl_composition is not None
                   else "biceps_source_preserving_correction_binding" if biceps_composition is not None
                   else "conforming_surface_refinement_binding" if edge_composition is not None
                   else "passive_attachment_composition_binding")
    if fhl_composition is not None:
        require(binding_key not in receipt["provenance"],
                "anatomy receipt already contains this FHL source-seam provenance binding")
    elif biceps_composition is not None:
        require(binding_key not in receipt["provenance"],
                "anatomy receipt already contains this biceps correction provenance binding")
    receipt["provenance"][binding_key] = {
        "prior_receipt_path": str(source_receipt), "prior_receipt_sha256": sha(source_receipt),
        "composition_manifest_sha256": sha(manifest_path),
        "changed_stable_ids": changed_stable_ids,
        "scope": ("Proof-bound FHL source-seam correction composed as a direct child of this receipt; anatomy acceptance remains separate."
                  if fhl_composition is not None else
                  "Explicit source-derived conforming surface refinement only; existing physical owners, "
                  "mass, forces, and tendon state are unchanged."
                  if edge_composition is not None and biceps_composition is None else
                  "Proof-bound biceps source-geometry patch composed as a direct child of this receipt; anatomy acceptance remains separate."
                  if biceps_composition is not None else
                  "Passive attachment source binding only; physical owners and anatomical acceptance are unchanged.")
    }
    if fhl_composition is not None:
        receipt["provenance"][binding_key]["source_seam_correction"] = {
            "schema": fhl_composition["schema"],
            "direct_parent_payload_sha256": fhl_composition["direct_parent_payload_sha256"],
            "source_subset_payload_sha256": fhl_composition["source_subset_payload_sha256"],
            "row_patch_preparation_report_sha256": fhl_composition["row_patch_preparation_report_sha256"],
            "registration_sha256": fhl_composition["registration_sha256"],
            "exact_f32_self_unallowed_pairs": fhl_composition["exact_f32_self_unallowed_pairs"],
            "native_pose_qualification": "not_run",
        }
    if biceps_composition is not None and fhl_composition is None:
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
    parser.add_argument("--fhl-source-seam-correction", type=Path,
                        help="proof JSON for the exact pinned FHL 27/28 source-seam correction")
    parser.add_argument("--anatomy-receipt", type=Path,
                        help="bind the composed surface rows in a new native anatomy launch receipt")
    args = parser.parse_args(argv)
    edge_mode = args.conforming_edge_context is not None or args.conforming_edge is not None
    if edge_mode:
        if args.fhl_source_seam_correction is not None:
            parser.error("FHL correction mode cannot be combined with conforming edge mode")
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
            fhl_proof = (json.loads(args.fhl_source_seam_correction.read_text())
                         if args.fhl_source_seam_correction is not None else None)
            report = compose(args.source, args.output, args.row,
                             fhl_source_seam_correction=fhl_proof)
        if args.anatomy_receipt is not None:
            bind_anatomy_receipt(args.anatomy_receipt, args.output / args.source.name,
                                 args.output / "resting-anatomy-receipt.json")
    except (OSError, ValueError, KeyError, struct.error) as error:
        parser.exit(2, f'{error}\n')
    print(json.dumps({key: report[key] for key in ('payload_sha256', 'manifest_sha256', 'vertex_count', 'index_count', 'binding_table_byte_exact')}))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())

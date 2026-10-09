#!/usr/bin/env python3
"""Read-only exact geometry audit for the composed 1218 native baseline.

The tool never launches the scene and never reconstructs positions. It reads
only accepted MRVPACK captures already written by the designated run. A subset
scan remains partial evidence; only all eight declared captures can pass.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import resource
import struct
import sys
import time
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218")
RUN = ROOT / "native-baseline-310s-preparation/native-run"
OUT_ROOT = ROOT / "native-accepted-geometry-audit-001"
E = Path("/Users/n/numi-human-resting-evidence-20261005")
C = ROOT / "package-preparation-002/composed-candidate"
REVIEW = C / "composition-review.json"
REVIEW_SHA = "6ecf163986a45701204a58a572fb1766d58c909bc181f9ca19df8e6f4fc48807"
SKIN = C / "bodyparts3d-myosim-skinned-shell.nhskin"
SKIN_SHA = "ec5664d57dc1536dfe494e617330270794e03da654206721f5167b6247df07a7"
SKIN_MANIFEST = C / "common-atlas-skin-geometry-registration.manifest.json"
SKIN_MANIFEST_SHA = "52b3e15d73fe0589c747764d92193b073050c7dec83e5d30cf4d3a76068685e0"
SCENE = C / "resting-supine-scene.manifest.json"
SCENE_SHA = "6da4ff54bc75029f8d107a1cc9d8a1a2eef233eb8303c5c2943e9c7867a5009d"
RECEIPT = C / "composed-anatomy/resting-anatomy-receipt.json"
RECEIPT_SHA = "ebfb61b926f33dd6497176734324e7208e1eeab2f2b518e6d4f884e0a8da99ec"
ANATOMY_MANIFEST = C / "composed-anatomy/resting-anatomy-manifest.json"
ANATOMY_MANIFEST_SHA = "bf1f3264256517a602278f1a459d7c8d403d690a26307fa94b7198a7b88aa4db"
NHA = E / "native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-thorax.nhanatomy"
NHA_SHA = "1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241"
NHA_COMPOSITION = E / "native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/composition-report.json"
NHA_COMPOSITION_SHA = "ddbfae12a5acd5c35521cb913121f08e49fb8c336c89b7dd60df3272ec0a7fbc"
NHA_MANIFEST = E / "native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-anatomy-manifest.json"
NHA_MANIFEST_SHA = "de298b7cc413e0cf079b39cb5d4792ebb537642f23a52cb4747f2dc37fac917f"
TISS = Path("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue")
TISS_SHA = "1cd0c3d5bd1ff6d163f6544be4d0fa169e8bf696bf6c729ef0b18879b0856c48"
TISS_MANIFEST = TISS.with_suffix(".manifest.json")
TISS_MANIFEST_SHA = "f052eff736f040f2f6ca1a1a99d2939ca0ed5c0bead055dfb39d3141f05a3cac"
TISS_REPORT = TISS.parent / "report.json"
TISS_REPORT_SHA = "67f4b635b7ddfbbfe8486cf9f0373a97e98928ac251b43ac6394bf962ca91954"
TISS_SOURCE_RECEIPT = TISS.parent / "resting-anatomy-receipt.json"
TISS_SOURCE_RECEIPT_SHA = "10ecac382b42e08b8ba296f361d419a0a7db1fb6e458448596e127acbea14d62"
TISS_SOURCE_REVISION = Path("/Users/n/numi-human-conforming-composition-source-1216/source-revision.json")
TISS_SOURCE_REVISION_SHA = "51e3a732e18fbdb6922871b435d98f42d01c00511d2d08a0fc4cac1009faba6f"
SOURCE = Path("/Users/n/numi-human-conforming-composition-source-1216/src/numilab_human")
SOURCE_INIT = SOURCE / "__init__.py"
SOURCE_INIT_SHA = "e06e46ca88bde0b4c4a6235c6135dafdb2e75d27633776dba0098698a082cf42"
SOURCE_MODEL = SOURCE / "model.py"
SOURCE_MODEL_SHA = "a12a26af00fcab009d525094f3fb3d87ea575eff30cd15b91b1a0e12f0a316c3"
SOURCE_CI = SOURCE / "cardiac_cavity_intersections.py"
SOURCE_CI_SHA = "934c64fa6a64a06044aee6db44cdcd71dc497fcec97a20a80029575faaaff4b4"
SOURCE_GEOM = SOURCE / "cardiac_cavity_geometry.py"
SOURCE_GEOM_SHA = "f6e98744dad9e23cd3b505efc02e7faa618fccce81d2ee08b07f948ef182fb72"
SOURCE_CLEARANCE = SOURCE / "common_atlas_skin_clearance.py"
SOURCE_CLEARANCE_SHA = "ec198693fce562db27c4b761e3dbbc4b1bff4f0b9458ae67ab7ba596a97a000f"
SOURCE_TISS_CODEC = SOURCE / "passive_attachment_composition.py"
SOURCE_TISS_CODEC_SHA = "b568f2354c4eaebf4eb9c5212982bd8115d65766e997b6d1642725322b14337b"
RUN_OWNER = Path("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/native-refinement-1217-attempt2/audit_skin_1217.py")
RUN_OWNER_SHA = "3c8b2a10a66788ff198c91e3eaa3a12f5e4c2380225b39be456d1d5e581a44dc"
AUDIT_1172 = E / "native-lung-late-skin-audit-runner-1172/revision-003/audit_full_skin_cycle_1172.py"
AUDIT_1172_SHA = "b1b9be5959610325067f703c32bd55e2f568c11124ec232f0d066ed75a6fdcbb"
INVENTORY = E / "native-complete-skin-containment-audit-890/pair-summary-v3.csv"
INVENTORY_SHA = "a43484aa8d65b207cc498bb470097ec5f06ef2bb9872837bde9f197e19e815e3"
NINE_SUMMARY = ROOT / "local-self-reduction-audit-001/scan-001/summary.json"
NINE_SUMMARY_SHA = "032a8eac0621a50ca6582ab79d94c289966f59d025cec864752a01f06a6c4b4c"
NINE_DECL = ROOT / "local-self-reduction-audit-001/scan-001/declaration.json"
NINE_DECL_SHA = "20eaf46dff1671d971176e5de2b96c3e72ee4f4017f5a48b8550e4d5ae04dbd9"
NINE_SCRIPT = ROOT / "local-self-reduction-audit-001/audit_local_reduction_1218_root003.py"
NINE_SCRIPT_SHA = "b89702107bb320ca48415781bdd3b62b2c048d490d4b3ec2c318667977b89d5b"

SKIN_KEY = (51007, 1)
M63, M64 = (51005, 63), (51005, 64)
MUSCLE_FACE_COUNTS = {M63: 6688, M64: 6690}
OCULAR = {(51010, sid) for sid in range(381, 398)}
CAPTURES = (0, 9983, 19999, 47519, 152191, 153183, 154143, 155000)
DT, ROOTS, TARGETS = 0.002, 155000, 859
F32_DT = struct.unpack("<f", struct.pack("<f", DT))[0]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, expected):
    path = Path(path).resolve()
    need(path.is_file() and not path.is_symlink() and sha(path) == expected,
         "pinned input missing, symlinked, or changed: " + str(path))
    return path


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, "cannot load module " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_outer_owner():
    for p, h in ((SOURCE_INIT, SOURCE_INIT_SHA), (SOURCE_MODEL, SOURCE_MODEL_SHA),
                 (SOURCE_CI, SOURCE_CI_SHA), (SOURCE_GEOM, SOURCE_GEOM_SHA),
                 (SOURCE_CLEARANCE, SOURCE_CLEARANCE_SHA)):
        pin(p, h)
    package_name = "_numi_human_outer_envelope_1216_for_1218"
    for name in tuple(sys.modules):
        if name == package_name or name.startswith(package_name + "."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(package_name, SOURCE_INIT,
        submodule_search_locations=[str(SOURCE)])
    need(spec is not None and spec.loader is not None, "cannot load frozen 1216 package")
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    mod = importlib.import_module(package_name + ".common_atlas_skin_clearance")
    need(Path(mod.__file__).resolve() == SOURCE_CLEARANCE.resolve(),
         "outer-envelope owner resolved to an unexpected module")
    return mod


def load_owners():
    pin(RUN_OWNER, RUN_OWNER_SHA)
    wrapper = load_file(RUN_OWNER, "_native_skin_scan_wrapper_1217_for_1218")
    wrapper.SKIN, wrapper.SKIN_SHA = SKIN, SKIN_SHA
    owner = wrapper.load_runner()
    ci, capture_clearance, validator, core = owner.load_predicates()
    owner_files = {
        RUN_OWNER: RUN_OWNER_SHA, AUDIT_1172: AUDIT_1172_SHA, INVENTORY: INVENTORY_SHA,
        owner.AUDIT_CORE: owner.AUDIT_CORE_SHA, owner.HELPER: owner.HELPER_SHA,
        owner.PRED: owner.PRED_SHA, owner.CLEARANCE: owner.CLEARANCE_SHA,
        **owner.PREDICATE_SOURCE_PINS,
    }
    for p, h in owner_files.items():
        pin(p, h)
    outer = load_outer_owner()
    pin(SOURCE_TISS_CODEC, SOURCE_TISS_CODEC_SHA)
    codec = load_file(SOURCE_TISS_CODEC, "_nhtiss4_codec_1216_for_1218")
    return wrapper, owner, ci, capture_clearance, validator, core, outer, codec, owner_files


def validate_candidate(codec):
    static = {
        REVIEW: REVIEW_SHA, SKIN: SKIN_SHA, SKIN_MANIFEST: SKIN_MANIFEST_SHA,
        SCENE: SCENE_SHA, RECEIPT: RECEIPT_SHA, ANATOMY_MANIFEST: ANATOMY_MANIFEST_SHA,
        NHA: NHA_SHA, NHA_COMPOSITION: NHA_COMPOSITION_SHA, NHA_MANIFEST: NHA_MANIFEST_SHA,
        TISS: TISS_SHA, TISS_MANIFEST: TISS_MANIFEST_SHA, TISS_REPORT: TISS_REPORT_SHA,
        TISS_SOURCE_RECEIPT: TISS_SOURCE_RECEIPT_SHA, TISS_SOURCE_REVISION: TISS_SOURCE_REVISION_SHA,
        NINE_SUMMARY: NINE_SUMMARY_SHA, NINE_DECL: NINE_DECL_SHA, NINE_SCRIPT: NINE_SCRIPT_SHA,
    }
    for p, h in static.items():
        pin(p, h)
    review = json.loads(REVIEW.read_text())
    need(review.get("status") == "source_candidate_composed_native_pending"
         and review.get("candidate_admitted") is False and review.get("native_run_performed") is False,
         "candidate must remain pending native qualification")
    for key in ("NHA_payload_and_functional_bindings_unchanged", "current_refined_NHTISS_unchanged",
                "fixed_flat_bed_and_support_witnesses_unchanged", "full_86_bindings_byte_identical",
                "indices_and_face_order_byte_identical", "normals_recomputed_by_existing_owner"):
        need(review.get("preservation", {}).get(key) is True, "candidate preservation gate failed: " + key)
    for key, path, digest in (("candidate_skin", SKIN, SKIN_SHA),
        ("candidate_manifest", SKIN_MANIFEST, SKIN_MANIFEST_SHA), ("candidate_scene", SCENE, SCENE_SHA),
        ("composed_receipt", RECEIPT, RECEIPT_SHA)):
        row = review.get(key, {})
        need(Path(row.get("path", "")).resolve() == path.resolve() and row.get("sha256") == digest
             and row.get("bytes") == path.stat().st_size, "composition review binding mismatch: " + key)

    reg = json.loads(SKIN_MANIFEST.read_text())
    output = reg.get("output_payload", {})
    need(Path(output.get("path", "")).resolve() == SKIN.resolve() and output.get("sha256") == SKIN_SHA
         and output.get("bytes") == SKIN.stat().st_size, "registration manifest does not bind candidate NHSKIN")
    preservation = reg.get("preservation", {})
    need(preservation.get("full_86_column_weight_matrix_byte_identical") is True
         and preservation.get("triangle_indices_and_order_byte_identical") is True
         and preservation.get("skin_contact_or_physics_owner_changed") is False,
         "candidate NHSKIN binding/topology/owner preservation failed")
    nine = review.get("nine_pose_audit", {})
    need(nine.get("summary_status") == "complete_exact_intersection_free"
         and nine.get("aggregate_geometry_gate") is True and nine.get("inputs_unchanged") is True
         and nine.get("target_count") == TARGETS
         and nine.get("summary", {}).get("sha256") == NINE_SUMMARY_SHA
         and nine.get("declaration", {}).get("sha256") == NINE_DECL_SHA
         and nine.get("audit_script", {}).get("sha256") == NINE_SCRIPT_SHA,
         "candidate review does not bind the successful nine-pose audit")
    nine_summary = json.loads(NINE_SUMMARY.read_text())
    need(nine_summary.get("status") == "complete_exact_intersection_free"
         and nine_summary.get("all_nine_target_self_degenerate_and_closed_envelope_gates_pass") is True
         and nine_summary.get("inputs_unchanged") is True
         and nine_summary.get("target_surface_count") == TARGETS,
         "nine-pose predecessor audit is not a complete passing result")

    nha_manifest = json.loads(NHA_MANIFEST.read_text())
    need(nha_manifest.get("payload", {}).get("path") == str(NHA)
         and nha_manifest.get("payload", {}).get("sha256") == NHA_SHA,
         "1178 NHA manifest mismatch")
    nha_composition = json.loads(NHA_COMPOSITION.read_text()).get("outputs", {}).get("final_nha", {})
    need(nha_composition.get("path") == str(NHA) and nha_composition.get("sha256") == NHA_SHA,
         "1178 NHA composition mismatch")
    receipt = json.loads(RECEIPT.read_text())
    need(receipt.get("payload", {}).get("path") == str(NHA)
         and receipt.get("payload", {}).get("sha256") == NHA_SHA,
         "composed anatomy receipt does not bind NHA")
    muscle = receipt.get("provenance", {}).get("native_muscle_surfaces", {})
    need(muscle.get("payload_path") == str(TISS) and muscle.get("sha256") == TISS_SHA
         and muscle.get("manifest_path") == str(TISS_MANIFEST)
         and muscle.get("manifest_sha256") == TISS_MANIFEST_SHA,
         "composed anatomy receipt does not bind refined NHTISS")

    tm = json.loads(TISS_MANIFEST.read_text())
    payload = tm.get("payload", {})
    need(payload.get("file") == TISS.name and payload.get("sha256") == TISS_SHA
         and payload.get("vertex_count") == 433151 and payload.get("index_count") == 1884573,
         "refined NHTISS manifest mismatch")
    refinement = tm.get("source", {}).get("conforming_edge_refinement_composition", {})
    need(refinement.get("stable_id") == 64 and refinement.get("edge_local_vertex_ids") == [2597, 3054]
         and refinement.get("physical_route_mass_and_force_state_unchanged") is True,
         "refined stable-64 NHTISS lineage is missing")
    tr = json.loads(TISS_REPORT.read_text())
    need(tr.get("payload_sha256") == TISS_SHA and tr.get("manifest_sha256") == TISS_MANIFEST_SHA
         and tr.get("stable_id") == 64 and tr.get("edge_local_vertex_ids") == [2597, 3054]
         and tr.get("binding_table_byte_exact") is True and tr.get("inputs_unchanged") is True,
         "NHTISS report does not validate source-derived conforming refinement")
    anatomy_manifest = json.loads(ANATOMY_MANIFEST.read_text())
    need(anatomy_manifest.get("payload", {}).get("sha256") == NHA_SHA
         and anatomy_manifest.get("native_muscle_surfaces", {}).get("sha256") == TISS_SHA,
         "final anatomy manifest source binding mismatch")

    parsed = codec._read_nhtiss4(TISS)
    manifest = json.loads(TISS_MANIFEST.read_text())
    source_rows = {}
    for key, face_count in MUSCLE_FACE_COUNTS.items():
        manifest_row = codec._manifest_row(manifest, key[1])
        binary_rows = [row for row in parsed["records"] if int(row[6]) == key[1]]
        need(len(binary_rows) == 1 and manifest_row.get("stable_id") == key[1],
             "source NHTISS manifest/binary rows do not uniquely bind stable ID %d" % key[1])
        sl = codec._row_slices(parsed, binary_rows[0])
        need(sl["stable_id"] == key[1] and sl["index_count"] // 3 == face_count,
             "source NHTISS row count mismatch for stable ID %d" % key[1])
        source_rows[key] = sl
    return static, source_rows


def getarg(argv, name):
    try:
        i = argv.index(name)
        need(i + 1 < len(argv), "missing argv value: " + name)
        return argv[i + 1]
    except ValueError:
        raise ValueError("native invocation lacks " + name)


def validate_invocation(run, require_closed):
    ip = run / "invocation.json"
    need(ip.is_file() and not ip.is_symlink(), "native invocation is not available")
    inv = json.loads(ip.read_text())
    argv, assets = inv.get("argv"), inv.get("asset_sha256")
    need(isinstance(argv, list) and isinstance(assets, dict), "invocation lacks argv/assets")
    for opt, path in (("--body-scene", SCENE), ("--skin-payload", SKIN),
        ("--soft-tissue-payload", TISS), ("--torso-anatomy-payload", NHA),
        ("--resting-anatomy-receipt", RECEIPT)):
        actual = Path(getarg(argv, opt)).resolve()
        need(actual == path.resolve() and assets.get(str(actual)) == sha(path),
             "actual invocation does not bind exact " + opt)
    need(len(argv) > 3 and Path(argv[3]).resolve() == run.resolve(),
         "invocation output path differs from designated run")
    need(float(getarg(argv, "--muscle-step-seconds")) == DT
         and int(getarg(argv, "--muscle-step-count")) == ROOTS,
         "invocation is not the 310 s, 2 ms native baseline")
    for asset_path, expected in assets.items():
        pin(asset_path, expected)
    mp = run / "run-metadata.json"
    metadata = None
    if mp.is_file() and not mp.is_symlink():
        metadata = json.loads(mp.read_text())
        need(metadata.get("argv") == argv and metadata.get("asset_sha256") == assets,
             "run metadata differs from invocation")
        if full:
            need(metadata.get("exit_code") == 0, "full audit requires successful native exit")
            need(metadata.get("loaded_metal_runtime", {}).get("verified") is True,
                 "full audit requires verified loaded runtime")
            changed = metadata.get("source_files_changed_during_run")
            need(changed == [] or changed is False, "native source-change state is not clean")
    elif full:
        raise ValueError("full audit requires closed successful run metadata")
    return inv, metadata, assets, ip, mp


def audit_muscles(step, run, out, owner, ci, capture_clearance, validator, core, outer,
                  source_rows, nha_sha):
    import numpy as np
    pack = run / "accepted-geometry" / ("step-%d.mrvpack" % step)
    receipt = pack.with_suffix(".receipt.json")
    accepted = owner.verify_pack(step, pack, receipt, validator, nha_sha)
    positions, surfaces, counts = capture_clearance._pack_surfaces(pack, {M63, M64})
    need(owner.EXPECTED_SKIN_KEY in surfaces and M63 in surfaces and M64 in surfaces,
         "native pack lacks skin or one of the two muscle surfaces")
    skin_ids = np.unique(surfaces[owner.EXPECTED_SKIN_KEY]["faces"])
    row_results, record_data, faces_data = {}, {}, {}
    witness_path = out / ("step-%d.muscle-crossing-witnesses.jsonl" % step)
    with witness_path.open("x") as stream:
        for key in (M63, M64):
            faces = np.asarray(surfaces[key]["faces"])
            sl = source_rows[key]
            local_expected = np.asarray(sl["local_faces"], dtype=np.int64)
            face_base = int(faces.min())
            local_faces = faces.astype(np.int64) - face_base
            need(len(faces) == MUSCLE_FACE_COUNTS[key]
                 and np.array_equal(local_faces, local_expected),
                 "captured face rows/order differ from exact current NHTISS stable ID %d" % key[1])
            ids = np.unique(faces)
            local = np.searchsorted(ids, faces)
            records, record_rows, deg = core.exact_records(positions[ids], local, ci)
            audit = ci._audit_pair(records, records, same_surface=True)
            for a, b in audit["triangle_pairs"]:
                core.crossing_witness(stream, semantic=key[0], stable_id=key[1],
                    skin_pair_index=a, target_pair_index=b, skin_record=records[a],
                    target_record=records[b], skin_row=record_rows[a], target_row=record_rows[b],
                    skin_faces=faces, target_faces=faces, positions=positions, ci=ci,
                    role="muscle_surface_self_intersection")
            target = outer._prepare_closed_clearance_target(
                positions[ids], local, allow_nested_enclosure=True)
            inside_local = outer._closed_target_inside_vertices(positions[skin_ids], target)
            rep = target["report"]
            row_results[key] = {
                "surface": list(key), "face_count": int(len(faces)),
                "source_face_count": int(len(local_expected)), "captured_vertex_base": face_base,
                "source_first_vertex": int(sl["first_vertex"]), "source_face_order_exact": True,
                "self_intersection_count": int(audit["count"]),
                "quotient_self_intersection_count": int(rep["self_intersection_audit"]["count"]),
                "degenerate_face_rows": [int(x) for x in deg],
                "component_face_rows": rep["component_face_rows"],
                "component_pair_proofs": rep["component_pair_proofs"],
                "face_count_in_prepared_target": int(rep["face_count"]),
                "all_components_closed_oriented_unused_free": rep["all_components_closed_oriented_unused_free"],
                "embedded_closed_target": rep["embedded_closed_target"],
                "inside_semantics": rep["inside_semantics"],
                "external_skin_enclosure_only": rep["external_skin_enclosure_only"],
                "outer_component_face_count": rep["outer_component_face_count"],
                "inside_skin_vertex_count": int(len(inside_local)),
                "inside_skin_pack_vertex_ids": [int(x) for x in skin_ids[inside_local]],
                "target_input_hashes": rep["input_hashes"], "target_quotient_hashes": rep["quotient_hashes"],
                "outer_envelope_clear": bool(rep["embedded_closed_target"]
                    and rep["all_components_closed_oriented_unused_free"] and len(inside_local) == 0),
            }
            record_data[key], faces_data[key] = (records, record_rows), faces
        (r63, rows63), (r64, rows64) = record_data[M63], record_data[M64]
        cross = ci._audit_pair(r63, r64, same_surface=False)
        for a, b in cross["triangle_pairs"]:
            core.crossing_witness(stream, semantic=M63[0], stable_id=M63[1],
                skin_pair_index=a, target_pair_index=b, skin_record=r63[a], target_record=r64[b],
                skin_row=rows63[a], target_row=rows64[b], skin_faces=faces_data[M63],
                target_faces=faces_data[M64], positions=positions, ci=ci,
                role="stable_63_64_cross_intersection")
    result = {
        "accepted_step": step, "accepted_receipt": accepted,
        "capture_sha256": sha(pack), "receipt_sha256": sha(receipt), "pack_counts": counts,
        "surfaces": [row_results[M63], row_results[M64]],
        "stable_63_64_cross_pair_count": int(cross["count"]),
        "stable_63_64_cross_aabb_candidate_pairs": int(cross["aabb_candidate_pairs"]),
        "crossing_witnesses_sha256": sha(witness_path),
        "all_rows_match_source_face_topology": all(v["source_face_order_exact"] for v in row_results.values()),
        "all_muscle_self_zero": all(v["self_intersection_count"] == 0
            and v["quotient_self_intersection_count"] == 0 and not v["degenerate_face_rows"]
            and v["all_components_closed_oriented_unused_free"] for v in row_results.values()),
        "all_outer_envelopes_clear": all(v["outer_envelope_clear"] for v in row_results.values()),
        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    writej(out / ("step-%d.muscle-geometry.json" % step), result)
    return result


def writej(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-check", action="store_true")
    mode.add_argument("--scan", action="store_true")
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", help="ordered subset of the exact eight-step capture schedule")
    args = parser.parse_args()
    wrapper, owner, ci, capture_clearance, validator, core, outer, codec, owner_files = load_owners()
    static, source_rows = validate_candidate(codec)
    keys, expected = owner.inventory()
    need(len(keys) == TARGETS and M63 in keys and M64 in keys,
         "the pinned 859-target inventory must include stable muscle IDs 63 and 64")
    need(owner.EXPECTED_SKIN_KEY == SKIN_KEY, "pinned scan owner uses unexpected skin semantic/stable ID")
    static.update({Path(p).resolve(): h for p, h in owner_files.items()})
    static.update({RUN_OWNER.resolve(): RUN_OWNER_SHA, Path(__file__).resolve(): sha(__file__)})
    static.update({SOURCE_INIT.resolve(): SOURCE_INIT_SHA, SOURCE_MODEL.resolve(): SOURCE_MODEL_SHA,
        SOURCE_CI.resolve(): SOURCE_CI_SHA, SOURCE_GEOM.resolve(): SOURCE_GEOM_SHA,
        SOURCE_CLEARANCE.resolve(): SOURCE_CLEARANCE_SHA, SOURCE_TISS_CODEC.resolve(): SOURCE_TISS_CODEC_SHA})
    if args.prepare_check:
        print(json.dumps({"status": "static_inputs_valid_no_capture_scan",
            "audit_script_sha256": sha(__file__), "candidate_skin_sha256": SKIN_SHA,
            "candidate_scene_sha256": SCENE_SHA, "candidate_receipt_sha256": RECEIPT_SHA,
            "NHA_sha256": NHA_SHA, "NHTISS_sha256": TISS_SHA,
            "nine_pose_summary_sha256": NINE_SUMMARY_SHA, "target_count": len(keys),
            "ocular_target_count": len(OCULAR), "stable_muscle_face_counts": {
                str(k): len(v["local_faces"]) for k, v in source_rows.items()},
            "capture_steps": list(CAPTURES), "owner_files": len(static)}, sort_keys=True))
        return 0

    steps = args.steps or list(CAPTURES)
    need(steps == sorted(steps) and len(steps) == len(set(steps))
         and set(steps).issubset(set(CAPTURES)), "--steps must be an ordered subset of the declared schedule")
    full = tuple(steps) == CAPTURES
    require_closed = full or CAPTURES[-1] in steps
    need(args.out is not None, "--scan requires a fresh --out path")
    run = args.run.expanduser().resolve()
    need(run == RUN.resolve() and run.is_dir() and not run.is_symlink(),
         "--run must be the designated 1218 native baseline")
    out = args.out.expanduser().resolve()
    need(out.parent == OUT_ROOT.resolve() and not out.exists(),
         "--out must be a new direct child of the retained audit root")
    invocation, metadata, assets, invocation_path, metadata_path = validate_invocation(run, require_closed)
    pack_paths = []
    capture_pins = {}
    for step in steps:
        pack = run / "accepted-geometry" / ("step-%d.mrvpack" % step)
        receipt = pack.with_suffix(".receipt.json")
        need(pack.is_file() and not pack.is_symlink() and receipt.is_file() and not receipt.is_symlink(),
             "requested actual accepted pack/receipt is missing for step %d" % step)
        receipt_doc = json.loads(receipt.read_text())
        need(int(receipt_doc.get("accepted_step", -1)) == step
             and abs(float(receipt_doc.get("accepted_time_s", float("nan"))) - step * F32_DT) < 1e-10,
             "accepted receipt does not match requested step %d" % step)
        pack_paths.extend((pack, receipt))
        capture_pins[str(step)] = {"pack_sha256": sha(pack), "receipt_sha256": sha(receipt)}

    tracked = {str(Path(p).resolve()): sha(p) for p in static}
    tracked[str(invocation_path.resolve())] = sha(invocation_path)
    if metadata is not None:
        tracked[str(metadata_path.resolve())] = sha(metadata_path)
    for asset_path, digest in assets.items():
        pin(asset_path, digest)
        tracked[str(Path(asset_path).resolve())] = digest
    for path in pack_paths:
        tracked[str(path.resolve())] = sha(path)
    native_log = run / "native.log"
    log_before = sha(native_log) if native_log.is_file() else None
    out.mkdir()
    declaration = {"schema": "numi.human.native-accepted-geometry-audit-1218.declaration.v1",
        "status": "running_actual_captured_geometry_audit", "native_run": str(run),
        "native_invocation_sha256": sha(invocation_path), "native_log_sha256_at_start": log_before,
        "run_metadata_sha256": sha(metadata_path) if metadata_path.is_file() else None,
        "closed_successful_native_run_at_start": bool(full and metadata and metadata.get("exit_code") == 0),
        "capture_steps_requested": list(steps), "full_capture_schedule": list(CAPTURES),
        "capture_times_s": [s * DT for s in steps], "dt_seconds": DT, "root_count": ROOTS,
        "partial_capture_scan": not full, "capture_file_hashes_preflight": capture_pins,
        "candidate_review": {"path": str(REVIEW), "sha256": REVIEW_SHA},
        "NHSKIN": {"path": str(SKIN), "sha256": SKIN_SHA},
        "NHA": {"path": str(NHA), "sha256": NHA_SHA},
        "NHTISS": {"path": str(TISS), "sha256": TISS_SHA},
        "composed_receipt": {"path": str(RECEIPT), "sha256": RECEIPT_SHA},
        "prior_nine_pose_summary": {"path": str(NINE_SUMMARY), "sha256": NINE_SUMMARY_SHA},
        "scope": {"all_859_targets": True, "ocular_monitor_count": len(OCULAR),
            "contact_exemptions": [], "skin_self_and_degenerates": True,
            "stable_63_64_self_topology_outer_envelope": True, "stable_63_64_cross_scan": True},
        "predicate_owners": {"1217_wrapper_path": str(RUN_OWNER), "1217_wrapper_sha256": RUN_OWNER_SHA,
            "1172_scan_owner_path": str(AUDIT_1172), "1172_scan_owner_sha256": AUDIT_1172_SHA,
            "outer_envelope_path": str(SOURCE_CLEARANCE), "outer_envelope_sha256": SOURCE_CLEARANCE_SHA,
            "inputs_before": tracked},
        "scope_note": "Exact predicates over actual accepted Float32 capture coordinates. A subset is partial geometry evidence only. No extrapolated poses, contact exemptions, continuous-time, physiology, or measured-anatomy claim."}
    writej(out / "declaration.json", declaration)
    results = []
    start = time.monotonic()
    try:
        for step in steps:
            result = owner.run_pose(step, run, out, ci, capture_clearance, validator, core,
                                    keys, expected, NHA_SHA)
            muscle = audit_muscles(step, run, out, owner, ci, capture_clearance, validator,
                                   core, outer, source_rows, NHA_SHA)
            result["muscle_geometry"] = muscle
            results.append(result)
            with (out / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, "skin_target_pairs": result["all_skin_crossing_pair_count"],
                    "skin_self_pairs": result["skin_self_crossing_pair_count"],
                    "muscle_self_pairs": [x["self_intersection_count"] for x in muscle["surfaces"]],
                    "inside_skin_counts": [x["inside_skin_vertex_count"] for x in muscle["surfaces"]]},
                    sort_keys=True) + "\n")
                stream.flush()
            gc.collect()
    except BaseException as exc:
        writej(out / "failure.json", {"status": "scan_incomplete", "error_type": type(exc).__name__,
            "error": str(exc), "completed_steps": [r["accepted_step"] for r in results]})
        raise

    after = {path: sha(path) for path in tracked}
    unchanged = tracked == after
    log_after = sha(native_log) if native_log.is_file() else None
    log_changed = log_before != log_after
    complete = full and len(results) == len(CAPTURES) and [r["accepted_step"] for r in results] == list(CAPTURES)
    coverage = complete and all(r["pair_coverage_complete"] for r in results)
    skin_pass = coverage and all(r["all_skin_crossing_pair_count"] == 0
        and r["skin_self_crossing_pair_count"] == 0 and not r["skin_degenerate_face_rows"]
        and r["invalid_target_triangle_count"] == 0 for r in results)
    muscle_pass = coverage and all(r["muscle_geometry"]["all_rows_match_source_face_topology"]
        and r["muscle_geometry"]["all_muscle_self_zero"] and r["muscle_geometry"]["all_outer_envelopes_clear"]
        and r["muscle_geometry"]["stable_63_64_cross_pair_count"] == 0 for r in results)
    closed = bool(metadata and metadata.get("exit_code") == 0 and not log_changed)
    clear = bool(skin_pass and muscle_pass)
    status = ("complete_exact_geometry_clear" if clear and unchanged and closed else
        "complete_with_intersections_or_invalid_geometry" if coverage and unchanged else
        "partial_actual_capture_geometry_only" if not full and unchanged else "incomplete_or_input_changed")
    summary = {"schema": "numi.human.native-accepted-geometry-audit-1218.summary.v1",
        "status": status, "native_run": str(run), "capture_steps_requested": list(steps),
        "full_capture_schedule": list(CAPTURES), "capture_times_s": [s * DT for s in steps],
        "partial_capture_scan": not full, "all_eight_captures_complete": complete,
        "all_859_target_skin_self_pair_coverage_complete": coverage,
        "all_target_and_skin_self_gates_clear": skin_pass,
        "stable_63_64_self_topology_outer_envelope_and_cross_gates_clear": muscle_pass,
        "all_geometry_gates_clear": clear, "native_run_closed_successfully": closed,
        "input_hashes_unchanged": unchanged, "native_log_changed_during_scan": log_changed,
        "native_log_sha256_before": log_before, "native_log_sha256_after": log_after,
        "input_hashes_before": tracked, "input_hashes_after": after,
        "pose_results": results, "elapsed_wall_seconds": time.monotonic() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "qualification": "A full clear result is limited to these eight actual accepted captures; partial scans remain sampled-pose observations. No continuous-time, physiological, clinical, or measured-tissue qualification."}
    writej(out / "summary.json", summary)
    print(json.dumps({"status": status, "steps": list(steps), "complete": complete,
        "all_geometry_gates_clear": clear, "input_hashes_unchanged": unchanged,
        "native_run_closed_successfully": closed, "elapsed_wall_seconds": summary["elapsed_wall_seconds"]},
        sort_keys=True), flush=True)
    return 0 if unchanged and (complete or not full) else 2


if __name__ == "__main__":
    raise SystemExit(main())

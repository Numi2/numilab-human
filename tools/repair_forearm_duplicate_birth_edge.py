#!/usr/bin/env python3
"""Create a pinned row134 local topology repair from the 1228 emitted overlay."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from numilab_human.forearm_topology_patch import collapse_exact_duplicate_edge

D = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
SOURCE_REPORT = D / "forearm-current972-full21-source-emission-1228/report.json"
SOURCE_REPORT_SHA = "6fbe0ed784950fd83f7b57e1c77fe63ca3a5f20651e71c2124d62d4008044008"
WORLD_ROWS = D / "forearm-current972-full21-source-emission-1228/forearm-current972-world-rows.npz"
WORLD_ROWS_SHA = "753adf4e07f94da5e5a1ae43f2b91c26e94b15a9f70a9f4fddf66a4d53c71ecb"
INPUT_REPORT = D / "forearm-current972-shared-collector-inputs-1233/report.json"
INPUT_REPORT_SHA = "4f81da092059947b1492694e29a43637433df03473607b7185c881313b3783c1"
INPUT_OVERLAY = D / "forearm-current972-shared-collector-inputs-1233/row134-full21-world-construction-overlay.npz"
INPUT_OVERLAY_SHA = "405aae201f70b67cb7406ab5a8c465b9bd63ebda3e1d816f52fc4918b2323a3f"
EXPECTED_PAYLOAD_SHA = "44d5dc1c3256962cb85818059ffe1f890cebf864c1f4dbc16ba9b74da4b14b93"
EXPECTED_PARENT_SHA = "aee19bc8e771d6c5ff4d511f6bd75306879705297a73e657336fe34120267331"
KEEP, RETIRE, INCIDENT = 33321, 33322, (12, 66)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4<<20),b""): h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists(): raise SystemExit("refusing to overwrite retained output")
    for path,pin in ((SOURCE_REPORT,SOURCE_REPORT_SHA),(WORLD_ROWS,WORLD_ROWS_SHA),
                     (INPUT_REPORT,INPUT_REPORT_SHA),(INPUT_OVERLAY,INPUT_OVERLAY_SHA)):
        if sha(path)!=pin: raise ValueError(f"input pin mismatch: {path}")
    emission=json.loads(SOURCE_REPORT.read_text())
    if emission.get("composed_child_sha256")!=EXPECTED_PAYLOAD_SHA or emission.get("parent_bindings_extension_sha256")!=EXPECTED_PARENT_SHA:
        raise ValueError("full21 child/parent owner differs from pinned emission")
    with np.load(INPUT_OVERLAY,allow_pickle=False) as z:
        points=z["capture_positions_m"].copy()
        faces=z["capture_faces"].copy()
        gids=z["capture_global_vertex_ids"].copy()
    with np.load(WORLD_ROWS,allow_pickle=False) as z:
        slots=z["row134_source_slots_i64"].copy()
        weights=z["row134_core_weights_float64"].copy()
        canonical_points=z["row134_positions_world_f32"].copy()
        canonical_faces=z["row134_faces_local_u32"].copy()
        canonical_gids=z["row134_canonical_material_gids_u64"].copy()
    if (not np.array_equal(points,canonical_points) or not np.array_equal(faces,canonical_faces)
            or not np.array_equal(gids,canonical_gids)):
        raise ValueError("construction overlay differs from source-emitted row134")
    if points[KEEP].tobytes()!=points[RETIRE].tobytes():
        raise ValueError("pinned duplicate-birth emitted points no longer match")
    if not np.array_equal(slots[[KEEP,RETIRE]],np.asarray([-1,-1],dtype="<i8")):
        raise ValueError("duplicate edge endpoints are no longer inferred birth slots")
    # Keep the lower-index endpoint's entire native 21-term map; never average.
    coeff_delta=float(np.max(np.abs(weights[KEEP]-weights[RETIRE])))
    repaired,collapse=collapse_exact_duplicate_edge(
        faces,points,gids,KEEP,RETIRE,expected_incident_faces=INCIDENT)
    # Preserve complete old-to-new face ancestry, including endpoint substitutions.
    retained = np.ones(len(faces), dtype=bool)
    retained[np.asarray(INCIDENT, dtype=np.int64)] = False
    old_face_indices = np.flatnonzero(retained).astype("<u4")
    new_face_indices = np.arange(len(repaired), dtype="<u4")
    old_face_gids = gids[faces[retained]].astype("<u8", copy=True)
    new_face_gids = gids[repaired].astype("<u8", copy=True)
    if len(old_face_indices) != len(repaired) or not np.array_equal(
            new_face_indices, np.arange(len(repaired), dtype="<u4")):
        raise ValueError("full retained-face ancestry map is incomplete")
    # Audit vertex/material arrays stayed byte-identical, including both full21 rows.
    if (points.tobytes()!=canonical_points.tobytes()
            or gids.tobytes()!=canonical_gids.tobytes()
            or slots.tobytes()!=np.load(WORLD_ROWS,allow_pickle=False)["row134_source_slots_i64"].tobytes()
            or weights.tobytes()!=np.load(WORLD_ROWS,allow_pickle=False)["row134_core_weights_float64"].tobytes()):
        raise ValueError("index-only repair mutated source or material arrays")
    a.output.mkdir(parents=True)
    row132= D / "forearm-current972-shared-collector-inputs-1233/row132-full21-world-construction-overlay.npz"
    out134=a.output/"row134-full21-world-construction-overlay.npz"
    np.savez(out134,capture_positions_m=points,capture_faces=repaired,capture_global_vertex_ids=gids)
    ancestry={
        "schema":"numi.human.forearm-row134-inferred-topology-repair.v1",
        "status":"LOCAL_INDEX_ONLY_PATCH_READY_FOR_STRICT_GEOMETRY_GATE",
        "typed_identity":[51005,134],
        "accepted_capture_pack_sha256":"80fadf29f92ffff1ec7a3e22c70f645b003738720844689668a7081ed0232e7f",
        "accepted_capture_receipt_sha256":"6b24f77959226729d54fddc8365c4d9e89fd932988b6279479df619f611192cd",
        "source_emission_report":{"path":str(SOURCE_REPORT),"sha256":SOURCE_REPORT_SHA},
        "full21_source_payload":{"sha256":EXPECTED_PAYLOAD_SHA,"path":str(emission["composed_child_path"])},
        "parent_binding_extension":{"sha256":EXPECTED_PARENT_SHA,"path":str(emission["parent_bindings_extension_path"])},
        "source_world_rows":{"path":str(WORLD_ROWS),"sha256":WORLD_ROWS_SHA},
        "source_slots_for_both_births":[int(slots[KEEP]),int(slots[RETIRE])],
        "source_point_and_full21_arrays_changed":False,
        "chosen_full21_owner":{"canonical_gid":int(gids[KEEP]),"source_slot":int(slots[KEEP]),
                                "core_weights_float64":weights[KEEP].tolist()},
        "retired_birth":{"canonical_gid":int(gids[RETIRE]),"source_slot":int(slots[RETIRE]),
                         "core_weights_float64_retained_unreferenced":weights[RETIRE].tolist()},
        "full21_coefficient_max_abs_delta_between_duplicate_births":coeff_delta,
        "collapse":collapse,
        "face_ancestry":{
            "removed_emitted_row_face_ordinals":list(INCIDENT),
            "removed_faces_canonical_gid_sets":[sorted(map(int,gids[faces[i]])) for i in INCIDENT],
            "remaining_faces_in_original_order":True,
            "new_face_ancestry_created":False,
        },
        "overlay":{"path":str(out134),"sha256":sha(out134)},
        "row132_overlay":{"path":str(row132),"sha256":"432debef88da94fc74254edd1d757c55b6665502e169657ca2242d7eed7a9acd"},
        "anatomical_admission":False,"native_admission":False,
        "note":"Only the two duplicate-point edge faces were removed and the retired endpoint was substituted in remaining indices. Both full21 source records and all emitted positions/GIDs remain unchanged. This is an inferred local topology patch pending strict self/closed/mutual geometry gates and later source composition.",
    }
    ancestry_npz=a.output/"face-ancestry.npz"
    np.savez_compressed(
        ancestry_npz,
        old_face_indices=old_face_indices,
        new_face_indices=new_face_indices,
        old_face_canonical_gids=old_face_gids,
        new_face_canonical_gids=new_face_gids,
        removed_face_indices=np.asarray(INCIDENT,dtype="<u4"),
        removed_face_canonical_gids=gids[faces[np.asarray(INCIDENT,dtype=np.int64)]].astype("<u8"),
    )
    ancestry["face_ancestry"]={
        "path":str(ancestry_npz),"sha256":sha(ancestry_npz),
        "old_face_to_new_face_ordinal_map_complete":True,
        "old_face_indices":int(len(old_face_indices)),
        "new_face_indices":int(len(new_face_indices)),
        "all_rewritten_faces_have_ordered_old_and_new_canonical_gid_corners":True,
        "removed_faces":[int(x) for x in INCIDENT],
        "removed_face_canonical_gid_corners":gids[faces[np.asarray(INCIDENT,dtype=np.int64)]].astype(int).tolist(),
        "new_face_ancestry_kind":"inferred_local_endpoint_substitution_with_exact_old_and_new_gid_corner_map",
    }
    ancestry["collapse"]["closed_edge_incidence_two_after_collapse"]=True
    ancestry["collapse"]["full_vertex_link_orientation_manifold_admission"]="not_claimed; shared strict verifier owns this gate"
    outreport=a.output/"report.json"
    outreport.write_text(json.dumps(ancestry,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":ancestry["status"],"report":str(outreport),"report_sha256":sha(outreport),
                      "overlay_sha256":sha(out134),"faces_before":len(faces),"faces_after":len(repaired),
                      "coeff_max_delta":coeff_delta,"collapse":collapse},sort_keys=True))
if __name__=="__main__": main()


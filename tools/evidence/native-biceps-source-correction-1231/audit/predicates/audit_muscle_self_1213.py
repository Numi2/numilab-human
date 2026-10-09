#!/usr/bin/env python3
"""Exact self-intersection census for 148 NHTISS muscles at accepted step 0.

Per-surface exact Float32 coordinate quotient only; no coordinate tolerance,
movement, triangle edits, or cross-surface audit. Authored NHTISS4 coordinates
and accepted MRVPACK2 coordinates are audited as distinct coordinate spaces.
"""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

EVID = Path("/Users/n/numi-human-resting-evidence-20261005")
OUT = Path("/Users/n/numi-human-retained-delivery-20261009/passive-muscle-self-audit-1213")
TISSUE = EVID / "passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
MANIFEST = EVID / "passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
RUN = Path("/Users/n/numi-human-retained-delivery-20261009/native-lung1178-thumb1187-smoke-1191/native-run")
PACK = RUN / "accepted-geometry/step-0.mrvpack"
RECEIPT = PACK.with_suffix(".receipt.json")
RUN_META = RUN / "run-metadata.json"
HIST_SRC = Path("/Users/n/numi-human-common-skin-multipose-001/src")
PINNED = {
    TISSUE: "b3d0381f73e05b04ed7ea1ba23eb6559073ad20aa0d9e2ad0f97a18fab3359bd",
    MANIFEST: "82cdd937e0f2daf0a8704fb21246353c38148527f8602d674ac865f935264dde",
    HIST_SRC / "numilab_human/cardiac_cavity_intersections.py": "11f042cc4809a27735c02b66a8fb07ebc92ee66597d7e1814a75231e2cf2d2bb",
    HIST_SRC / "numilab_human/common_atlas_skin_clearance.py": "fbfb5dfbb9b3f7e5fa297a61c793c66cb13fe884d5cebc3a7d1a6b9908961ac1",
    HIST_SRC / "numilab_human/__init__.py": "e06e46ca88bde0b4c4a6235c6135dafdb2e75d27633776dba0098698a082cf42",
    HIST_SRC / "numilab_human/model.py": "caae0f9a64db518712b491780b697c32800b13dba4b5b60ad128b67f5794c0df",
    HIST_SRC / "numilab_human/physiology.py": "efb717aee62893268b266abe6a5c2901b0259333ccbae90110fda2070d84ecef",
    HIST_SRC / "numilab_human/cardiac_cavity_geometry.py": "f6e98744dad9e23cd3b505efc02e7faa618fccce81d2ee08b07f948ef182fb72",
}


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def edge_diagnostics(faces: np.ndarray, vertex_count: int):
    incidences = {}
    parent = list(range(vertex_count))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    duplicate_faces=Counter(tuple(sorted(map(int,f))) for f in faces)
    degenerate_rows=[]
    referenced=set()
    for fi, face in enumerate(faces):
        a,b,c=map(int,face)
        referenced.update((a,b,c))
        if len({a,b,c})<3:
            degenerate_rows.append(fi)
        for x,y in ((a,b),(b,c),(c,a)):
            union(x,y)
            k=(min(x,y),max(x,y))
            incidences.setdefault(k,[]).append(1 if x<y else -1)
    boundary=sum(len(v)==1 for v in incidences.values())
    nonmanifold=sum(len(v)>2 for v in incidences.values())
    orientation_mismatch=sum(len(v)==2 and v[0]==v[1] for v in incidences.values())
    roots={find(int(v)) for v in referenced}
    duplicate_face_rows=sum(n-1 for n in duplicate_faces.values() if n>1)
    return {
        "referenced_vertex_count":len(referenced),
        "unused_quotient_vertex_count":vertex_count-len(referenced),
        "edge_count":len(incidences),
        "boundary_edge_count":int(boundary),
        "nonmanifold_edge_count":int(nonmanifold),
        "orientation_mismatch_edge_count":int(orientation_mismatch),
        "face_component_count":len(roots),
        "degenerate_face_rows":degenerate_rows,
        "duplicate_face_row_count":int(duplicate_face_rows),
        "closed_oriented_manifold_candidate":not (boundary or nonmanifold or orientation_mismatch or degenerate_rows),
    }


def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def sub(a,b): return tuple(x-y for x,y in zip(a,b))


def audit_space(ci, coords, faces, vertex_id_map, original_vertex_count, name):
    coords=np.asarray(coords,dtype="<f4")
    vertex_id_map=np.asarray(vertex_id_map,dtype=np.int64)
    require(coords.ndim==2 and coords.shape[1]==3 and np.isfinite(coords).all(),f"{name}: invalid xyz")
    require(len(vertex_id_map)==len(coords) and len(np.unique(vertex_id_map))==len(vertex_id_map),f"{name}: source vertex identity map invalid")
    # Exact numerical Float32 coordinate quotient per surface; no tolerance.
    # Unreferenced NHTISS rows are recorded separately because MRVPACK omits them.
    quotient, inverse=np.unique(coords,axis=0,return_inverse=True)
    qfaces=inverse[np.asarray(faces,dtype=np.int64)]
    topology=edge_diagnostics(qfaces,len(quotient))
    lattice=[ci.float32_point_lattice_key(tuple(float(x) for x in p)) for p in quotient]
    degenerate=set(topology["degenerate_face_rows"])
    for row,face in enumerate(qfaces):
        if row in degenerate:
            continue
        tri=[lattice[int(v)] for v in face]
        if not any(cross(sub(tri[1],tri[0]),sub(tri[2],tri[0]))):
            degenerate.add(row)
    topology["degenerate_face_rows"] = sorted(degenerate)
    topology["closed_oriented_manifold_candidate"] = not (
        topology["boundary_edge_count"] or topology["nonmanifold_edge_count"] or
        topology["orientation_mismatch_edge_count"] or degenerate)
    good_faces=[tuple(map(int,f)) for i,f in enumerate(qfaces) if i not in degenerate]
    records=ci._records(lattice,good_faces)
    good_rows=[i for i in range(len(qfaces)) if i not in degenerate]
    # Keep the original NHTISS face row as the predicate record identity.
    records=[(r[0],r[1],r[2],good_rows[i],r[4]) for i,r in enumerate(records)]
    result=ci._audit_pair(records,records,same_surface=True)
    return {
        "coordinate_space":name,
        "input_vertex_count":int(original_vertex_count),
        "face_referenced_input_vertex_count":int(len(coords)),
        "unreferenced_input_vertex_count":int(original_vertex_count-len(coords)),
        "exact_f32_quotient_vertex_count":int(len(quotient)),
        "exact_duplicate_coordinate_vertex_count":int(len(coords)-len(quotient)),
        "face_count":int(len(faces)),
        "topology":topology,
        "self_intersection":{
            "aabb_candidate_pairs":int(result["aabb_candidate_pairs"]),
            "allowed_shared_vertex_or_edge_pairs":int(result["allowed_shared_vertex_or_edge_pairs"]),
            "unallowed_pair_count":int(result["count"]),
            "audited_nondegenerate_face_count":int(len(good_faces)),
            "omitted_degenerate_face_count":len(degenerate),
        },
        "unallowed_pairs":[[int(a),int(b)] for a,b in result["triangle_pairs"]],
        "unallowed_pair_witnesses":[{
            "face_rows":[int(a),int(b)],
            "quotient_vertex_ids":[[int(x) for x in inverse[faces[int(a)]]],[int(x) for x in inverse[faces[int(b)]]]],
            "source_vertex_ids":[[int(x) for x in vertex_id_map[faces[int(a)]]],[int(x) for x in vertex_id_map[faces[int(b)]]]],
            "triangle_xyz_f32_m":[np.asarray(coords)[faces[int(a)]].astype(float).tolist(),np.asarray(coords)[faces[int(b)]].astype(float).tolist()],
        } for a,b in result["triangle_pairs"]],
    }


def worker(job):
    # Spawned workers import the pinned exact predicate module from the immutable historical package.
    sys.path.insert(0,str(HIST_SRC))
    from numilab_human import cardiac_cavity_intersections as ci
    sid=job["stable_id"]
    source=audit_space(ci,job["source_vertices"],job["faces"],job["vertex_id_map"],job["nhtiss_vertex_count"],"authored_nhtiss4_float32_owner_local")
    posed=audit_space(ci,job["posed_vertices"],job["faces"],job["vertex_id_map"],job["nhtiss_vertex_count"],"accepted_mrvpack2_step0_float32_world")
    return {"stable_id":sid,"member_id":job["member_id"],"label":job["label"],
            "layer":job["layer"],"body_bindings":job["body_bindings"],
            "nhtiss_vertex_count":job["nhtiss_vertex_count"],
            "nhtiss_face_row_range":[0,int(len(job["faces"]))],
            "captured_pack_face_rows_identity":job["captured_pack_face_rows_identity"],
            "authored_source":source,"accepted_step0":posed}


def main():
    for path,expected in PINNED.items():
        require(path.is_file() and sha(path)==expected,f"pinned source/input mismatch: {path}")
    import importlib
    sys.path.insert(0,str(HIST_SRC))
    from numilab_human import common_atlas_skin_clearance as clearance
    from numilab_human import cardiac_cavity_intersections as ci
    require(Path(clearance.__file__).resolve()==(HIST_SRC/"numilab_human/common_atlas_skin_clearance.py").resolve(),"loader resolved outside historical source")
    require(Path(ci.__file__).resolve()==(HIST_SRC/"numilab_human/cardiac_cavity_intersections.py").resolve(),"predicate resolved outside historical source")
    manifest=json.loads(MANIFEST.read_text())
    require(manifest.get("schema")=="numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1","NHTISS manifest schema")
    rows=manifest["source"]["surfaces"]
    muscle_rows=[r for r in rows if r.get("layer")=="muscle"]
    require(len(muscle_rows)==148 and len({int(r["stable_id"]) for r in muscle_rows})==148,"expected exact 148 muscle surface inventory")
    tissue_raw=TISSUE.read_bytes()
    magic,abi,nrecords,nbindings,nvertices,nindices,fingerprint,archive_sha=struct.unpack_from("<8s6I32s",tissue_raw,0)
    require(magic==b"NHTISS4\0" and abi==5 and nrecords==150,"unsupported NHTISS4 ABI/inventory")
    require(len(tissue_raw)==64+32*nrecords+36*nbindings+56*nvertices+4*nindices,"NHTISS byte ranges")
    tiss_records=np.frombuffer(tissue_raw,dtype="<u4",count=nrecords*8,offset=64).reshape(-1,8)
    vertex_offset=64+32*nrecords+36*nbindings
    source_positions=np.ndarray((nvertices,3),dtype="<f4",buffer=tissue_raw,offset=vertex_offset,strides=(56,4)).copy()
    index_offset=vertex_offset+56*nvertices
    tiss_indices=np.frombuffer(tissue_raw,dtype="<u4",count=nindices,offset=index_offset).copy()
    records_by_id={int(rec[6]):rec for rec in tiss_records if int(rec[7])==1}
    require(set(records_by_id)=={int(r["stable_id"]) for r in muscle_rows},"NHTISS muscle ID inventory differs from manifest")
    muscle_keys={(51005,int(r["stable_id"])) for r in muscle_rows}
    pack_positions,pack_surfaces,pack_counts=clearance._pack_surfaces(PACK,muscle_keys)
    require({k for k in pack_surfaces if k[0]==51005}==muscle_keys,"step0 pack lacks exact muscle primitive IDs")
    receipt=json.loads(RECEIPT.read_text())
    require(receipt.get("accepted_step")==0 and receipt.get("physical_endpoint")=="accepted" and receipt.get("surface_audit_endpoint")=="passed","step0 receipt does not prove accepted state")
    require(Path(receipt.get("accepted_pack_path","")).resolve()==PACK.resolve(),"step0 receipt points to another pack")
    require(receipt.get("pack_file_sha256")==sha(PACK),"step0 receipt pack SHA mismatch")
    runmeta=json.loads(RUN_META.read_text())
    expected_tissue=runmeta.get("asset_sha256",{}).get(str(TISSUE))
    require(expected_tissue==sha(TISSUE),"1191 run metadata does not bind NHTISS payload")
    jobs=[]
    manifest_by_id={int(r["stable_id"]):r for r in muscle_rows}
    for sid in sorted(manifest_by_id):
        mr=manifest_by_id[sid]
        rec=records_by_id[sid]
        first_binding,binding_count,first_vertex,vertex_count,first_index,index_count,stable_id,layer_code=map(int,rec)
        require(stable_id==sid and layer_code==1 and index_count%3==0,"NHTISS muscle record identity malformed")
        source_faces=(tiss_indices[first_index:first_index+index_count].reshape(-1,3)-first_vertex).astype(np.int64)
        captured=pack_surfaces[(51005,sid)]["faces"]
        offsets=captured.astype(np.int64)-source_faces
        require(len(source_faces)==len(captured) and np.all(offsets==offsets.flat[0]),f"NHTISS/captured step0 face row identity differs for {sid}")
        pack_offset=int(offsets.flat[0])
        used_local=np.unique(source_faces)
        compact_faces=np.searchsorted(used_local,source_faces)
        source_v=source_positions[first_vertex+used_local].copy()
        posed_v=pack_positions[pack_offset+used_local].copy()
        require(source_v.shape==posed_v.shape,"source/captured referenced vertex maps differ")
        require(np.array_equal(captured,pack_offset+source_faces),f"captured face vertex identity differs for {sid}")
        jobs.append({"stable_id":sid,"member_id":mr["member_id"],"label":mr["label"],
            "layer":mr["layer"],"body_bindings":mr.get("body_bindings",[]),
            "nhtiss_vertex_count":vertex_count,"faces":compact_faces,"vertex_id_map":used_local,
            "source_vertices":source_v,"posed_vertices":posed_v,
            "captured_pack_face_rows_identity":True})
    start=time.monotonic()
    outcomes=[]
    progress=OUT/"progress.json"
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures={executor.submit(worker,job):job["stable_id"] for job in jobs}
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            outcomes.append(future.result())
            progress.write_text(json.dumps({"completed_surfaces":i,"total_surfaces":148,"completed_ids":sorted(x["stable_id"] for x in outcomes),"elapsed_wall_s":time.monotonic()-start},indent=2,sort_keys=True)+"\n")
            print(f"completed {i}/148 stable_id={futures[future]}",flush=True)
    outcomes.sort(key=lambda r:r["stable_id"])
    pairs_path=OUT/"unallowed-self-pairs.jsonl"
    with pairs_path.open("x") as f:
        for row in outcomes:
            for space_key in ("authored_source","accepted_step0"):
                result=row[space_key]
                for witness in result["unallowed_pair_witnesses"]:
                    f.write(json.dumps({"stable_id":row["stable_id"],"member_id":row["member_id"],"label":row["label"],"coordinate_space":result["coordinate_space"],**witness},separators=(",",":"),allow_nan=False)+"\n")
    summary={}
    for key in ("authored_source","accepted_step0"):
        vals=[r[key] for r in outcomes]
        summary[key]={"surface_count":len(vals),"surfaces_with_unallowed_self_pairs":sum(v["self_intersection"]["unallowed_pair_count"]>0 for v in vals),"unallowed_self_pair_count":sum(v["self_intersection"]["unallowed_pair_count"] for v in vals),"surfaces_with_degenerate_faces":sum(v["self_intersection"]["omitted_degenerate_face_count"]>0 for v in vals),"surfaces_with_open_or_nonmanifold_quotient":sum(not v["topology"]["closed_oriented_manifold_candidate"] for v in vals),"total_aabb_candidates":sum(v["self_intersection"]["aabb_candidate_pairs"] for v in vals),"total_allowed_shared_vertex_or_edge_pairs":sum(v["self_intersection"]["allowed_shared_vertex_or_edge_pairs"] for v in vals),"qualification":"posed/source self-intersection census only; no pairwise surface or whole-body qualification"}
    report={"schema":"numi.human.passive-muscle-self-audit.v1","executed_script":{"path":str(Path(__file__).resolve()),"sha256":sha(Path(__file__).resolve()),"python":sys.version,"argv":sys.argv},"scope":"All148 NHTISS4 muscle surfaces, source geometry and actual accepted MRVPACK2 step0 separately; exact Float32 per-surface coordinate quotient only; ci._audit_pair(same_surface=True). No movement, tolerance weld, cap, cross-surface scan, native step, or GPU work.","inputs":{"nhtiss_payload":{"path":str(TISSUE),"sha256":sha(TISSUE),"abi":abi,"stable_surface_records":nrecords,"binding_count":nbindings,"vertex_count":nvertices,"index_count":nindices,"source_archive_sha256":archive_sha.hex()},"nhtiss_manifest":{"path":str(MANIFEST),"sha256":sha(MANIFEST),"surface_count":len(rows),"muscle_count":len(muscle_rows),"tendon_count":sum(r["layer"]=="tendon" for r in rows)},"accepted_capture":{"pack_path":str(PACK),"pack_sha256":sha(PACK),"receipt_path":str(RECEIPT),"receipt_sha256":sha(RECEIPT),"accepted_step":0,"accepted_time_s":receipt.get("accepted_time_s"),"body_state_sha256":receipt.get("accepted_body_state_sha256"),"run_metadata_path":str(RUN_META),"run_metadata_sha256":sha(RUN_META),"pack_counts":pack_counts},"predicate_and_loader_sources":{str(p):sha(p) for p in PINNED}},"method":{"quotient":"numpy.unique over packed Float32 xyz rows independently for each surface; exact coordinate equality only; faces remapped through returned inverse map with face order unchanged","predicate":"historical common_atlas_skin_clearance._pack_surfaces for MRVPACK2 surfaces plus cardiac_cavity_intersections._audit_pair(...,same_surface=True); exact integer-lattice triangle predicates","face_identity":"NHTISS stable ID and face row preserved; all148 captured pack face-row sequences validated exactly equal to corresponding NHTISS face-row sequence before audit","source_vs_pose":"authored NHTISS Float32 vertex rows are owner-local source geometry; captured MRVPACK Float32 rows are actual accepted step0 world geometry. Their outcomes are kept separate."},"summary":summary,"per_surface":outcomes,"unallowed_pair_ledger":{"path":str(pairs_path),"sha256":sha(pairs_path),"row_count":sum(s[key]["self_intersection"]["unallowed_pair_count"] for s in outcomes for key in ("authored_source","accepted_step0"))},"timing":{"wall_seconds":time.monotonic()-start,"workers":3},"qualification":"geometry-only exact self-intersection census; does not qualify muscle-target clearance, skin, interactions between different muscles, mechanics, or whole-body anatomy"}
    report_path=OUT/"report.json"
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"report":str(report_path),"sha256":sha(report_path),"summary":summary,"pair_ledger_sha256":sha(pairs_path),"elapsed_wall_s":report["timing"]["wall_seconds"]},indent=2),flush=True)

if __name__=="__main__": main()

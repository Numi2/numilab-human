"""Refine current skin/bone contact faces through existing NHSKIN owners.

Offline inferred mesh preparation only. All geometry still requires native
Float32 replay, complete nearby-structure checks and retained-motion admission.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from numilab_human import model as h
from numilab_human import resting_anatomy_verify as v
from numilab_human import skin_source_payload_preflight as sp
from numilab_human import skin_patch_refinement as refinement
from numilab_human import common_atlas_skin_geometry_registration as geometry
from numilab_human import gate_preparation_boundary as gp
# Reuse the tested exact meter-coordinate witness conversion.
from repair_resting_skin_contacts import barycentric

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ("source","authoring-base","source-points","authoring-overlay",
                "witness-dir","registration","capture","receipt","bone-artifact","body-manifest","output"):
        ap.add_argument("--"+key,type=Path,required=True)
    args=ap.parse_args()
    out=args.output
    out.mkdir(exist_ok=False)
    started=time.monotonic()
    paths=[Path(__file__),Path(__file__).with_name("repair_resting_skin_contacts.py"),
           args.source,args.authoring_base,args.source_points,args.authoring_overlay,
           args.registration,args.capture,args.receipt,
           *[args.witness_dir/f"bone-{sid}.json" for sid in (10,12,14,16)],
           *[Path(m.__file__) for m in (h,v,sp,refinement,geometry,gp)]]
    pins={str(p):sha(p) for p in paths}
    raw=args.source.read_bytes()
    source=sp.decode_payload(raw)
    bone_ids,bone_proof=v.load_compiled_bone_identities(args.bone_artifact,args.body_manifest,capture=args.capture)
    if bone_proof["source_archive_sha256"]!=source["source_archive_sha256"]:
        raise ValueError("skin/bone source archive mismatch")
    for name in ("bone_payload","bone_manifest","runtime_manifest","native_invocation"):
        entry=bone_proof[name]
        pins[entry["path"]]=entry["sha256"]
    for sid in (10,12,14,16):
        if sid not in bone_ids:raise ValueError("typed compiled bone identity absent")
    base=np.load(args.authoring_base,allow_pickle=False)
    points=np.load(args.source_points,allow_pickle=False)
    overlay=np.load(args.authoring_overlay,allow_pickle=False)
    faces=source["indices"].reshape(-1,3).astype(np.int64)
    used=np.unique(faces)
    if not np.array_equal(points["full_weights"],source["full_weights"]):
        raise ValueError("full material support changed")
    xyz=points["source_positions_m"]
    if xyz.dtype!=np.dtype("<f4") or xyz.shape!=source["vertices_f"][:,:3].shape or not np.isfinite(xyz).all():
        raise ValueError("source point schema or units invalid")
    compact_faces=np.searchsorted(used,faces)
    if not np.array_equal(compact_faces,base["capture_faces"]) or not np.array_equal(compact_faces,overlay["capture_faces"]):
        raise ValueError("source face ancestry differs from current authoring frame")
    gids=overlay["capture_global_vertex_ids"]
    world=overlay["capture_positions_m"]
    if world.dtype!=np.dtype("<f4") or world.shape!=(len(used),3) or not np.isfinite(world).all():
        raise ValueError("invalid authoring frame positions")
    if gids.dtype.kind not in "iu" or not np.array_equal(gids,base["capture_global_vertex_ids"]) or len(np.unique(gids))!=len(gids):
        raise ValueError("ordered vertex identity differs")
    selected=[]
    for sid in (10,12,14,16):
        witness=json.loads((args.witness_dir/f"bone-{sid}.json").read_text())
        selected.extend(int(a) for a,_ in witness.get("unexpected_triangle_pairs",[]))
    selected=np.unique(np.asarray(selected,dtype=np.int64))
    if not len(selected) or selected.min()<0 or selected.max()>=len(faces):
        raise ValueError("no valid current contact faces")
    receipt=json.loads(args.receipt.read_text())
    poses={r["body_index"]:(np.array(r["position_m"]),np.array(h._myosim_matrix_from_quaternion_xyzw(r["quaternion_xyzw"]))) for r in receipt["accepted_body_poses"]}
    old_affine=np.zeros((len(used),3,3))
    for j,body in enumerate(source["bindings_u"][:,0]):
        _,rot=poses[int(body)]
        bind=source["bindings_f"][j]
        rest=np.array(h._myosim_matrix_from_quaternion_xyzw(bind[4:8].tolist()))*float(bind[8])
        old_affine+=source["full_weights"][used,j,None,None].astype(float)*(rot@rest)
    rebuilt=(base["capture_positions_m"].astype(float)+np.einsum("nij,nj->ni",old_affine,
        (xyz.astype(float)-source["vertices_f"][:,:3].astype(float))[used])).astype("<f4")
    if not np.array_equal(rebuilt.view("<u4"),world.view("<u4")):
        raise ValueError("source override and pinned authoring world differ")
    registration=json.loads(args.registration.read_text())
    matrix=registration["coordinate_system"]["global_source_mm_to_myosim_world_m"]
    trial=bytearray(raw)
    offset=60+36*source["binding_count"]
    for i,p in enumerate(xyz):
        struct.pack_into("<3f",trial,offset+i*56,*p)
    candidate,normal_proof=geometry.compose_disjoint_skin_position_corrections(
        raw,[bytes(trial)],global_source_matrix=matrix)
    refined,proof=refinement.refine_skin_patch(candidate,selected,global_source_matrix=matrix)
    new=sp.decode_payload(refined)
    nv=int(source["vertex_count"])
    births=proof["appended_vertex_births"]
    count=len(births)
    weights=new["full_weights"][nv:].astype(float)
    affine=np.zeros((count,3,3))
    bias=np.zeros((count,3))
    for j,body in enumerate(new["bindings_u"][:,0]):
        t,rot=poses[int(body)]
        bind=new["bindings_f"][j]
        rest=np.array(h._myosim_matrix_from_quaternion_xyzw(bind[4:8].tolist()))*float(bind[8])
        affine+=weights[:,j,None,None]*(rot@rest)
        bias+=weights[:,j,None]*(t+rot@bind[1:4].astype(float))
    desired=np.array([(world[np.searchsorted(used,b["parent_edge_vertex_ids"][0])].astype(float)+world[np.searchsorted(used,b["parent_edge_vertex_ids"][1])].astype(float))*.5 for b in births])
    born=np.linalg.solve(affine,(desired-bias)[...,None])[...,0].astype("<f4")
    trial=bytearray(refined)
    for i,p in enumerate(born):
        struct.pack_into("<3f",trial,offset+(nv+i)*56,*p)
    refined,midpoint_normal_proof=geometry.compose_disjoint_skin_position_corrections(
        refined,[bytes(trial)],global_source_matrix=matrix)
    new=sp.decode_payload(refined)
    new_world=np.concatenate([world,(np.einsum("nij,nj->ni",affine,born.astype(float))+bias).astype("<f4")])
    new_faces=np.searchsorted(np.unique(new["indices"]),new["indices"].reshape(-1,3))
    new_gids=refinement.appended_material_vertex_ids(gids,count)
    triangles=new_world[new_faces].astype(float)
    fn=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    if np.any(np.linalg.norm(fn,axis=1)==0):
        raise ValueError("refined authoring frame has degenerate face")
    normals=np.zeros_like(new_world,dtype=float)
    for corner in range(3):
        np.add.at(normals,new_faces[:,corner],fn)
    lengths=np.linalg.norm(normals,axis=1)
    if np.any(lengths==0):
        raise ValueError("refined authoring frame has undefined vertex normal")
    normals=(normals/lengths[:,None]).astype("<f4")
    proof.update(source_pins=pins,source_position_override_sha256=sha(args.source_points),
        current_authoring_normal_owner=normal_proof,midpoint_normal_owner=midpoint_normal_proof,
        source_face_parent_domain="immediate input NHSKIN face ordinals, hash-bound",
        maximum_midpoint_reconstruction_error_m=float(np.linalg.norm(new_world[-count:].astype(float)-desired,axis=1).max()),
        exact_emission=False,native_admission=False)
    prep=dict(status="PASS_CONTACT_PATCH_PREPARATION",predicate_phase_started=False,
              source_pins=pins,compiled_bone_identity=bone_proof,
              bone_target_identities=[bone_ids[sid] for sid in (10,12,14,16)],
              selected_faces=len(selected),birth_count=count,
              units="metres",exact_emission=False,anatomical_admission=False,
              scope="Offline common-atlas authoring approximation; native replay pending.")
    frozen=gp.freeze_preparation(out/"preparation.json",prep,
        runner_argv=[sys.executable,*sys.argv],input_pins=pins)
    gp.verify_frozen_preparation(frozen)
    (out/"refined-source.nhskin").write_bytes(refined)
    (out/"refinement-proof.json").write_text(json.dumps(proof,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(out/"refined-authoring-base.npz",capture_positions_m=new_world,
        capture_faces=new_faces,capture_normals=normals,capture_global_vertex_ids=new_gids)
    np.savez_compressed(out/"diagnostic.npz",source_positions_m=new["vertices_f"][:,:3],full_weights=new["full_weights"])
    skin=v.prepare_surface((51007,1),new_world,new_faces,normals=normals,source_vertex_ids=new_gids)
    self_result=skin.self_audit()
    (out/"self.json").write_text(json.dumps(self_result,indent=2)+"\n")
    if self_result["count"]:
        raise ValueError("refined authoring frame self-intersects")
    cap,identity=v.load_capture(args.capture,args.receipt)
    contacts=[]
    inventory=[]
    for sid in (10,12,14,16):
        pos,bf,bn,bg=cap[(51004,sid)]
        bone=v.prepare_surface((51004,sid),pos,bf,normals=bn,source_vertex_ids=bg)
        if not bone.closed:
            raise ValueError("target is not a closed solid")
        result=v.audit_surface_pair(skin,bone)
        (out/f"bone-{sid}.json").write_text(json.dumps(result,indent=2)+"\n")
        pairs=result.get("unexpected_triangle_pairs",[])
        inventory.append(dict(bone=sid,pair_count=len(pairs)))
        tri=pos[bf].astype(float)
        volume=np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
        if not volume:raise ValueError("target volume zero")
        for sf,bi in pairs:
            normal=np.cross(tri[bi,1]-tri[bi,0],tri[bi,2]-tri[bi,0])
            normal*=np.sign(volume)/np.linalg.norm(normal)
            for point in skin.intersection_points_meters(bone,int(sf),int(bi)):
                contacts.append((sf,sid,bi,barycentric(skin.triangle_meters(int(sf)),point),normal))
    if not contacts:raise ValueError("no remaining contacts; use native replay directly")
    np.savez_compressed(out/"exact-contact-cache.npz",
        skin_face_ids=np.array([x[0] for x in contacts],dtype=np.int64),
        target_ids=np.array([x[1] for x in contacts],dtype=np.int64),
        target_face_ids=np.array([x[2] for x in contacts],dtype=np.int64),
        barycentrics=np.array([x[3] for x in contacts]),outward_normals=np.array([x[4] for x in contacts]))
    report=dict(status="PASS_LOCAL_REFINEMENT_AUTHORING_ONLY",source_pins=pins,
        capture_identity=identity,inventory=inventory,selected_faces=len(selected),
        birth_count=count,output_faces=len(new_faces),elapsed_seconds=time.monotonic()-started,
        anatomical_admission=False,native_admission=False,exact_emission=False,
        next_change="Bounded contact construction on this source; then actual GPU replay, all-near and retained-motion gates.")
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report),flush=True)
    return 0

if __name__=="__main__":
    try:
        result=main()
    except Exception as error:
        import traceback
        if "--output" in sys.argv:
            directory=Path(sys.argv[sys.argv.index("--output")+1])
            if directory.is_dir():
                failure=dict(status="FAIL_CONTACT_PATCH_REFINEMENT",
                    error=repr(error),traceback=traceback.format_exc(),
                    argv=sys.argv,runner_sha256=sha(Path(__file__)),
                    anatomical_admission=False,native_admission=False,
                    next_change="Repair the retained preparation/self/contact witness before any broader gate.")
                with (directory/"failure.json").open("x") as stream:
                    stream.write(json.dumps(failure,indent=2)+"\n")
        raise
    raise SystemExit(result)


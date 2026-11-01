#!/usr/bin/env python3
"""Focused tiny controls for the 949 append-only source namespace helpers."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OWNER = HERE / "verify_current_source_evidence_949_append.py"
spec = importlib.util.spec_from_file_location("verify949_controls", OWNER)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load 949 source owner")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def rejects(name, fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(name + " was accepted")


def main():
    parent_f64 = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float64)
    parent_f32 = parent_f64.astype(np.float32)
    midpoint = np.asarray(((0.5, 0.0, 0.0),), dtype=np.float32)
    candidate_f64 = np.vstack((parent_f64, midpoint.astype(np.float64)))
    candidate_f32 = np.vstack((parent_f32, midpoint))
    V.assert_append_prefix(parent_f64, parent_f32, candidate_f64, candidate_f32, midpoint, "valid")

    changed_prefix = candidate_f64.copy()
    changed_prefix[0, 0] = 1.0e-9
    rejects("changed old full-precision row", lambda: V.assert_append_prefix(
        parent_f64, parent_f32, changed_prefix, candidate_f32, midpoint, "bad-prefix"))

    wrong_midpoint = candidate_f32.copy()
    wrong_midpoint[-1, 0] = np.float32(0.625)
    rejects("wrong midpoint", lambda: V.assert_append_prefix(
        parent_f64, parent_f32, candidate_f64, wrong_midpoint, midpoint, "bad-midpoint"))

    rejects("wrong parent pin", lambda: V.assert_parent_pin(
        {"path": "/tmp/parent-a", "sha256": "a" * 64},
        {"path": "/tmp/parent-b", "sha256": "a" * 64}, "parent"))

    rejects("edited O source", lambda: V.assert_outer_source_unchanged(
        {"O": {"path": "/tmp/O.npz", "sha256": "b" * 64}},
        {"O": {"path": "/tmp/O.npz", "sha256": "c" * 64}}))

    base_points=np.asarray(((0.,0.,0.),(1.,0.,0.),(0.,1.,0.)),np.float32)
    base_surface={
        "points_world_m":base_points.astype(np.float64),"points_world_m_f32":base_points,
        "faces":np.asarray(((0,1,2),),np.int64),
        "source_input_face_rows":np.asarray((0,),np.int64),
        "quality_star_face_output_to_input_row":np.asarray((17,),np.int64),
        "quality_star_point_output_to_input_row":np.asarray((0,1,2),np.int64),
        "source_input_vertex_quotient":np.asarray((0,1,2),np.int64),
        "endpoint_quotient_output_to_current_parent_rows":np.asarray((0,1,2),np.int64),
        "quality_star_patch_union_offsets":np.asarray((0,1),np.int64),
        "quality_star_patch_union_face_source_layer":np.asarray((2,),np.int64),
        "face_ancestry_offsets":np.asarray((0,1),np.int64),
        "face_ancestry_source_original_ids":np.asarray((91,),np.int64),
        "face_source_layer":np.asarray((2,),np.int64),
        "source_collapse_operation_rows":np.asarray((8,),np.int64),
        "vertex_temperature":np.asarray((0.,2.,4.),np.float32),
        "typed_parent_candidate_path":np.asarray(("/tmp/old-I.npz",)),
        "typed_parent_candidate_sha256":np.asarray(("a"*64,)),
    }
    ops=[]; points=base_points.copy(); faces=base_surface["faces"].copy()
    for oi,edge in enumerate(((0,1),(0,3))):
        hit=np.flatnonzero(np.isin(faces,edge).sum(axis=1)==2)
        parents=[];new_faces=[]
        for row,face in enumerate(faces):
            f=list(map(int,face))
            if edge[0] in f and edge[1] in f:
                w=next(v for v in f if v not in edge)
                for j in range(3):
                    u,v=f[j],f[(j+1)%3]
                    if (u,v)==edge or (u,v)==edge[::-1]:
                        new_faces.extend(((u,len(points),w),(len(points),v,w)));parents.extend((row,row));break
            else:new_faces.append(tuple(f));parents.append(row)
        x=((points[edge[0]].astype(np.float64)+points[edge[1]].astype(np.float64))*.5).astype(np.float32)
        ops.append({"surface_edge_rows":{"I":list(edge)},"fraction_by_surface":{"I":.5},
                    "surface_parent_face_rows":{"I":hit.tolist()}})
        points=np.vstack((points,x));faces=np.asarray(new_faces,np.int64)
    final_surface={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in base_surface.items()}
    final_surface["points_world_m_f32"]=points
    final_surface["points_world_m"]=points.astype(np.float64)
    final_surface["faces"]=faces
    final_surface["source_input_face_rows"]=np.asarray((0,0,0),np.int64)
    final_surface["edge_bisection_output_face_to_immediate_parent_row"]=np.asarray((0,0,0),np.int64)
    final_surface["quality_star_face_output_to_input_row"]=np.asarray((17,17,17),np.int64)
    final_surface["quality_star_point_output_to_input_row"]=np.asarray((0,1,2,-1,-1),np.int64)
    final_surface["endpoint_quotient_output_to_current_parent_rows"]=np.asarray((0,1,2,-1,-1),np.int64)
    final_surface["quality_star_patch_union_offsets"]=np.asarray((0,1,2,3),np.int64)
    final_surface["quality_star_patch_union_face_source_layer"]=np.asarray((2,2,2),np.int64)
    final_surface["face_ancestry_offsets"]=np.asarray((0,1,2,3),np.int64)
    final_surface["face_ancestry_source_original_ids"]=np.asarray((91,91,91),np.int64)
    final_surface["face_source_layer"]=np.asarray((2,2,2),np.int64)
    final_surface["source_collapse_operation_rows"]=np.asarray([],np.int64)
    final_surface["vertex_temperature"]=np.asarray((0.,2.,4.,1.,.5),np.float32)
    final_surface["typed_parent_candidate_path"]=np.asarray(("/tmp/old-I.npz",))
    final_surface["typed_parent_candidate_sha256"]=np.asarray(("a"*64,))
    final_surface["edge_bisection_new_point_parent_edge_rows"]=np.asarray(((0,1),(0,3)),np.int64)
    final_surface["edge_bisection_new_point_parent_edge_fraction"]=np.asarray((.5,.5),np.float64)
    final_surface["edge_bisection_new_point_parent_edge_support_residual_m"]=np.asarray((0.,0.),np.float64)
    final_surface["edge_bisection_new_point_parent_surface_name"]=np.asarray(("I","I"),dtype="U8")
    final_surface["edge_bisection_new_point_parent_immediate_face_offsets"]=np.asarray((0,1,2),np.int64)
    final_surface["edge_bisection_new_point_parent_immediate_face_rows"]=np.asarray((0,0),np.int64)
    final_surface["edge_bisection_batch_journal_path"]=np.asarray(("/tmp/journal.json",))
    parent_pin={"path":"/tmp/old-I.npz","sha256":"a"*64}
    journal_pin={"path":"/tmp/journal.json","sha256":"b"*64}
    V._verify_surface_metadata_lineage(base_surface,final_surface,"I",ops,parent_pin,journal_pin)
    bad_csr={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in final_surface.items()}
    bad_csr["face_ancestry_source_original_ids"][1]=92
    rejects("tampered ancestry CSR",lambda:V._verify_surface_metadata_lineage(base_surface,bad_csr,"I",ops,parent_pin,journal_pin))
    bad_parent={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in final_surface.items()}
    bad_parent["edge_bisection_new_point_parent_immediate_face_rows"][0]=1
    rejects("wrong appended point face parent",lambda:V._verify_surface_metadata_lineage(base_surface,bad_parent,"I",ops,parent_pin,journal_pin))

    vbase={
        "points_world_m_f32":np.zeros((4,3),np.float32),
        "output_point_to_input_point_row":np.arange(4,dtype=np.int64),
        "output_node_to_input_node":np.arange(4,dtype=np.int64),
        "input_node_to_output_node":np.arange(4,dtype=np.int64),
        "output_node_to_input_node_rows":np.arange(4,dtype=np.int64),
        "output_point_to_input_point_rows":np.arange(4,dtype=np.int64),
        "input_node_to_output_node_rows":np.arange(4,dtype=np.int64),
        "source_interior_vertex_quotient":np.asarray((0,1,-1,3),np.int64),
        "source_input_vertex_quotient":np.asarray((0,1,2,3),np.int64),
        "source_I_vertex_quotient":np.asarray((0,1,2,3),np.int64),
        "vertex_temperature":np.asarray((0.,2.,4.,6.),np.float32),
    }
    vops=[{"edge_volume_rows_by_role":{"Skin_current":[0,1]},"fraction_by_role":{"Skin_current":.5}}]
    vfinal={k:v.copy() for k,v in vbase.items()}
    vfinal["points_world_m_f32"]=np.vstack((vbase["points_world_m_f32"],np.zeros((1,3),np.float32)))
    for key in ("output_point_to_input_point_row","output_node_to_input_node",
                "output_node_to_input_node_rows","output_point_to_input_point_rows"):
        vfinal[key]=np.r_[vbase[key],-1]
    vfinal["source_interior_vertex_quotient"]=np.r_[vbase["source_interior_vertex_quotient"],-1]
    vfinal["source_input_vertex_quotient"]=np.r_[vbase["source_input_vertex_quotient"],-1]
    vfinal["source_I_vertex_quotient"]=np.r_[vbase["source_I_vertex_quotient"],-1]
    vfinal["vertex_temperature"]=np.r_[vbase["vertex_temperature"],1.]
    V._verify_volume_point_metadata(vbase,vfinal,"Skin_current",vops)
    bad_quotient={k:v.copy() for k,v in vfinal.items()}
    bad_quotient["source_interior_vertex_quotient"][1]=99
    rejects("changed inherited volume quotient",lambda:V._verify_volume_point_metadata(vbase,bad_quotient,"Skin_current",vops))

    map_base={
        "points_world_m_f32":np.zeros((4,3),np.float32),
        "skin_current_target_rebind_O_point_to_volume_rows":np.asarray((0,1,2),np.int64),
        "skin_current_target_rebind_I_point_to_volume_rows":np.asarray((1,2,3),np.int64),
    }
    map_final={k:v.copy() for k,v in map_base.items()}
    map_final["skin_current_target_rebind_I_point_to_volume_rows"]=np.asarray((1,2,3,4),np.int64)
    map_final["typed_O_candidate_path"]=np.asarray((str(Path("/tmp/O.npz").resolve()),))
    map_final["typed_O_candidate_sha256"]=np.asarray(("c"*64,))
    map_final["typed_I_candidate_path"]=np.asarray((str(Path("/tmp/I2.npz").resolve()),))
    map_final["typed_I_candidate_sha256"]=np.asarray(("d"*64,))
    map_ops=[{"edge_volume_rows_by_role":{"Skin_current":[1,2]},
              "volume_parent_cell_rows":{"Skin_current":[0]},"surface_edge_rows":{"I":[0,1]}}]
    state={"source_surfaces":{"O":{"path":"/tmp/O.npz","sha256":"c"*64},
                               "I":{"path":"/tmp/I2.npz","sha256":"d"*64}}}
    V._verify_volume_source_maps(map_base,map_final,"Skin_current",map_ops,state,Path("/tmp/state.json"))
    bad_map={k:v.copy() for k,v in map_final.items()}
    bad_map["skin_current_target_rebind_I_point_to_volume_rows"][-1]=2
    rejects("wrong current source-to-volume append row",lambda:V._verify_volume_source_maps(
        map_base,bad_map,"Skin_current",map_ops,state,Path("/tmp/state.json")))

    print("PASS: append/midpoint/parent/O controls; multi-op face/CSR replay; volume-map suffix, source crosswalk, and inherited-quotient controls")


if __name__ == "__main__":
    main()

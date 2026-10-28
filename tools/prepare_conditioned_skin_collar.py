#!/usr/bin/env python3
"""Reindex an existing skin/fat collar onto conditioned O; no volume admission."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def boundary(faces):
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    ordered=np.sort(edges,axis=1);unique,inverse,count=np.unique(ordered,axis=0,return_inverse=True,return_counts=True)
    if np.any(count>2):raise ValueError("nonmanifold O boundary")
    directed=edges[count[inverse]==1]
    return {tuple(sorted(map(int,e))):tuple(map(int,e)) for e in directed}

def prepare(candidate,collar,face_csv,local_repair_reference=None,local_repair_reference_sha256=None):
    points=np.asarray(candidate["points_world_m"],np.float32)
    faces=np.asarray(candidate["source_O_faces"],np.int64)
    cp=np.asarray(collar["points_world_m"],np.float32)
    old_faces=np.asarray(collar["outer_faces"],np.int64)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError("invalid candidate points")
    if faces.ndim!=2 or faces.shape[1]!=3 or faces.size==0 or faces.min()<0 or faces.max()>=len(points):raise ValueError("invalid candidate faces")
    tri=points.astype(np.float64)[faces]
    if np.any(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)==0):raise ValueError("degenerate conditioned triangle")
    ni=len(collar["donor_I_parent_node_rows"]);nd=len(collar["donor_D_parent_node_rows"])
    old_n=len(cp)-ni-nd
    local_repair_report=None
    reference_points=points
    if local_repair_reference is not None:
        from validate_connected_skin_local_repair import validate_local_repair
        if local_repair_reference_sha256!="5e686b14d9360267f4b40bbeb5ed6c9c68bf871933874b5aa8eaa5898757ec2b":
            raise ValueError("local collar repair requires the pinned conditioned reference")
        local_repair_report=validate_local_repair(local_repair_reference,candidate,local_repair_reference_sha256)
        reference_points=np.asarray(local_repair_reference["points_world_m"],np.float32)
    if old_n!=len(points) or not np.array_equal(cp[:old_n],reference_points):raise ValueError("conditioned O must preserve exact original point table")
    old_boundary=boundary(old_faces);new_boundary=boundary(faces)
    if old_boundary.keys()!=new_boundary.keys():raise ValueError("conditioned O changed retained head boundary edges")
    same=sum(old_boundary[k]==new_boundary[k] for k in old_boundary)
    if same not in (0,len(old_boundary)):raise ValueError("conditioned O changed only part of the head boundary orientation")
    global_reversal=bool(old_boundary) and same==0
    used=np.unique(faces);inverse=np.full(len(points),-1,np.int64);inverse[used]=np.arange(len(used))
    fields=list(face_csv[0]) if face_csv else []
    expected=["face","v0","v1","v2","O033_source_face_parent_row","baseline806_source_face_row_exact"]
    if fields!=expected or len(face_csv)!=len(faces):raise ValueError("conditioned registered face CSV schema/count mismatch")
    exact=[]
    for i,row in enumerate(face_csv):
        if int(row["face"])!=i or [int(row[f"v{c}"]) for c in range(3)]!=inverse[faces[i]].tolist():raise ValueError("registered face CSV is not exact current compact topology")
        v=int(row["baseline806_source_face_row_exact"])
        if v < -1:raise ValueError("invalid exact source face identity")
        exact.append(v)
    off=np.asarray(candidate["face_ancestry_offsets"],np.int64)
    keys=("face_ancestry_parent_face_rows","face_ancestry_source_original_ids","face_ancestry_source_face_rows")
    arrays=[np.asarray(candidate[k],np.int64) for k in keys]
    if off.shape!=(len(faces)+1,) or off[0]!=0 or np.any(np.diff(off)<=0) or any(len(a)!=off[-1] for a in arrays):raise ValueError("conditioned full face ancestry CSR malformed")
    if any(np.any(a<0) for a in arrays):raise ValueError("conditioned ancestry contains missing typed identities")
    out={k:np.asarray(collar[k]).copy() for k in collar.files}
    out["points_world_m"]=np.vstack([points[used],cp[old_n:]])
    out["outer_faces"]=inverse[faces]
    for key in ("skin_sidewall_faces","fat_sidewall_faces"):
        old=np.asarray(collar[key],np.int64)
        if old.ndim!=2 or old.shape[1]!=3 or np.any(old<0) or np.any(old>=len(cp)):raise ValueError("invalid collar sidewall corners")
        new=old.copy();mask=old<old_n
        mapped=inverse[old[mask]]
        if np.any(mapped<0):raise ValueError("conditioned O removed a physical collar vertex")
        new[mask]=mapped;new[~mask]=old[~mask]-old_n+len(used)
        if not np.array_equal(cp[old],out["points_world_m"][new]):raise ValueError("collar sidewall geometry changed during reindexing")
        out[key]=new
    ids=np.asarray(collar["donor_O_to_candidate_vertex"],np.int64)
    if np.any(ids<0) or np.any(ids>=old_n) or np.any(inverse[ids]<0):raise ValueError("collar donor O map is incomplete")
    out["donor_O_to_candidate_vertex"]=inverse[ids].astype(np.uint32)
    out["exact_candidate_face_source_rows"]=np.asarray(exact,np.int64)
    out["face_ancestry_offsets"]=off.copy()
    for k,a in zip(keys,arrays):out[k]=a.copy()
    out["compact_O_to_candidate_vertex_row"]=used
    # Geometry source aliases are distinct from the O033 full parent table.
    outer={k:np.asarray(candidate[k]).copy() for k in candidate.files}
    outer["points_world_m"]=points[used];outer["source_O_faces"]=inverse[faces]
    outer["faces"]=outer["source_O_faces"]
    for key in ("candidate_vertex_to_O033_row",):
        if key in outer:outer[key]=outer[key][used]
    if "O033_to_candidate_vertex_row" in outer:
        old=outer["O033_to_candidate_vertex_row"]
        if np.any(old<0) or np.any(old>=len(inverse)) or np.any(inverse[old]<0):raise ValueError("conditioned endpoint contraction map is not retained")
        outer["O033_to_candidate_vertex_row"]=inverse[old]
    return outer,out,{"original_O_point_rows":old_n,"current_O_point_rows":len(used),
        "current_O_faces":len(faces),"sidewall_faces":len(out["skin_sidewall_faces"])+len(out["fat_sidewall_faces"]),
        "boundary_edges":len(boundary(faces)),"O_boundary_globally_reversed":global_reversal,"full_ancestry_entries":int(off[-1]),
        "source_original_ids":np.unique(arrays[1]).tolist(),"sidewall_coordinates_exact":True,
        "local_repair":local_repair_report,"admitted":False,"physical_steps":0}

def main():
    a=argparse.ArgumentParser();a.add_argument("--candidate",type=Path,required=True);a.add_argument("--collar",type=Path,required=True);a.add_argument("--registered-faces",type=Path,required=True);a.add_argument("--wall-csv",type=Path,required=True);a.add_argument("--output",type=Path,required=True);a.add_argument("--local-repair-reference",type=Path);args=a.parse_args()
    with args.registered_faces.open(newline="") as f:rows=list(csv.DictReader(f))
    reference=np.load(args.local_repair_reference,allow_pickle=False) if args.local_repair_reference else None
    reference_sha=sha(args.local_repair_reference) if args.local_repair_reference else None
    outer,collar,receipt=prepare(np.load(args.candidate,allow_pickle=False),np.load(args.collar,allow_pickle=False),rows,reference,reference_sha)
    with args.wall_csv.open(newline="") as f:
        reader=csv.DictReader(f);wall_header=reader.fieldnames;wall_rows=list(reader)
    old=np.load(args.collar,allow_pickle=False);old_walls=np.vstack([old["skin_sidewall_faces"],old["fat_sidewall_faces"]]);new_walls=np.vstack([collar["skin_sidewall_faces"],collar["fat_sidewall_faces"]])
    if len(wall_rows)!=len(old_walls):raise ValueError("wall CSV count differs from retained collar")
    for i,row in enumerate(wall_rows):
        if int(row["wall_face_row"])!=i or [int(row[f"output_node{c}"]) for c in range(3)]!=old_walls[i].tolist():raise ValueError("wall CSV is not exact original collar topology")
        for c in range(3):row[f"output_node{c}"]=int(new_walls[i,c])
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/"conditioned-compact-O.npz",**outer)
    np.savez_compressed(args.output/"conditioned-head-collar.npz",**collar)
    with (args.output/"donor-sidewall-faces.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=wall_header,lineterminator="\n");writer.writeheader();writer.writerows(wall_rows)
    receipt["inputs"]={str(p):sha(p) for p in (args.candidate,args.collar,args.registered_faces,args.wall_csv)}
    if args.local_repair_reference:receipt["inputs"][str(args.local_repair_reference)]=reference_sha
    receipt["outputs"]={p.name:sha(p) for p in args.output.iterdir() if p.is_file()}
    (args.output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))
if __name__=="__main__":main()

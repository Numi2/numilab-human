"""Reindex the existing paired collar using canonical head identities."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def key(p):return np.asarray(p,dtype="<f4").tobytes()
def triangles(points,faces):
    return {tuple(sorted(key(p) for p in tri)) for tri in np.asarray(points,np.float32)[faces]}
def prepare(outer,protected,source_collar,skin_closed,fat_closed,output,receipt):
    paths=list(map(Path,(outer,protected,source_collar,skin_closed,fat_closed,output,receipt)))
    outer,protected,source_collar,skin_closed,fat_closed,output,receipt=paths
    if output.exists() or receipt.exists():raise FileExistsError("fresh collar output required")
    a=arrays(outer);c=arrays(source_collar)
    points=np.asarray(a["points_world_m"],np.float32);faces=np.asarray(a["faces"],np.int64)
    with protected.open(newline="") as f:rows=list(csv.DictReader(f))
    if len(rows)!=149:raise ValueError("canonical149 inventory incomplete")
    heads={(r["role"],int(r["base_volume_node_row"])):r for r in rows if r["mapping_kind"]=="head_ring_boundary_target"}
    if len(heads)!=135:raise ValueError("canonical head inventory is not135 identities")
    lookup={}
    for i,p in enumerate(points):lookup.setdefault(key(p),[]).append(i)
    xyz={};o_ids=[];mapped_heads=[]
    for role in ("O","I","D"):
        targets=[]
        for parent in np.asarray(c["donor_"+role+"_parent_node_rows"],np.int64):
            r=heads[(role,int(parent))]
            target=np.asarray([float(r["target_world_"+axis+"_m"]) for axis in "xyz"],np.float32)
            targets.append(target)
            if role=="O":
                matches=lookup.get(key(target),[])
                if len(matches)!=1:raise ValueError("canonical O head lacks unique current F32 vertex")
                o_ids.append(matches[0])
            mapped_heads.append({"role":role,"parent_node_row":int(parent),"canonical_protected_row":int(r["canonical_protected_row"]),"target_world_m_f32":target.tolist()})
        if len(targets)!=45:raise ValueError("45 ring identities per role required")
        xyz[role]=np.asarray(targets,np.float32)
    old_no=len(c["points_world_m"])-90;new_no=len(points)
    remap={int(old):int(new) for old,new in zip(c["donor_O_to_candidate_vertex"],o_ids)}
    remap.update({old_no+i:new_no+i for i in range(90)})
    skin=np.asarray([[remap[int(v)] for v in tri] for tri in c["skin_sidewall_faces"]],np.int64)
    fat=np.asarray([[remap[int(v)] for v in tri] for tri in c["fat_sidewall_faces"]],np.int64)
    newpoints=np.vstack((points,xyz["I"],xyz["D"]))
    ss,ff=arrays(skin_closed),arrays(fat_closed)
    if triangles(newpoints,skin)!=triangles(ss["points_world_m_f64"],ss["faces"][-90:]):
        raise ValueError("canonical skin collar differs from the retained fine rim")
    if triangles(newpoints,fat)!=triangles(ff["points_world_m_f64"],ff["faces"][-90:]):
        raise ValueError("canonical fat collar differs from the retained fine rim")
    c.update(points_world_m=newpoints,outer_faces=faces,
             donor_O_to_candidate_vertex=np.asarray(o_ids,np.uint32),
             skin_sidewall_faces=skin,fat_sidewall_faces=fat,
             exact_candidate_face_source_rows=np.full(len(faces),-1,np.int64))
    np.savez_compressed(output,**c)
    result={"status":"PASS_EXISTING_PAIRED_RIMS_EXACT_F32_REINDEX",
            "inputs":{str(f):sha(f) for f in paths[:5]},"output":str(output),"output_sha256":sha(output),
            "skin_rim_facets":len(skin),"fat_rim_facets":len(fat),"mapped_head_rows":mapped_heads,
            "retained_fine_rim_geometry_identical":True,"physical_admission":False,"runtime":0,"accepted_steps":0,
            "owner_sha256":sha(__file__)}
    receipt.write_text(json.dumps(result,indent=2)+"\n")
    return result
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for n in ("outer","protected","source-collar","skin-closed","fat-closed","output","receipt"):ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();r=prepare(a.outer,a.protected,a.source_collar,a.skin_closed,a.fat_closed,a.output,a.receipt)
    print(json.dumps({k:v for k,v in r.items() if k!="mapped_head_rows"}))

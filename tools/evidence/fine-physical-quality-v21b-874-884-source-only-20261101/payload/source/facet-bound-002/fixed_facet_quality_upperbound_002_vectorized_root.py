#!/usr/bin/env python3
"""Read-only native-q upper bounds for bad tetrahedra incident to fixed boundary facets."""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np

H=Path(__file__).resolve().parents[1]
OWNER=H/"coupled-anchor-collapse-780-root/ordinary-allrole-v25e/verify_targeted_inner_back_remainder_v25e_witnessaware.py"
OWNER_SHA="c24e9449153b63c6f21a0798188fa45e1205f430a03dbf3e081ab75485946e91"
Q_TARGET=1.e-5
PAT=np.asarray(((1,2,3),(0,3,2),(0,1,3),(0,2,1)),np.int64)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def pin(path):
    p=Path(path).resolve()
    return {"path":str(p),"sha256":sha(p)}

def load_owner():
    actual=sha(OWNER)
    if actual!=OWNER_SHA:raise ValueError("pinned parent-domain/source-map owner SHA mismatch: "+actual)
    spec=importlib.util.spec_from_file_location("facet_audit_v25e_verifier",OWNER)
    if spec is None or spec.loader is None:raise RuntimeError("cannot import pinned parent-domain owner")
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod

def facet_upper_bound(points,tri):
    """For fixed face ABC, q=det6/L^3 <= 2*area(ABC)/L_face^2 for every apex."""
    f=np.asarray(tri,dtype=np.int64)
    a,b,c=np.asarray(points[f],dtype=np.float64)
    area=0.5*float(np.linalg.norm(np.cross(b-a,c-a)))
    L=max(float(np.linalg.norm(b-a)),float(np.linalg.norm(c-b)),float(np.linalg.norm(a-c)))
    qbound=2.*area/(L*L) if L>0 else 0.
    return area,L,qbound

def _surface_face_roles(volume_role,surfaces,smap,vmap):
    allowed={"Skin":("O","I"),"Fat":("I","D")}[volume_role]
    max_global=max(int(np.max(vmap[volume_role],initial=-1)),max(int(np.max(smap[r],initial=-1)) for r in allowed))
    inv=np.full(max_global+1,-1,np.int64)
    vm=np.asarray(vmap[volume_role],np.int64)
    if len(np.unique(vm))!=len(vm):raise ValueError(volume_role+" volume local-to-global point map is not injective")
    inv[vm]=np.arange(len(vm),dtype=np.int64)
    lookup={};per_role={}
    for role in allowed:
        sf=np.asarray(surfaces[role]["faces"],np.int64)
        sm=np.asarray(smap[role],np.int64)
        gf=sm[sf]
        if np.any(gf<0) or (gf.size and int(gf.max())>max_global):
            raise ValueError(role+" active source face has an unmapped/out-of-range point")
        local=inv[gf]
        if np.any(local<0):raise ValueError(role+" source face point is absent from the "+volume_role+" volume")
        per_role[role]=local
        for f in local:
            key=tuple(sorted(map(int,f)))
            prior=lookup.get(key)
            if prior is not None and prior!=role:raise ValueError("one volume face maps to two distinct source roles")
            lookup[key]=role
    return lookup,per_role

def _volume_role_audit(volume_role,volume,surfaces,smap,vmap,fixed_global):
    p=np.asarray(volume["points_world_m_f32"],np.float32)
    t=np.asarray(volume["tetrahedra"],np.int64)
    det,nd=OWNER_MOD.R.M.M.measurements(p,t)
    passed=np.asarray(OWNER_MOD.R.M.M.passes(det,nd),bool)
    if np.any(~np.isfinite(det)) or np.any(~np.isfinite(nd)) or np.any(det<=0):
        raise ValueError(volume_role+" baseline has nonpositive/nonfinite cells")
    source_lookup,source_faces=_surface_face_roles(volume_role,surfaces,smap,vmap)
    raw_faces=t[:,PAT].reshape(-1,3)
    owner_cells=np.repeat(np.arange(len(t),dtype=np.int64),4)
    sorted_faces=np.sort(raw_faces,axis=1)
    unique_faces,inverse,counts=np.unique(sorted_faces,axis=0,return_inverse=True,return_counts=True)
    boundary_mask=counts[inverse]==1
    bfaces=raw_faces[boundary_mask]
    bcells=owner_cells[boundary_mask]
    bkeys=[tuple(map(int,row)) for row in np.sort(bfaces,axis=1)]
    bkeyset=set(bkeys)
    missing={role:sum(tuple(sorted(map(int,f))) not in bkeyset for f in fs) for role,fs in source_faces.items()}
    if any(missing.values()):raise ValueError(volume_role+" source facets are not all exact volume boundary triangles: "+repr(missing))
    global_face_rows=np.asarray(vmap[volume_role],np.int64)[bfaces]
    is_fixed=np.fromiter((all(int(x) in fixed_global for x in row) for row in global_face_rows),dtype=bool,count=len(bfaces))
    xyz=np.asarray(p[bfaces],np.float64)
    e01=xyz[:,1]-xyz[:,0];e02=xyz[:,2]-xyz[:,0];e12=xyz[:,2]-xyz[:,1]
    areas=.5*np.linalg.norm(np.cross(e01,e02),axis=1)
    lengths=np.maximum.reduce([np.linalg.norm(e01,axis=1),np.linalg.norm(e02,axis=1),np.linalg.norm(e12,axis=1)])
    upper=np.divide(2.*areas,lengths*lengths,out=np.zeros_like(areas),where=lengths>0)
    del xyz,e01,e02,e12
    source_roles=[source_lookup.get(k,"wall_or_unmapped_boundary") for k in bkeys]
    role_summary={}
    for role in ("O","I","D","wall_or_unmapped_boundary"):
        mask=np.fromiter((x==role for x in source_roles),dtype=bool,count=len(source_roles))
        if np.any(mask):
            role_summary[role]={"facets":int(mask.sum()),"fixed_coordinate_facets":int(np.count_nonzero(mask&is_fixed)),
                "native_q_impossible_if_fixed":int(np.count_nonzero(mask&is_fixed&(upper<Q_TARGET))),
                "all_facets_with_fixed_shape_upperbound_below_q":int(np.count_nonzero(mask&(upper<Q_TARGET))),
                "minimum_q_upperbound":float(upper[mask].min()),"median_q_upperbound":float(np.median(upper[mask]))}
    bad=np.flatnonzero(~passed)
    bycell={}
    for fi,cell in enumerate(bcells):bycell.setdefault(int(cell),[]).append(fi)
    cells=[]
    for cell in bad:
        facets=[]
        for fi in bycell.get(int(cell),[]):
            qbound=float(upper[fi]);area=float(areas[fi]);L=float(lengths[fi]);fixed=bool(is_fixed[fi]);role=source_roles[fi]
            facets.append({"face_local_point_rows":list(map(int,bfaces[fi])),"source_role":role,
                "all_three_vertex_coordinates_fixed":fixed,"area_m2":area,"longest_face_edge_m":L,
                "required_area_m2_for_q_threshold":float(Q_TARGET*L*L/2.),
                "fixed_face_native_q_upper_bound":qbound,
                "q_bound_to_threshold_ratio":qbound/Q_TARGET if Q_TARGET else None,
                "impossible_to_pass_native_q_if_this_facet_shape_and_boundary_status_are_preserved":bool(fixed and qbound<Q_TARGET)})
        xyz=p[t[int(cell)]].astype(np.float64)
        longest=max(float(np.linalg.norm(xyz[a]-xyz[b])) for a in range(4) for b in range(a+1,4))
        cells.append({"cell_row":int(cell),"oriented_det6_m3":float(det[cell]),"normalized_det":float(nd[cell]),
            "largest_edge_m":longest,
            "boundary_facets":facets,
            "has_fixed_facet_upperbound_below_native_threshold":any(x["impossible_to_pass_native_q_if_this_facet_shape_and_boundary_status_are_preserved"] for x in facets)})
    # Keep per-bad-cell detail compact; current failure census is only O(10^2), not the full tet domain.
    return {"material":volume_role,"tetrahedra":int(len(t)),"native_bad_cells":int(len(bad)),
        "positive_baseline_cells":True,"source_boundary_face_counts":{r:int(len(f)) for r,f in source_faces.items()},
        "source_faces_missing_from_exact_volume_boundary":missing,
        "total_volume_boundary_facets":int(len(bfaces)),"role_boundary_summary":role_summary,
        "bad_cells_with_any_volume_boundary_facet":int(sum(bool(bycell.get(int(c))) for c in bad)),
        "bad_cells_with_coordinate_fixed_facet":int(sum(any(is_fixed[fi] for fi in bycell.get(int(c),[])) for c in bad)),
        "bad_cells_proven_impossible_by_fixed_facet_shape":int(sum(any(is_fixed[fi] and upper[fi]<Q_TARGET for fi in bycell.get(int(c),[])) for c in bad)),
        "bad_cells":cells}

def audit_state(state_path,output_path):
    global OWNER_MOD
    OWNER_MOD=load_owner()
    parent=OWNER_MOD.state(Path(state_path).resolve())
    surfaces,volumes,points,tets,vmap,smap=OWNER_MOD.R.build_parent_domain(parent)
    back=OWNER_MOD.R.independent_back_lineage(parent,surfaces,volumes)
    fixed,anchor_counts=OWNER_MOD.R.source_fixed_rows(parent,surfaces,smap,back["fat_rows"],vmap)
    fixed=set(map(int,fixed))
    # O is fixed throughout this target policy; actual current I/Fat back is also immutable.
    o_active=np.unique(np.asarray(surfaces["O"]["faces"],np.int64).reshape(-1))
    fixed.update(map(int,np.asarray(smap["O"],np.int64)[o_active]))
    fixed.update(map(int,np.asarray(smap["I"],np.int64)[np.asarray(back["current_i_rows"],np.int64)]))
    fixed.update(map(int,np.asarray(vmap["Fat"],np.int64)[np.asarray(back["fat_rows"],np.int64)]))
    reports={r:_volume_role_audit(r,volumes[r],surfaces,smap,vmap,fixed) for r in ("Skin","Fat")}
    report={"schema":"h706.fixed-boundary-facet-q-upper-bound.v1","status":"PASS_READ_ONLY_FIXED_FACET_UPPER_BOUND_AUDIT",
        "input_state":pin(state_path),"quality_owner":{"path":str(OWNER.resolve()),"sha256":OWNER_SHA},
        "native_quality_threshold":{"normalized_det_minimum":Q_TARGET,"absolute_volume_m3_minimum":1.e-18},
        "bound_formula":"For any apex X on a tetra with fixed face ABC, det6/Ltet^3 = 2*A(ABC)*height/Ltet^3 <= 2*A(ABC)/Lface^2 because Ltet>=max(Lface,height).",
        "fixed_coordinate_policy":"all active O points, exact current I back rows, current+historical Fat back rows, and authenticated current scope/rim/route rows from pinned independent v25 owner",
        "fixed_global_point_count":len(fixed),"protected_anchor_counts":anchor_counts,
        "material_audits":reports,"physical_admission":False,"native_steps":0,
        "interpretation":"A bad cell with any coordinate-fixed boundary facet whose upper bound is below 1e-5 cannot pass native quality while that facet is preserved. A bound above threshold is not proof that an admissible apex or topology exists.",
        "mass_reconciliation":"PENDING"}
    out=Path(output_path).resolve()
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"input_state":report["input_state"],
        "native_bad_cells_by_role":{r:reports[r]["native_bad_cells"] for r in reports},
        "fixed_facet_proven_impossible_by_role":{r:reports[r]["bad_cells_proven_impossible_by_fixed_facet_shape"] for r in reports},
        "report":str(out)},sort_keys=True))
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--state",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();audit_state(args.state,args.output)
if __name__=="__main__":main()

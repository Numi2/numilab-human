#!/Users/numi/numi-human-prep-venv-20261005/bin/python
"""Rebase a frozen source-only held-flow ledger onto a current volume mesh.

Exact ordered/signed cell matches are attempted first. Only old cells without
such a match are centroid-point-located in the current tetrahedral mesh.
Source quadrature coordinates, masses, and parent corner moments are never
changed; rows failing containment, material, radius, or density are released.
No nearest-cell fallback or physical/native admission is performed.
"""
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
H=ROOT/"fine-physical-volume-706"
S=ROOT/"fractional-semantic-support-090-sparse-graph"
B=S/"fat-held-exchange-031"
DEFAULT_BASE=H/"combined-child-handoff-142"
DEFAULT_MER=B/"run-061-global-held-merge-top13-060/attempt-004/merged-held-state-060.npz"
DEFAULT_STATE=B/"run-073-release-next8-from-round6-attempt014/phase1-master-state.npz"
DEFAULT_PCSV=S/"fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv"
DEFAULT_QUAD=S/"fat-853406-run-003-root/source-quadrature-162-root/source-quadrature.npz"
DEFAULT_SUPPORT=H/"connected-region-mass-support-143-root/fat-connected-region-support.npz"
DEFAULT_SN=ROOT/"native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv.nodes.csv"
DEFAULT_TET=ROOT/"native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv.tets.csv"
RADIUS=0.019353
LOW_DENSITY=850.0
HIGH_DENSITY=950.0

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def read_mesh(node_path,ele_path):
    n=np.loadtxt(node_path,skiprows=1,usecols=(0,1,2,3),ndmin=2,dtype=np.float64)
    node_ids=n[:,0].astype(np.int64)
    points=np.asarray(n[:,1:4],dtype=np.float32)
    order=np.argsort(node_ids,kind="stable")
    if len(np.unique(node_ids))!=len(node_ids):
        raise RuntimeError("duplicate TetGen node IDs")
    e=np.loadtxt(ele_path,skiprows=1,ndmin=2,dtype=np.float64)
    elem_ids=e[:,0].astype(np.int64)
    conn_ids=e[:,1:5].astype(np.int64)
    sort_ids=node_ids[order]
    ix=np.searchsorted(sort_ids,conn_ids)
    if np.any(ix>=len(sort_ids)) or not np.array_equal(sort_ids[ix],conn_ids):
        raise RuntimeError("TetGen element point ID is missing")
    tets=order[ix].astype(np.int64)
    material=e[:,5].astype(np.int32) if e.shape[1]>5 else np.zeros(len(e),np.int32)
    return points,tets,material,elem_ids

def centroids(points,tets):
    c=np.zeros((len(tets),3),dtype=np.float64)
    # Match the native four-corner accumulation order.
    for k in range(4):
        c+=points[tets[:,k]].astype(np.float64)*0.25
    return c

def signed_det(points,tets):
    v=points[tets].astype(np.float64)
    return np.einsum("ij,ij->i",np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]),v[:,3]-v[:,0])

def exact_key_maps(old_points,old_tets,old_mat,new_points,new_tets,new_mat):
    old_det=signed_det(old_points,old_tets)
    new_det=signed_det(new_points,new_tets)
    old_sign=np.sign(old_det).astype(np.int8)
    new_sign=np.sign(new_det).astype(np.int8)
    if np.any(old_sign==0) or np.any(new_sign==0):
        raise RuntimeError("zero-volume cell prevents exact signed keying")
    joined=np.concatenate((old_points.astype("<f4"),new_points.astype("<f4")),axis=0)
    coord_bits=np.ascontiguousarray(joined).view("<u4").reshape((-1,3))
    # Compact exact Float32 coordinate identities with a stable structured-key
    # sort. This avoids np.unique(axis=0)'s multi-array temporary while keeping
    # bitwise coordinate equality and deterministic row IDs.
    coord_dtype=np.dtype([("x","<u4"),("y","<u4"),("z","<u4")])
    coord_records=np.empty(len(coord_bits),dtype=coord_dtype)
    coord_records["x"]=coord_bits[:,0]
    coord_records["y"]=coord_bits[:,1]
    coord_records["z"]=coord_bits[:,2]
    coord_order=np.argsort(coord_records,kind="stable",order=("x","y","z"))
    first=np.ones(len(coord_order),dtype=bool)
    if len(first)>1:
        first[1:]=coord_records[coord_order[1:]]!=coord_records[coord_order[:-1]]
    coord_ids=np.empty(len(coord_order),dtype=np.int64)
    coord_ids[coord_order]=np.cumsum(first,dtype=np.int64)-1
    del joined,coord_bits,coord_records,coord_order,first
    old_ids=coord_ids[:len(old_points)]
    new_ids=coord_ids[len(old_points):]
    dtype=np.dtype([(f"v{i}","<u4") for i in range(4)]+[("mat","<i4"),("sign","i1")])
    fields=[f"v{i}" for i in range(4)]+["mat","sign"]
    def pack(vertices,materials,signs):
        keys=np.empty(len(vertices),dtype=dtype)
        for j in range(4):keys[f"v{j}"]=vertices[:,j]
        keys["mat"]=materials
        keys["sign"]=signs
        return keys
    def match(old_keys,new_keys):
        order=np.argsort(new_keys,kind="stable",order=fields)
        if len(order)>1 and np.any(new_keys[order[1:]]==new_keys[order[:-1]]):
            raise RuntimeError("duplicate signed cell key within current mesh")
        pos=np.searchsorted(new_keys,old_keys,sorter=order)
        ok=pos<len(order)
        out=np.full(len(old_keys),-1,dtype=np.int64)
        rows=np.flatnonzero(ok)
        if len(rows):
            candidate=order[pos[rows]]
            equal=new_keys[candidate]==old_keys[rows]
            out[rows[equal]]=candidate[equal]
        return out
    old_order=pack(old_ids[old_tets],old_mat,old_sign)
    new_order=pack(new_ids[new_tets],new_mat,new_sign)
    ordered=match(old_order,new_order)
    del old_order,new_order
    old_sorted=np.sort(old_ids[old_tets],axis=1)
    new_sorted=np.sort(new_ids[new_tets],axis=1)
    old_canon=pack(old_sorted,old_mat,old_sign)
    new_canon=pack(new_sorted,new_mat,new_sign)
    canonical=match(old_canon,new_canon)
    del old_canon,new_canon,old_sorted,new_sorted
    result=ordered.copy()
    reordered=(result<0)&(canonical>=0)
    result[reordered]=canonical[reordered]
    kinds=np.full(len(old_tets),3,dtype=np.uint8)
    kinds[ordered>=0]=0
    kinds[reordered]=1
    return result,kinds,old_sign,new_sign


def det3(a,b,c):
    return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])

def exact_inside(point,verts):
    # Exact dyadic predicates for Float32 mesh vertices and the Float64
    # average of four Float32 old-cell corners.
    from fractions import Fraction
    f=lambda x:Fraction.from_float(float(x))
    p=[f(x) for x in point]
    v=[[f(x) for x in row] for row in verts]
    sub=lambda x,y:[x[i]-y[i] for i in range(3)]
    den=det3(sub(v[1],v[0]),sub(v[2],v[0]),sub(v[3],v[0]))
    if den==0:return False
    nums=[
        det3(sub(v[1],p),sub(v[2],p),sub(v[3],p)),
        det3(sub(p,v[0]),sub(v[2],v[0]),sub(v[3],v[0])),
        det3(sub(v[1],v[0]),sub(p,v[0]),sub(v[3],v[0])),
        det3(sub(v[1],v[0]),sub(v[2],v[0]),sub(p,v[0])),
    ]
    q=[x/den for x in nums]
    return all(x>=0 and x<=1 for x in q) and sum(q)==1

def point_locate_changed(old_centers,changed_rows,old_mat,new_points,new_tets,new_mat,new_centers,new_volumes):
    out=np.full(len(old_centers),-1,dtype=np.int64)
    ambiguous=np.zeros(len(old_centers),dtype=np.uint8)
    wrong_material=np.zeros(len(old_centers),dtype=np.uint8)
    if not len(changed_rows):
        return out,ambiguous,wrong_material,{"query_count":0,"candidate_pair_count_before_sphere_filter":0,"candidate_pair_count_after_sphere_filter":0,"exact_boundary_predicate_count":0,"radius_bin_count":0,"query_tile_size":256,"max_cell_circumradius_m":0.0}
    verts=new_points[new_tets].astype(np.float64)
    radii=np.linalg.norm(verts-new_centers[:,None,:],axis=2).max(axis=1)
    if np.any(radii<=0):raise RuntimeError("current tetrahedron has zero centroid radius")
    edge=np.stack((verts[:,1]-verts[:,0],verts[:,2]-verts[:,0],verts[:,3]-verts[:,0]),axis=2)
    inv=np.linalg.inv(edge)
    cond=np.linalg.norm(edge,ord=np.inf,axis=(1,2))*np.linalg.norm(inv,ord=np.inf,axis=(1,2))
    # Bound fast floating barycentric filters by a conservative condition estimate.
    bary_tol=np.minimum(1e-4,np.maximum(1e-12,128.*np.finfo(np.float64).eps*cond))
    bins=np.floor(np.log2(radii)).astype(np.int32)
    total_before=0;total_after=0;exact_checks=0;tile=256
    for radius_bin in np.unique(bins):
        cells=np.flatnonzero(bins==radius_bin)
        tree=cKDTree(new_centers[cells])
        query_r=np.nextafter(float(np.max(radii[cells]))+1e-12,np.inf)
        for start in range(0,len(changed_rows),tile):
            rows=changed_rows[start:start+tile]
            hits=tree.query_ball_point(old_centers[rows],query_r,workers=1)
            for row,local in zip(rows,hits):
                if not local:continue
                local=np.asarray(local,dtype=np.int64)
                cand=cells[local]
                total_before+=len(cand)
                center_delta=old_centers[row]-new_centers[cand]
                center_distance=np.linalg.norm(center_delta,axis=1)
                sphere=center_distance<=radii[cand]+1e-12
                cand=cand[sphere]
                if not len(cand):continue
                total_after+=len(cand)
                delta=old_centers[row]-verts[cand,0]
                u=np.einsum("nij,nj->ni",inv[cand],delta)
                bary=np.column_stack((1.-u.sum(axis=1),u))
                minb=bary.min(axis=1);maxb=bary.max(axis=1);tol=bary_tol[cand]
                ill=cond[cand]>1e8
                possible=((minb>=-tol)&(maxb<=1.+tol))|ill
                exact_candidates=np.flatnonzero(possible&(ill|(minb<np.maximum(1e-9,tol))|(maxb>1.-np.maximum(1e-9,tol))))
                exact_ok={}
                for q in exact_candidates:
                    exact_checks+=1
                    exact_ok[int(q)]=exact_inside(old_centers[row],new_points[new_tets[cand[q]]])
                inside=np.zeros(len(cand),dtype=bool)
                easy=possible&~ill&(minb>=np.maximum(1e-9,tol))&(maxb<=1.-np.maximum(1e-9,tol))
                inside[easy]=True
                for q,ok in exact_ok.items():inside[q]=ok
                if not np.any(inside):continue
                contained=cand[inside]
                same=contained[new_mat[contained]==old_mat[row]]
                if not len(same):
                    wrong_material[row]=1
                    continue
                if len(same)>1:ambiguous[row]=1
                order=np.lexsort((same,-new_volumes[same]))
                out[row]=int(same[order[0]])
    return out,ambiguous,wrong_material,{"query_count":int(len(changed_rows)),"candidate_pair_count_before_sphere_filter":int(total_before),"candidate_pair_count_after_sphere_filter":int(total_after),"exact_boundary_predicate_count":int(exact_checks),"radius_bin_count":int(len(np.unique(bins))),"query_tile_size":tile,"max_cell_circumradius_m":float(np.max(radii,initial=0.0))}


def load_candidate(path):
    with np.load(path,allow_pickle=False) as z:
        p=np.asarray(z["points_world_m"],dtype=np.float32).copy()
        t=np.asarray(z["tetrahedra"],dtype=np.int64).copy()
        m=np.asarray(z["material_indices"],dtype=np.int32).copy()
    if p.ndim!=2 or p.shape[1]!=3 or t.ndim!=2 or t.shape[1]!=4 or len(m)!=len(t):
        raise RuntimeError("candidate volume arrays malformed")
    if np.any(t<0) or np.any(t>=len(p)) or not np.isfinite(p).all():
        raise RuntimeError("candidate volume coordinates/connectivity invalid")
    return p,t,m

def read_csv(path):
    with Path(path).open(newline="") as f:return list(csv.DictReader(f))

def source_parent_geometry(parent_csv,nodes_csv,tets_csv):
    pars=read_csv(parent_csv);pnodes=read_csv(nodes_csv);ptets=read_csv(tets_csv)
    global_rows=np.asarray([int(r["global_source330_row"]) for r in pars],dtype=np.int64)
    mass=np.asarray([float(r["mass_kg"]) for r in pars],dtype=np.float64)
    region=np.asarray([int(r["region"]) for r in pars],dtype=np.int32)
    material=np.asarray([int(r["material"]) for r in pars],dtype=np.int32)
    node_ids=np.asarray([int(r["parent_node_index"]) for r in pnodes],dtype=np.int64)
    xyz=np.asarray([[float(r[f"rest_{a}_m"]) for a in "xyz"] for r in pnodes],dtype=np.float64)
    tet_rows=np.asarray([int(r["parent_tet_row"]) for r in ptets],dtype=np.int64)
    tet_ids=np.asarray([[int(r[f"parent_node{k}"]) for k in range(4)] for r in ptets],dtype=np.int64)
    no=np.argsort(node_ids,kind="stable");ns=node_ids[no];ix=np.searchsorted(ns,tet_ids)
    if np.any(ix>=len(ns)) or not np.array_equal(ns[ix],tet_ids):raise RuntimeError("source330 tet/node identity map failed")
    tet_nodes=no[ix]
    to=np.argsort(tet_rows,kind="stable");ts=tet_rows[to];pi=np.searchsorted(ts,global_rows)
    if np.any(pi>=len(ts)) or not np.array_equal(ts[pi],global_rows):raise RuntimeError("source330 role/global row map failed")
    v=xyz[tet_nodes[to[pi]]]
    if v.shape!=(len(pars),4,3) or np.any(mass<=0) or not np.isfinite(v).all():raise RuntimeError("source330 parent geometry/mass invalid")
    return global_rows,mass,region,material,v

def build_flows(merge_path,state_path,parent_csv,nodes_csv,tets_csv,quad_path):
    g,mass,region,material,V=source_parent_geometry(parent_csv,nodes_csv,tets_csv)
    with np.load(merge_path,allow_pickle=False) as z:
        fc=z["flow_child_row"].astype(np.int32);fp=z["flow_parent_role_row"].astype(np.int32);fm=z["flow_mass_kg"].astype(np.float64)
        kind=z["flow_origin_kind"].astype(np.uint8);site=z["flow_proposal_site_row"].astype(np.int32);car=z["flow_prior_carrier_row"].astype(np.int64)
        propRows=z["proposal_site_rows"].astype(np.int64);propP=z["proposal_site_parent_rows"].astype(np.int32);propC=z["proposal_site_child_rows"].astype(np.int32);propB=z["proposal_site_barycentric"].astype(np.float64)
        oldChildMass=z["child_mass_kg"].astype(np.float64);oldChildVol=z["child_reference_volume_m3"].astype(np.float64)
        oldCorner=z["parent_corner_mass_held_kg"].astype(np.float64)
    with np.load(quad_path,allow_pickle=False) as z:
        qo=z["parent_offsets"].astype(np.int64);qb=z["carrier_barycentric"].astype(np.float64);qchild=z["carrier_target_child_rows"].astype(np.int32);qglobal=z["global_source330_tet_rows"].astype(np.int64)
    if not np.array_equal(qglobal,g) or len(qo)!=len(g)+1:raise RuntimeError("FSSCARR01 parent/global map mismatch")
    if qo[0]!=0 or qo[-1]!=len(qb) or np.any(np.diff(qo)<0):raise RuntimeError("FSSCARR01 offsets are malformed")
    if np.any(fp<0)|np.any(fp>=len(g)) or np.any(fc<0)|np.any(fc>=len(oldChildMass)):raise RuntimeError("merged held-flow row domain invalid")
    if np.any(~np.isfinite(fm)) or np.any(fm<0):raise RuntimeError("merged held-flow mass invalid")
    bary=np.empty((len(fc),4),dtype=np.float64)
    k0=kind==0;k1=kind==1
    if np.any(~(k0|k1)):raise RuntimeError("unknown merged held-flow origin kind")
    sentinel=k0&(car==4294967295);bary[sentinel]=.25
    qrows=np.flatnonzero(k0&~sentinel)
    if len(qrows):
        qr=car[qrows]
        if np.any(qr<0)|np.any(qr>=len(qb)):raise RuntimeError("merged held-flow carrier row is out of range")
        if np.any(qr<qo[fp[qrows]]) or np.any(qr>=qo[fp[qrows]+1]):raise RuntimeError("carrier row belongs to another source parent")
        # carrier_target_child_rows is a witnessed construction recipient, not
        # an exclusive flow destination. The source carrier may flow to any
        # eligible child under the shared-capacity radius/semantic checks below.
        carrier_witness_receiver_mismatch_count=int(np.count_nonzero(qchild[qr]!=fc[qrows]))
        bary[qrows]=qb[qr]
    else:
        carrier_witness_receiver_mismatch_count=0
    prows=np.flatnonzero(k1);site_ids=site[prows]
    if len(prows):
        si=np.searchsorted(propRows,site_ids)
        if np.any(si>=len(propRows)) or not np.array_equal(propRows[si],site_ids) or np.any(propP[si]!=fp[prows]) or np.any(propC[si]!=fc[prows]):raise RuntimeError("merged held-flow proposal crosswalk invalid")
        bary[prows]=propB[si]
    with np.load(state_path,allow_pickle=False) as z:
        active=z["parent_role_rows"].astype(np.int32);stateChildMass=z["candidate_global_child_mass_kg"].astype(np.float64).copy();stateFixed=z["fixed_child_mass_kg"].astype(np.float64).copy()
        sp=z["site_parent_role_rows"].astype(np.int32);sc=z["site_child_rows"].astype(np.int32);sb=z["site_barycentric"].astype(np.float64).copy();sx=z["site_original_parent_share"].astype(np.float64).copy()
        if not np.array_equal(z["parent_global_source330_rows"],g[active]):raise RuntimeError("master-state active global parent map mismatch")
    # Reconstruct the state fixed ledger using the producer's exact subtraction order.
    activeFlow=np.isin(fp,active)
    replay=oldChildMass.copy();np.subtract.at(replay,fc[activeFlow],fm[activeFlow])
    fixed_delta=float(np.max(np.abs(replay-stateFixed),initial=0.0))
    if fixed_delta!=0.0:raise RuntimeError(f"master fixed ledger differs from merged-flow replay: {fixed_delta}")
    stateMass=stateFixed.copy();np.add.at(stateMass,sc,mass[sp]*sx)
    state_delta=float(np.max(np.abs(stateMass-stateChildMass),initial=0.0))
    if state_delta>2e-13*max(float(np.max(np.abs(stateChildMass),initial=0.0)),1e-300):
        raise RuntimeError(f"master candidate mass does not replay from fixed + active sites: {state_delta}")
    keepFixed=~activeFlow
    allP=np.r_[fp[keepFixed],sp[sx>0]]
    allC=np.r_[fc[keepFixed],sc[sx>0]]
    allM=np.r_[fm[keepFixed],mass[sp[sx>0]]*sx[sx>0]]
    allB=np.vstack((bary[keepFixed],sb[sx>0]))
    origin=np.r_[np.zeros(np.count_nonzero(keepFixed),dtype=np.uint8),np.ones(np.count_nonzero(sx>0),dtype=np.uint8)]
    baseline_mass=np.zeros(len(oldChildMass),dtype=np.float64);np.add.at(baseline_mass,allC,allM)
    mass_replay=float(np.max(np.abs(baseline_mass-stateChildMass),initial=0.0))
    if mass_replay>2e-13*max(float(np.max(np.abs(stateChildMass),initial=0.0)),1e-300):
        raise RuntimeError(f"combined held + active master site ledger mismatch: {mass_replay}")
    world=np.einsum("ni,nij->nj",allB,V[allP],optimize=True)
    if np.any(allB < -2e-12) or np.any(np.abs(allB.sum(axis=1)-1)>2e-12):raise RuntimeError("source quadrature barycentric invalid")
    if np.any(mass[sp]*sx < -1e-14*np.maximum(mass[sp],1e-300)) or np.any(sx>1.+1e-10):
        raise RuntimeError("active master has invalid source-parent shares")
    activeCorner=np.zeros((len(mass),4));np.add.at(activeCorner,(sp[:,None],np.arange(4)[None,:]),(mass[sp]*sx)[:,None]*sb)
    activeTarget=mass[active]/4.
    activeRel=np.max(np.abs(activeCorner[active] - activeTarget[:,None])/np.maximum(activeTarget[:,None],1e-300),initial=0.)
    if activeRel>1e-8:raise RuntimeError(f"active master source M/4 ledger fails replay: {activeRel}")
    mergedCorner=np.zeros((len(mass),4));np.add.at(mergedCorner,(fp[:,None],np.arange(4)[None,:]),fm[:,None]*bary)
    mergedRel=np.max(np.abs(mergedCorner-oldCorner)/np.maximum(mass[:,None],1e-300),initial=0.)
    if mergedRel>1e-10:raise RuntimeError(f"merged held parent-corner ledger mismatch: {mergedRel}")
    mergedChild=np.zeros(len(oldChildMass));np.add.at(mergedChild,fc,fm)
    mergedChildDelta=float(np.max(np.abs(mergedChild-oldChildMass),initial=0.))
    if mergedChildDelta>2e-13*max(float(np.max(np.abs(oldChildMass),initial=0.)),1e-300):
        raise RuntimeError(f"merged held child ledger mismatch: {mergedChildDelta}")
    if len(allC) and np.any(material[allP]<0):raise RuntimeError("invalid source parent material")
    return {"parent_global_rows":g,"parent_mass_kg":mass,"parent_region":region,"parent_material":material,"parent_tet_vertices":V,
        "flow_parent":allP,"flow_child_old":allC,"flow_mass_kg":allM,"flow_barycentric":allB,"flow_source_point_world_m":world,
        "flow_origin_kind":origin,"old_state_child_mass_kg":stateChildMass,"old_child_volume_m3":oldChildVol,
        "old_parent_corner_mass_kg":oldCorner,"merged_corner_replay_relative_error":float(mergedRel),"active_M4_replay_relative_error":float(activeRel),
        "merged_child_ledger_max_abs_kg":mergedChildDelta,"fixed_replay_max_abs_kg":fixed_delta,"state_replay_max_abs_kg":state_delta,
        "combined_replay_max_abs_kg":mass_replay,"old_child_mass_from_site_rows":baseline_mass,"source_global_rows":g,
        "carrier_witness_receiver_mismatch_count":carrier_witness_receiver_mismatch_count}

def sorted_target_flow_groups(valid,target,mass):
    ids=np.flatnonzero(valid)
    order=np.lexsort((ids,mass[ids],target[ids]))
    return ids[order],target[ids[order]]

def rebase_flows(flows,old_mat,current_map,mapping_kind,current_points,current_tets,current_mat,current_centers,current_volumes,support_path,region_proven=None,current_region_support_path=None):
    p=flows["flow_parent"];oldc=flows["flow_child_old"];m=flows["flow_mass_kg"];b=flows["flow_barycentric"];source=flows["flow_source_point_world_m"]
    if support_path is not None:
        with np.load(support_path,allow_pickle=False) as z:
            off=z["cell_admissible_region_offsets"].astype(np.int64);reg=z["cell_admissible_region_ids"].astype(np.int32)
        if len(off)!=len(old_mat)+1 or off[0]!=0 or off[-1]!=len(reg):raise RuntimeError("old 143 support domain mismatch")
        child=np.repeat(np.arange(len(old_mat),dtype=np.uint64),np.diff(off))
        allowed=child*(1<<32)+reg.astype(np.uint64)
        if not len(allowed):raise RuntimeError("old 143 support is empty")
        if np.any(allowed[1:]<=allowed[:-1]):raise RuntimeError("old 143 region CSR is not sorted/unique")
        key=oldc.astype(np.uint64)*(1<<32)+flows["parent_region"][p].astype(np.uint64)
        q=np.searchsorted(allowed,key);ok=(q<len(allowed))&(allowed[np.minimum(q,max(len(allowed)-1,0))]==key)
        if not np.all(ok):raise RuntimeError("frozen source ledger contains an unauthenticated old child/region assignment")
    target=current_map[oldc]
    reason=np.zeros(len(m),dtype=np.uint8)
    reason[target<0]=1
    reason[(target<0)&(mapping_kind[oldc]==4)]=2
    valid=target>=0
    current_region_ok=np.zeros(len(m),dtype=bool)
    if current_region_support_path is not None:
        with np.load(current_region_support_path,allow_pickle=False) as z:
            current_off=z["cell_admissible_region_offsets"].astype(np.int64)
            current_reg=z["cell_admissible_region_ids"].astype(np.int32)
        if len(current_off)!=len(current_tets)+1 or current_off[0]!=0 or current_off[-1]!=len(current_reg) or np.any(np.diff(current_off)<=0):
            raise RuntimeError("current 143 region support domain/offsets are malformed")
        current_child=np.repeat(np.arange(len(current_tets),dtype=np.uint64),np.diff(current_off))
        current_allowed=current_child*(1<<32)+current_reg.astype(np.uint64)
        if not len(current_allowed) or np.any(current_allowed[1:]<=current_allowed[:-1]):
            raise RuntimeError("current 143 region support is empty or not sorted/unique")
        eligible=np.flatnonzero(valid)
        keys=target[eligible].astype(np.uint64)*(1<<32)+flows["parent_region"][p[eligible]].astype(np.uint64)
        at=np.searchsorted(current_allowed,keys)
        inrange=at<len(current_allowed)
        hit=np.zeros(len(eligible),dtype=bool)
        if np.any(inrange):
            hit[inrange]=current_allowed[at[inrange]]==keys[inrange]
        current_region_ok[eligible]=hit
        reason[eligible[~hit]]=5
        valid[eligible[~hit]]=False
    if region_proven is not None:
        rp=np.asarray(region_proven,dtype=bool)
        if rp.shape!=(len(old_mat),):
            raise RuntimeError("canonical current-region proof must cover the full old child row domain")
        proven=rp[oldc]
        reason[valid&~proven]=5
        valid &= proven
    sameMat=np.zeros(len(m),dtype=bool);sameMat[valid]=current_mat[target[valid]]==flows["parent_material"][p[valid]]
    reason[valid&~sameMat]=2
    valid &= sameMat
    # The source site and mass are unchanged. Recompute only its recipient-centroid
    # distance on the current mesh under the original hard radius.
    dist=np.full(len(m),np.inf)
    dist[valid]=np.linalg.norm(source[valid]-current_centers[target[valid]],axis=1)
    radiusOK=valid&(dist<=RADIUS)
    reason[valid&~radiusOK]=3
    valid=radiusOK
    massBy=np.zeros(len(current_tets),dtype=np.float64);np.add.at(massBy,target[valid],m[valid])
    upper=HIGH_DENSITY*current_volumes
    over=np.flatnonzero(massBy>upper)
    densityReleased=0;densityReleaseMass=0.0
    # Build exact target groups once, with the same (mass,row) ordering as the
    # former per-overfull-cell full-flow scan.
    ordered_flow_ids,ordered_flow_targets=sorted_target_flow_groups(valid,target,m)
    # Deterministically release the smallest flow rows needed to get each target
    # under its current upper-density bound. No flow is reassigned or clipped.
    for c in over:
        lo=np.searchsorted(ordered_flow_targets,c,"left")
        hi=np.searchsorted(ordered_flow_targets,c,"right")
        ids=ordered_flow_ids[lo:hi]
        excess=float(massBy[c]-upper[c]);taken=0.0
        for i in ids:
            valid[i]=False;reason[i]=4;taken+=float(m[i]);densityReleased+=1
            if taken>=excess and massBy[c]-taken<=upper[c]:break
        densityReleaseMass+=taken
        massBy[c]-=taken
        if massBy[c]>upper[c]:
            for i in ids:
                if reason[i]==4:continue
                valid[i]=False;reason[i]=4;taken+=float(m[i]);densityReleased+=1
                massBy[c]-=float(m[i]);densityReleaseMass+=float(m[i])
                if massBy[c]<=upper[c]:break
    if np.any(massBy>upper):raise RuntimeError("density release did not close every upper bound")
    finalMass=np.zeros(len(current_tets),dtype=np.float64);np.add.at(finalMass,target[valid],m[valid])
    density=finalMass/current_volumes
    cornerBefore=np.zeros((len(flows["parent_mass_kg"]),4));cornerAfter=np.zeros_like(cornerBefore)
    np.add.at(cornerBefore,(p[:,None],np.arange(4)[None,:]),m[:,None]*b)
    np.add.at(cornerAfter,(p[valid,None],np.arange(4)[None,:]),m[valid,None]*b[valid])
    parentRetained=np.bincount(p[valid],weights=m[valid],minlength=len(flows["parent_mass_kg"]))
    parentReleased=np.bincount(p[~valid],weights=m[~valid],minlength=len(flows["parent_mass_kg"]))
    beforeDef=np.maximum(LOW_DENSITY*current_volumes-finalMass,0.)
    retainedReasons={str(k):int(np.count_nonzero(reason==k)) for k in range(6)}
    return {"target":target,"distance":dist,"retained":valid,"reason":reason,"current_mass_kg":finalMass,"current_density_kg_m3":density,
        "corner_before_kg":cornerBefore,"corner_after_kg":cornerAfter,"parent_retained_kg":parentRetained,"parent_released_kg":parentReleased,
        "current_region_permitted":current_region_ok,"low_deficit_kg":beforeDef,"density_release_count":densityReleased,"density_release_mass_kg":densityReleaseMass,
        "release_reason_counts":retainedReasons,"pre_density_mass_kg":massBy+np.bincount(target[reason==4],weights=m[reason==4],minlength=len(current_tets))}

def self_test():
    # Exact ordered key, even permutation with preserved sign, changed centroid
    # located in split cells, material mismatch, and no-containment all exercise
    # the same production helpers.
    p=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[4,0,0],[5,0,0],[4,1,0],[4,0,1]],dtype=np.float32)
    old=np.asarray([[0,1,2,3],[4,5,6,7]],dtype=np.int64);om=np.asarray([16,16],dtype=np.int32)
    # First cell exact but reordered by an even permutation. The second cell
    # remains geometrically present but changes material, so it is rejected by
    # point location rather than silently matched.
    q=p.copy()
    nt=np.asarray([[1,2,0,3],[4,5,6,7]],dtype=np.int64);nm=np.asarray([16,15],dtype=np.int32)
    exact,kind,_,_=exact_key_maps(p,old,om,q,nt,nm)
    assert exact[0]==0 and kind[0]==1 and exact[1]==-1
    c=centroids(q,nt);vol=np.abs(signed_det(q,nt))/6
    loc,amb,wrong,stats=point_locate_changed(centroids(p,old),np.asarray([1]),om,q,nt,nm,c,vol)
    assert loc[1]==-1 and wrong[1]==1
    qNo=p[:4].copy();tNo=np.asarray([[0,1,2,3]],dtype=np.int64);mNo=np.asarray([16],dtype=np.int32)
    cNo=centroids(qNo,tNo);vNo=np.abs(signed_det(qNo,tNo))/6
    locNo,ambNo,wrongNo,_=point_locate_changed(centroids(p,old),np.asarray([1]),om,qNo,tNo,mNo,cNo,vNo)
    assert locNo[1]==-1 and wrongNo[1]==0
    # Rebuild the second region as four positive subcells meeting at its old
    # centroid; containment is valid and deterministic despite boundary ties.
    q2=np.vstack((p[:4],p[4:8],p[4:8].mean(axis=0)))
    a=8
    split=np.asarray([[4,5,6,a],[4,5,a,7],[4,a,6,7],[a,5,6,7]],dtype=np.int64)
    # Orient split cells consistently positive.
    for i,t in enumerate(split):
        if signed_det(q2,np.asarray([t]))[0]<0:split[i,[0,1]]=split[i,[1,0]]
    nm2=np.full(4,16,dtype=np.int32)
    cc2=centroids(q2,split);vv2=np.abs(signed_det(q2,split))/6
    loc2,amb2,wrong2,_=point_locate_changed(centroids(p,old),np.asarray([1]),om,q2,split,nm2,cc2,vv2)
    assert loc2[1]>=0 and amb2[1]==1 and wrong2[1]==0
    # The same production ledger rebasing function releases only rows that
    # exceed current cell capacity; it never clips or reassigns their mass.
    fake={"flow_parent":np.asarray([0,0],np.int32),"flow_child_old":np.asarray([0,1],np.int32),
      "flow_mass_kg":np.asarray([.6,.6]),"flow_barycentric":np.full((2,4),.25),
      "flow_source_point_world_m":np.zeros((2,3)),"parent_region":np.asarray([5],np.int32),
      "parent_material":np.asarray([16],np.int32),"parent_mass_kg":np.asarray([2.4])}
    rr=rebase_flows(fake,np.asarray([16,16],np.int32),np.asarray([0,0],np.int64),np.asarray([0,0],np.uint8),
      p[:4],np.asarray([[0,1,2,3]],np.int64),np.asarray([16],np.int32),np.zeros((1,3)),np.asarray([.001]),None)
    assert rr["density_release_count"]==1 and np.isclose(rr["current_mass_kg"][0],.6)
    assert np.count_nonzero(rr["retained"])==1 and np.count_nonzero(rr["reason"]==4)==1
    rrreg=rebase_flows(fake,np.asarray([16,16],np.int32),np.asarray([0,0],np.int64),np.asarray([0,0],np.uint8),
      p[:4],np.asarray([[0,1,2,3]],np.int64),np.asarray([16],np.int32),np.zeros((1,3)),np.asarray([.01]),None,
      region_proven=np.asarray([False,True]))
    assert rrreg["reason"][0]==5 and rrreg["retained"][1] and np.isclose(rrreg["current_mass_kg"][0],.6)
    dv=np.asarray([.2,.1,.1,.1,.05]);dt=np.asarray([1,0,1,0,2]);valid=np.ones(5,dtype=bool)
    ordids,ordtargets=sorted_target_flow_groups(valid,dt,dv)
    for cell in range(3):
      got=ordids[ordtargets==cell]
      ref=np.flatnonzero(dt==cell);ref=ref[np.lexsort((ref,dv[ref]))]
      assert np.array_equal(got,ref)
    print(json.dumps({"status":"PASS_REBASE_HELPER_CONTROLS","ordered_signed_reordered_key":True,"changed_cell_point_location":True,"material_mismatch_rejected":True,"no_containment_rejected":True,"ambiguous_boundary_containment_deterministic":True,"density_rows_released_without_clipping_or_reassignment":True,"density_release_target_group_order_matches_reference":True,"unproven_current_region_released_without_reassignment":True},indent=2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--role",choices=("Fat","Skin"),required=False,default="Fat")
    ap.add_argument("--candidate",type=Path)
    ap.add_argument("--candidate-report",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--baseline-dir",type=Path,default=DEFAULT_BASE)
    ap.add_argument("--merge",type=Path,default=DEFAULT_MER)
    ap.add_argument("--master-state",type=Path,default=DEFAULT_STATE)
    ap.add_argument("--parent-csv",type=Path,default=DEFAULT_PCSV)
    ap.add_argument("--quad",type=Path,default=DEFAULT_QUAD)
    ap.add_argument("--support",type=Path,default=DEFAULT_SUPPORT)
    ap.add_argument("--source-nodes",type=Path,default=DEFAULT_SN)
    ap.add_argument("--source-tets",type=Path,default=DEFAULT_TET)
    ap.add_argument("--geometry-only",action="store_true")
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test:
        self_test();return
    if not args.candidate or not args.output:ap.error("--candidate and --output are required")
    t0=time.monotonic();out=args.output
    out.mkdir(parents=True,exist_ok=False)
    node=args.baseline_dir/f"{args.role.lower()}.node";ele=args.baseline_dir/f"{args.role.lower()}.ele"
    oldp,oldt,oldm,oldids=read_mesh(node,ele)
    newp,newt,newm=load_candidate(args.candidate)
    if args.candidate_report:
        rd=json.loads(args.candidate_report.read_text())
        if rd.get("physical_admission") is not False or rd.get("native_steps")!=0:
            raise RuntimeError("candidate geometry report must remain source-only")
        roles=[x for x in rd.get("roles",[]) if x.get("role")==args.role]
        if len(roles)!=1 or Path(roles[0].get("output",""))!=args.candidate or roles[0].get("output_sha256")!=sha(args.candidate):
            raise RuntimeError("candidate/report role output hash binding failed")
    oldc=centroids(oldp,oldt);newc=centroids(newp,newt)
    olddet=signed_det(oldp,oldt);newdet=signed_det(newp,newt)
    oldvol=np.abs(olddet)/6.;newvol=np.abs(newdet)/6.
    if np.any(newvol<=0):raise RuntimeError("candidate has nonpositive current volume cells")
    exact,kind,oldSign,newSign=exact_key_maps(oldp,oldt,oldm,newp,newt,newm)
    changed=np.flatnonzero(exact<0)
    located,amb,wrong,locstats=point_locate_changed(oldc,changed,oldm,newp,newt,newm,newc,newvol)
    mapping=exact.copy();mapping[changed]=located[changed]
    mapkind=kind.copy();mapkind[changed[located[changed]>=0]]=2;mapkind[changed[(located[changed]<0)&(wrong[changed]>0)]]=4
    report={"schema":"numi.human.current-volume-held-ledger-rebase.v1","status":"SOURCE_ONLY_CURRENT_GEOMETRY_REBASE_CLASSIFIED","physical_admission":False,"native_steps":0,"mass_transport":"NOT_ADMITTED",
        "role":args.role,"baseline_geometry":{"node_path":str(node),"node_sha256":sha(node),"element_path":str(ele),"element_sha256":sha(ele),"point_count":int(len(oldp)),"cell_count":int(len(oldt))},
        "candidate_geometry":{"path":str(args.candidate),"sha256":sha(args.candidate),"report_path":str(args.candidate_report) if args.candidate_report else None,"report_sha256":sha(args.candidate_report) if args.candidate_report else None,"point_count":int(len(newp)),"cell_count":int(len(newt)),"material_counts":{str(int(x)):int(np.count_nonzero(newm==x)) for x in np.unique(newm)},"minimum_signed_volume_m3":float(np.min(newdet/6.)),"minimum_absolute_volume_m3":float(np.min(newvol))},
        "ordered_signed_cell_key_map":{"exact_ordered_count":int(np.count_nonzero(kind==0)),"exact_signed_permutation_count":int(np.count_nonzero(kind==1)),"exact_geometry_count":int(np.count_nonzero(exact>=0)),"changed_old_cell_count":int(len(changed)),"changed_old_cell_pointlocated_count":int(np.count_nonzero(located[changed]>=0)),"changed_old_cell_uncontained_count":int(np.count_nonzero(located[changed]<0)),"changed_centroid_multiple_containers_count":int(np.count_nonzero(amb[changed])),"changed_centroid_only_wrong_material_count":int(np.count_nonzero(wrong[changed])),"candidate_cells_without_exact_old_key_count":int(len(newt)-np.count_nonzero(np.isin(np.arange(len(newt)),exact[exact>=0]))),"old_cell_sign_counts":{str(int(x)):int(np.count_nonzero(oldSign==x)) for x in np.unique(oldSign)},"current_cell_sign_counts":{str(int(x)):int(np.count_nonzero(newSign==x)) for x in np.unique(newSign)},"point_location":locstats},
        "rebase_policy":{"exact_match_first":True,"exact_cell_key_includes_material_and_signed_orientation":True,"changed_cell_point_location_target":"old receiving-cell centroid, tested against current tetrahedra","point_location_has_nearest_fallback":False,"ambiguous_true_containers":"same-material containing cell with largest current volume, then lowest row; no nearest fallback","source_quadrature_site_points_masses_and_barycentrics_changed":False,"failed_rows_released_without_reassignment":True,"source_mass_quarter_moments_density_bounds_and_radius_changed":False,"canonical_current_region_admission":"inherited only for exact old-cell identity; point-located changed cells without current compiler lineage are released as region-unproven"},
        "input_pins":{},"output_hashes":{},"elapsed_seconds":None}
    mapnpz=out/"cell-row-rebase.npz"
    np.savez_compressed(mapnpz,baseline_cell_row_to_current_cell_row=mapping.astype(np.int64),baseline_cell_row_to_current_exact_row=exact.astype(np.int64),baseline_cell_row_mapping_kind=mapkind.astype(np.uint8),changed_cell_pointlocation_ambiguous=amb.astype(np.uint8),changed_cell_wrong_material_only=wrong.astype(np.uint8),baseline_cell_signed_orientation=oldSign,current_cell_signed_orientation=newSign)
    report["input_pins"]={str(p):sha(p) for p in [node,ele,args.candidate]+([args.candidate_report] if args.candidate_report else [])}
    report["output_hashes"][mapnpz.name]={"sha256":sha(mapnpz),"size_bytes":mapnpz.stat().st_size}
    if args.role=="Fat" and not args.geometry_only:
        flows=build_flows(args.merge,args.master_state,args.parent_csv,args.source_nodes,args.source_tets,args.quad)
        if len(flows["old_child_mass_kg"])!=len(oldt):raise RuntimeError("source-only held ledger does not match baseline child row domain")
        child_volume_delta=np.abs(flows["old_child_volume_m3"]-oldvol)
        child_volume_rel=child_volume_delta/np.maximum(np.abs(oldvol),np.finfo(np.float64).tiny)
        max_old_volume_rel=float(np.max(child_volume_rel,initial=0.0))
        if max_old_volume_rel>1e-12:raise RuntimeError(f"baseline .ele vs merged-held child volume mismatch: {max_old_volume_rel}")
        baseline_distance=np.linalg.norm(flows["flow_source_point_world_m"]-oldc[flows["flow_child_old"]],axis=1)
        if np.any(baseline_distance>RADIUS):raise RuntimeError("frozen source sites fail baseline 19.353 mm child-centroid radius")
        oldMat=oldm
        newCent=centroids(newp,newt)
        # Verify old recipient material and inherited one-hop source region before
        # attempting current-cell placement.
        if np.any(oldMat[flows["flow_child_old"]]!=flows["parent_material"][flows["flow_parent"]]):
            raise RuntimeError("source-only held ledger has cross-material baseline rows")
        rebased=rebase_flows(flows,oldMat,mapping,mapkind,newp,newt,newm,newCent,newvol,args.support,region_proven=np.isin(mapkind,(0,1)))
        flownpz=out/"rebased-held-flow-ledger.npz"
        np.savez_compressed(flownpz,flow_parent_role_rows=flows["flow_parent"],flow_global_source330_rows=flows["parent_global_rows"][flows["flow_parent"]],
            flow_baseline_child_rows=flows["flow_child_old"],flow_current_child_rows=rebased["target"].astype(np.int64),
            flow_mass_kg=flows["flow_mass_kg"],flow_source_barycentric=flows["flow_barycentric"],flow_source_point_world_m=flows["flow_source_point_world_m"],
            flow_retained=rebased["retained"].astype(np.uint8),flow_release_reason=rebased["reason"],flow_current_centroid_distance_m=rebased["distance"],
            parent_source_mass_kg=flows["parent_mass_kg"],parent_corner_mass_before_kg=rebased["corner_before_kg"],parent_corner_mass_after_kg=rebased["corner_after_kg"],
            current_cell_mass_kg=rebased["current_mass_kg"],current_cell_density_kg_m3=rebased["current_density_kg_m3"])
        q=flows["parent_mass_kg"]/4.
        err=rebased["corner_after_kg"]-q[:,None]
        report["fat_ledger_rebase"]={"merge_path":str(args.merge),"merge_sha256":sha(args.merge),"master_state_path":str(args.master_state),"master_state_sha256":sha(args.master_state),
            "parent_csv_path":str(args.parent_csv),"parent_csv_sha256":sha(args.parent_csv),"source_nodes_sha256":sha(args.source_nodes),"source_tets_sha256":sha(args.source_tets),"carrier_packet_sha256":sha(args.quad),"support143_sha256":sha(args.support),
            "baseline_fixed_plus_active_master_replay_max_abs_kg":flows["combined_replay_max_abs_kg"],"state_fixed_replay_max_abs_kg":flows["fixed_replay_max_abs_kg"],"state_candidate_replay_max_abs_kg":flows["state_replay_max_abs_kg"],
            "merged_source_parent_corner_replay_relative_error":flows["merged_corner_replay_relative_error"],"active_master_M_over_4_replay_relative_error":flows["active_M4_replay_relative_error"],"merged_child_ledger_max_abs_kg":flows["merged_child_ledger_max_abs_kg"],
            "baseline_geometry_parity":{"merged_child_volume_max_relative_difference":max_old_volume_rel,"baseline_flow_site_max_distance_m":float(np.max(baseline_distance,initial=0.0)),"baseline_flow_radius_failures":int(np.count_nonzero(baseline_distance>RADIUS))},
            "parent_role_count":int(len(flows["parent_mass_kg"])),"flow_row_count":int(len(flows["flow_mass_kg"])),"baseline_candidate_flow_mass_kg":float(np.sum(flows["flow_mass_kg"])),"current_retained_flow_mass_kg":float(np.sum(flows["flow_mass_kg"][rebased["retained"]])),
            "released_flow_mass_kg":float(np.sum(flows["flow_mass_kg"][~rebased["retained"]])),"release_reason_counts":rebased["release_reason_counts"],
            "release_mass_by_reason_kg":{str(k):float(np.sum(flows["flow_mass_kg"][rebased["reason"]==k])) for k in range(5)},
            "source_parent_mass_total_kg":float(flows["parent_mass_kg"].sum()),"unallocated_source_mass_before_kg":float(flows["parent_mass_kg"].sum()-np.sum(flows["flow_mass_kg"])),
            "unallocated_source_mass_after_kg":float(flows["parent_mass_kg"].sum()-np.sum(flows["flow_mass_kg"][rebased["retained"]])),
            "source_corner_ledger":{"before_max_abs_M_over_4_delta_kg":float(np.max(np.abs(rebased["corner_before_kg"]-q[:,None]),initial=0.0)),"after_retained_max_abs_M_over_4_delta_kg":float(np.max(np.abs(err),initial=0.0)),"total_retained_corner_mass_kg":rebased["corner_after_kg"].sum(axis=0).tolist(),"total_released_corner_mass_kg":(rebased["corner_before_kg"]-rebased["corner_after_kg"]).sum(axis=0).tolist(),"parents_fully_at_M_over_4_after":int(np.count_nonzero(np.max(np.abs(err),axis=1)<=1e-12*np.maximum(flows["parent_mass_kg"]/4.,1e-300))),"per_parent_corner_mass_after_npz":"rebased-held-flow-ledger.npz"},
            "current_density":{"retained_mass_kg":float(rebased["current_mass_kg"].sum()),"under_floor_cell_count":int(np.count_nonzero(rebased["current_density_kg_m3"]<LOW_DENSITY)),"under_floor_deficit_kg":float(rebased["low_deficit_kg"].sum()),"over_ceiling_cell_count":int(np.count_nonzero(rebased["current_density_kg_m3"]>HIGH_DENSITY)),"minimum_positive_density_kg_m3":float(np.min(rebased["current_density_kg_m3"][rebased["current_mass_kg"]>0],initial=np.inf)),"maximum_density_kg_m3":float(np.max(rebased["current_density_kg_m3"],initial=0.0)),"density_release_count":rebased["density_release_count"],"density_release_mass_kg":rebased["density_release_mass_kg"]},
            "distance":{"hard_radius_m":RADIUS,"retained_max_distance_m":float(np.max(rebased["distance"][rebased["retained"]],initial=0.0)),"radius_release_count":int(rebased["release_reason_counts"]["3"]),"distance_recomputed_on_current_centroid":True},
            "semantic_support_scope":"Original source-parent region was rechecked against authenticated 143 support on the baseline recipient row. A current recipient inherits that proof only when its cell key exactly matches the baseline cell (ordered or signed permutation); centroid-located changed recipients are released with reason 5 until current compiler region lineage is supplied.",
            "scope_caveat":"The candidate is a source-only rebased diagnostic. Released mass is not redistributed; nonzero M/4 residuals and lower-density deficits remain unqualified."}
        report["input_pins"].update({str(p):sha(p) for p in [args.merge,args.master_state,args.parent_csv,args.source_nodes,args.source_tets,args.quad,args.support]})
        report["output_hashes"][flownpz.name]={"sha256":sha(flownpz),"size_bytes":flownpz.stat().st_size}
    rp=out/"report.json";report["elapsed_seconds"]=time.monotonic()-t0;rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"role":args.role,"exact_ordered":report["ordered_signed_cell_key_map"]["exact_ordered_count"],"exact_signed_reordered":report["ordered_signed_cell_key_map"]["exact_signed_permutation_count"],"changed":report["ordered_signed_cell_key_map"]["changed_old_cell_count"],"pointlocated":report["ordered_signed_cell_key_map"]["changed_old_cell_pointlocated_count"],"uncontained":report["ordered_signed_cell_key_map"]["changed_old_cell_uncontained_count"],"output":str(rp),"elapsed_s":report["elapsed_seconds"]},indent=2),flush=True)

if __name__=="__main__":main()

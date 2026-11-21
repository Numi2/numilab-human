"""Fit the registered skin/fat shell using a single-axis displacement at each node."""
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
from scipy.spatial import cKDTree

from screen import skin

FLOOR_DET = 6e-18
FLOOR_QUALITY = 1e-5
EDGES = np.array(list(combinations(range(4),2)))
A, B = EDGES.T
MM = .001

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def prepare(scene, proposal, bed_path):
    receipt=json.loads(scene.read_text())
    env=receipt["environment"]
    head="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    def pin(key):
        p=Path(env[head+key])
        assert digest(p)==env[head+key.removesuffix("_CSV")+"_SHA256"]
        return p
    source=np.loadtxt(pin("TETGEN_NODE"),skiprows=1,usecols=(0,1,2,3))
    assert np.array_equal(source[:,0],np.arange(len(source))+1)
    x=source[:,1:].astype("<f4")
    assert np.array_equal(source[:,1:],x) and np.array_equal(skin,x[:len(skin)])
    del source
    raw=np.loadtxt(pin("TETGEN_ELEMENTS"),skiprows=1,usecols=(0,1,2,3,4,5),dtype=np.int32)
    assert np.array_equal(raw[:,0],np.arange(len(raw))+1)
    expected=json.loads(proposal.with_name("proposal-receipt.json").read_text())
    assert digest(proposal)==expected["proposal_sha256"]
    with np.load(proposal,allow_pickle=False) as pack:
        keep=pack["retained_input_cell_rows"]
        parents=pack["replacement_parent_cell_rows"]
        offset=pack["replacement_parent_offsets"]
        sizes=np.diff(offset)
        assert np.all(sizes>0)
        replacement_material=raw[parents[offset[:-1]],5]
        assert np.array_equal(raw[parents,5],np.repeat(replacement_material,sizes))
        tets=np.concatenate((raw[keep,1:5]-1,pack["replacement_tetrahedra"]),dtype=np.int32,casting="unsafe")
        role=np.concatenate((raw[keep,5],replacement_material))
    anchors=set()
    with pin("DEEP_ATTACHMENT_REQUESTS_CSV").open() as f:
        for row in csv.DictReader(f):
            assert row["source_domain"]=="D37_coarse_parent"
            anchors.add(int(row["stable_source_vertex"]))
    rows=[]
    with pin("COMBINED_NODE_MAP_CSV").open() as f:
        for row in csv.DictReader(f):
            if row["role"]=="D" and row["source_domain"]=="D37_coarse_parent" and int(row["source_vertex_row"]) in anchors:
                rows.append(int(row["joint_node_id"])-1)
    assert len(rows)==len(anchors)==len(set(rows))
    bed=json.loads(bed_path.read_text())
    near_bed=(x.astype(float)@np.array(bed["normal"])-bed["offset_m"])<=.002
    near_bed[rows]=True
    return x,tets,role,near_bed

def tet_metrics(p):
    edges=p[:,B]-p[:,A]
    diameter=np.linalg.norm(edges,axis=2).max(axis=1)
    d=p[:,1:]-p[:,:1]
    det=np.einsum("ni,ni->n",d[:,0],np.cross(d[:,1],d[:,2]))
    return det,diameter,edges

def exact_summary(x,tets,role):
    bad=0
    low_shape=0
    worst_det=np.inf
    worst_shape=np.inf
    by_material={}
    for start in range(0,len(tets),32768):
        d,l,_=tet_metrics(x[tets[start:start+32768]].astype(float))
        q=d/l**3
        bad+=int((d<=FLOOR_DET).sum())
        low_shape+=int((q<FLOOR_QUALITY).sum())
        worst_det=min(worst_det,float(d.min()))
        worst_shape=min(worst_shape,float(q.min()))
        for m in (15,16):
            by_material[m]=by_material.get(m,0.)+float(d[role[start:start+32768]==m].sum()/6)
    return dict(volume_floor_failures=bad,shape_floor_failures=low_shape,
                min_det_m3=worst_det,min_shape=worst_shape,material_volume=by_material)

def rows(coeff, ids, n):
    i=np.repeat(np.arange(len(ids)),ids.shape[1])
    keep=(ids>=0)&(coeff!=0)
    return coo_matrix((coeff[keep],(i[keep.ravel()],ids[keep])),shape=(len(ids),n)).tocsr()

def scale_rows(matrix,rhs):
    fact=np.maximum(1,np.asarray(abs(matrix).max(axis=1).toarray()).ravel())
    return matrix.multiply(1/fact[:,None]).tocsr(),rhs/fact

def selected_sites(path):
    available=[d for d in json.loads(path.read_text()) if d["axis"]==0]
    assert len(available)==16
    return [dict(route=d["route"],site=d["site"],**min(d["candidate_faces"],key=lambda v:v["gap_m"]))
            for d in available]

def system(points,tets,locked,sites,radius_mm,eligible=None):
    site_xyz=np.array([q["point_m"] for q in sites])
    site_xyz[:,0]=[q['skin_intercept_m'] for q in sites]
    distance=cKDTree(site_xyz).query(points)[0]
    mask=(distance<radius_mm*MM) if eligible is None else eligible
    candidates=np.flatnonzero(~locked & mask)
    local=np.full(len(points),-1,np.int32)
    local[candidates]=np.arange(len(candidates))
    active=np.any(local[tets]>=0,axis=1)
    ids=tets[active]
    xyz=points[ids].astype(float)
    det,diameter,_=tet_metrics(xyz)
    assert np.all(det>FLOOR_DET) and np.all(det/diameter**3>=FLOOR_QUALITY)
    e=xyz[:,1:]-xyz[:,:1]
    g=np.column_stack([np.cross(e[:,1],e[:,2])[:,0],
                       np.cross(e[:,2],e[:,0])[:,0],
                       np.cross(e[:,0],e[:,1])[:,0]])*MM
    g=np.c_[-g.sum(axis=1),g]
    pair=np.unique(np.sort(ids[:,EDGES].reshape(-1,2),axis=1),axis=0)
    pair=pair[np.any(local[pair]>=0,axis=1)]
    lengths=np.linalg.norm(points[pair[:,1]].astype(float)-points[pair[:,0]].astype(float),axis=1)/MM
    edge=rows(np.broadcast_to([-1.,1.],pair.shape),local[pair],len(candidates))
    bc=rows(np.array([np.array(q["bary"])*q["side"] for q in sites]),
            local[np.array([q["nodal_rows"] for q in sites])],len(candidates))
    det_op=rows(g/det[:,None],local[ids],len(candidates))
    # Safety allowance for round-to-nearest in the one modified Float32 axis.
    axial=np.abs(xyz[:,:,0])+.005
    ulps=abs(np.spacing(axial.astype("<f4"))).astype(float)
    uncertainty=np.sum(np.abs(g)*(.5*ulps/MM)*((local[ids]>=0)),axis=1)
    stencil_cap=5*np.clip((radius_mm-distance[candidates]/MM)/(radius_mm*.55),0,1)
    return (candidates,active,xyz,local[ids],det,diameter,g,det_op,edge,
            lengths,bc,uncertainty,stencil_cap)

def fit(points,tets,role,locked,sites,radius=16,rel_volume=.12,preserve_material_volume=False):
    constructed=system(points,tets,locked,sites,radius)
    (vertices,active,xyz,node_ids,det,diam,g,det_op,edge,lengths,bc,roundoff,caps)=constructed
    n=len(vertices)
    m=len(sites)
    assert n>0
    # A 0.25 mm route clearance is a trial requirement. Its feasibility is
    # measured separately from whether it provides enough tendon thickness.
    claims=np.array([s["gap_m"]/MM+.25 for s in sites])
    target=hstack([-bc,bc,-csr_matrix(np.eye(m))],format="csr")
    hard=vstack([det_op,-det_op,edge,-edge],format="csr")
    # Keep per-cell density perturbations bounded. Very acute reference
    # cells near the original shape floor are handled by the nonlinear
    # diameter cuts below instead of assumed uniform diameter increase.
    lower=np.maximum(FLOOR_DET/det,1-rel_volume)
    rhs=np.r_[np.full(len(det),rel_volume),1-lower,.50*lengths,.50*lengths]
    base_rows,base_rhs=scale_rows(hard,rhs)
    built=hstack([base_rows,-base_rows,csr_matrix((len(rhs),m))],format="csr")
    parts=[built,target]
    rhs_parts=[base_rhs,-claims]
    bounds=[(0,float(x)) for x in caps]*2+[(0,float(g)) for g in claims]
    penalty=np.r_[np.ones(2*n),np.full(m,10000.)]
    exact_volume_rows=None
    if preserve_material_volume:
        # Displacements act on one axis, so summed signed volumes are affine.
        # Normalize by each material's total baseline determinant before solving.
        absolute_det=rows(g,node_ids,n)
        volume=[]
        for material in np.unique(role[active]):
            selected=role[active]==material
            volume.append(csr_matrix(absolute_det[selected].sum(axis=0)/det[selected].sum()))
        op=vstack(volume,format="csr")
        exact_volume_rows=hstack([op,-op,csr_matrix((op.shape[0],m))],format="csr")
    logs=[]
    previous_output=None
    local_tets=tets[active]
    for k in range(5):
        now=time.monotonic()
        res=linprog(penalty,A_ub=vstack(parts,format="csc"),b_ub=np.concatenate(rhs_parts),
                    A_eq=exact_volume_rows,b_eq=None if exact_volume_rows is None else np.zeros(exact_volume_rows.shape[0]),
                    bounds=bounds,method="highs-ipm",options={"time_limit":18,"primal_feasibility_tolerance":1e-8,"threads":1})
        if not res.success:
            logs.append(dict(iteration=k,status=res.message,elapsed_s=time.monotonic()-now,node_rows=n,active_tetrahedra=len(det)))
            return previous_output,logs
        moves=res.x[:n]-res.x[n:2*n]
        cell_scalar=np.r_[moves,0.][np.where(node_ids<0,n,node_ids)]
        actual=points.copy()
        actual[vertices,0]=(points[vertices,0].astype(float)+MM*moves).astype("<f4")
        actual_det,actual_diam,_=tet_metrics(actual[local_tets].astype(float))
        q=actual_det/actual_diam**3
        q_bad=q<FLOOR_QUALITY
        v_bad=actual_det<=FLOOR_DET
        row={"iteration":k,"elapsed_s":time.monotonic()-now,"node_rows":n,
             "active_tetrahedra":len(det),"site_slacks_mm":res.x[2*n:].tolist(),
             "achieved_move_mm":(bc@((actual[vertices,0].astype(float)-points[vertices,0])/MM)).tolist(),
             "max_move_mm":float(np.max(np.abs(moves))),"volume_floor_failures":int(v_bad.sum()),
             "shape_floor_failures":int(q_bad.sum()),
             "volume_ratio_range":[float((actual_det/det).min()),float((actual_det/det).max())]}
        logs.append(row)
        print(json.dumps({"radius_mm":radius,**row}),file=sys.stderr,flush=True)
        if not (v_bad|q_bad).any():
            return actual,logs
        if previous_output is not None and np.array_equal(previous_output,actual):
            return actual,logs
        previous_output=actual
        violated=np.flatnonzero(v_bad|q_bad)
        # The exact determinant is affine in the single-axis motion. For
        # d >= q * max_edge^3 add a tangent lower-bound cut for the longest
        # edge at the rejected Float32 solution. The next replay evaluates the
        # true max again. The roundoff margin is enforced only for witnesses.
        point=xyz[violated].copy()
        point[:,:,0]+=cell_scalar[violated]*MM
        _,_,e=tet_metrics(point)
        size=np.linalg.norm(e,axis=2)
        longest=np.argmax(size,axis=1)
        ii=np.arange(len(violated))
        diameter=size[ii,longest]
        factor=FLOOR_QUALITY*3*diameter*e[ii,longest,0]*MM
        grad=np.zeros_like(g[violated])
        grad[ii,A[longest]]=-factor
        grad[ii,B[longest]]=factor
        constant=FLOOR_QUALITY*diameter**3-np.sum(grad*cell_scalar[violated],axis=1)
        coeff=(grad-g[violated])/det[violated,None]
        floor=np.maximum(constant,FLOOR_DET)
        slack=np.maximum(0,det[violated]-floor)
        b=(slack-np.minimum(.5*slack,roundoff[violated]*1.5))/det[violated]
        extra,new_rhs=scale_rows(rows(coeff,node_ids[violated],n),b)
        parts.append(hstack([extra,-extra,csr_matrix((len(b),m))],format="csr"))
        rhs_parts.append(new_rhs)
    return previous_output,logs

if __name__=="__main__":
    launch, proposal, bed, faces=map(Path,sys.argv[2:6])
    points,tets,material,locked=prepare(launch,proposal,bed)
    base=exact_summary(points,tets,material)
    assert base["volume_floor_failures"]==base["shape_floor_failures"]==0
    sites=selected_sites(faces)
    result=[]
    for radius in map(float,sys.argv[6:]):
        candidate,logs=fit(points,tets,material,locked,sites,radius=radius)
        report={"radius_mm":radius,"solves":logs}
        if candidate is not None:
            assert np.array_equal(candidate[locked],points[locked])
            assert np.array_equal(candidate[:,1:],points[:,1:])
            summary=exact_summary(candidate,tets,material)
            report["whole_volume"]=summary
            if summary["volume_floor_failures"]==summary["shape_floor_failures"]==0:
                changed=np.flatnonzero(np.any(candidate!=points,axis=1))
                saved=Path(f"envelope-{radius:g}mm.npz")
                np.savez_compressed(saved,node_rows=changed,position_f32=candidate[changed])
                report["candidate_sha256"]=digest(saved)
                report["changed_nodes"]=int(len(changed))
        result.append(report)
    print(json.dumps({"mesh_launch_sha256":digest(launch),"conditioned_topology_sha256":digest(proposal),
                      "sites":sites,"fixed_nodes":int(locked.sum()),"baseline":base,"trials":result},indent=2))

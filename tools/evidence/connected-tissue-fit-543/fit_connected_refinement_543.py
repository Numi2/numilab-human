#!/usr/bin/env python3
import argparse, csv, hashlib, json, math, shutil, time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack

P = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
BASE = P/"reduced-physical-boundary-400/attempt-002-parent-ancestry-walk/candidate.npz"
CTRL = P/"conforming-refinement-control-530/run-001/refinement-connectivity.npz"
LEDGER = P/"refined-reference-ledger-534/run-001/child-parent-reference-ledger.npz"
PROP = P/"reduced-physical-boundary-400/attempt-029-matched-fine-edge-bisection/run-001/edge-target-proposal.json"
TEMPLATE = P/"reduced-physical-boundary-400/attempt-029-matched-fine-edge-bisection/run-001"
CAP = P/"native-physical-parent-mass-330/native-source-identity-control-003"
SID = P/"full-current-surface-registration-458/run-001/sidecars"
POSE = CAP/"registered-skin-poses-100.json"
ROLES = ("O","I","D")
ROLE_BASE = {"O":0,"I":2411,"D":4822}
NEW_BASE = {"O":7233,"I":7319,"D":7405}
ROLE_MAT = {"O":15,"I":15,"D":16}
OUT = P/"reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def readcsv(path):
    with open(path,newline="") as f:return list(csv.DictReader(f))
def writecsv(path,rows,fieldnames=None):
    if not rows and fieldnames is None: raise ValueError("need CSV schema")
    fields=fieldnames or list(rows[0].keys())
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
def writejson(path,obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
def vol(Pts,T):
    a=Pts[T[:,1]]-Pts[T[:,0]]
    b=Pts[T[:,2]]-Pts[T[:,0]]
    c=Pts[T[:,3]]-Pts[T[:,0]]
    return np.einsum("ij,ij->i",a,np.cross(b,c))/6.0
def volume_jac(Pts,T,varnode,directions,nvar,rows):
    # Analytic signed-tet-volume gradient projected onto inserted-node target directions.
    tt=T[rows]
    p0=Pts[tt[:,0]];p1=Pts[tt[:,1]];p2=Pts[tt[:,2]];p3=Pts[tt[:,3]]
    g1=np.cross(p2-p0,p3-p0)/6.0
    g2=np.cross(p3-p0,p1-p0)/6.0
    g3=np.cross(p1-p0,p2-p0)/6.0
    g0=-(g1+g2+g3)
    rr=[];cc=[];vv=[]
    for k,g in enumerate((g0,g1,g2,g3)):
        nodes=tt[:,k];vars_=varnode[nodes];mask=vars_>=0
        if np.any(mask):
            rr.append(np.asarray(rows,dtype=np.int64)[mask])
            cc.append(vars_[mask])
            vv.append(np.einsum("ij,ij->i",g[mask],directions[nodes[mask]]))
    if not rr:return csr_matrix((len(rows),nvar),dtype=np.float64)
    J=coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(T.shape[0],nvar)).tocsr()
    return J[rows]
def exact_coords(base,dirs,varnode,alpha,cast32=False):
    x=base.copy()
    ids=np.flatnonzero(varnode>=0)
    x[ids]+=dirs[ids]*alpha[varnode[ids],None]
    if cast32:x=x.astype(np.float32).astype(np.float64)
    return x
def maxmin(a):
    return {"min":float(np.min(a)),"p50":float(np.quantile(a,.5)),"p95":float(np.quantile(a,.95)),"max":float(np.max(a))}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--iterations",type=int,default=4)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--trust-step",type=float,default=.10)
    ap.add_argument("--reference-base",choices=("canonical-source002",),default="canonical-source002")
    args=ap.parse_args()
    if args.iterations<0 or not np.isfinite(args.trust_step) or not 0<args.trust_step<=1:
        ap.error("iterations must be nonnegative and trust-step must be finite in (0,1]")
    OUT=args.output
    start=time.time()
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/"authoring-source-chain").mkdir()
    script=Path(__file__)
    shutil.copy2(script,OUT/"authoring-source-chain"/script.name)
    with np.load(BASE,allow_pickle=False) as z: base={k:z[k].copy() for k in z.files}
    with np.load(CTRL,allow_pickle=False) as z: ctrl={k:z[k].copy() for k in z.files}
    with np.load(LEDGER,allow_pickle=False) as z: ledger={k:z[k].copy() for k in z.files}
    with np.load(TEMPLATE/"candidate.npz",allow_pickle=False) as z: cand={k:z[k].copy() for k in z.files}
    prop=json.loads(PROP.read_text())
    node_rows=readcsv(CAP/"parent-layer-rows.csv")
    if len(node_rows)!=7233:raise ValueError("source parent node-row count changed")
    for i,r in enumerate(node_rows):
        if int(r["volume_node_row"])!=i:raise ValueError("source parent rows not identity ordered")
    nvar=86; nnode=7491
    if ctrl["tetrahedra"].shape!=(29642,4) or len(ctrl["inserted_node_rows"])!=258:raise ValueError("unexpected conforming topology")
    tets=ctrl["tetrahedra"].astype(np.int64)
    if len(ledger["child_mass_kg"])!=len(tets):raise ValueError("534 child mass/tet row mismatch")
    if not np.array_equal(cand["volume_tetrahedra"],tets):raise ValueError("029 topology differs from 530 control")
    for k in ("material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"):
        if not np.array_equal(cand[k],ledger[k]):raise ValueError(f"534 lineage mismatch: {k}")
    if not np.array_equal(cand["child_parent_tet_row"],ledger["child_parent_rows"]):raise ValueError("child source-parent row mismatch")
    # Match the 258 proposal rows to the stable role/edge order in the tested 530 topology.
    targets={int(x["node"]):x for x in prop["targets"]}
    if len(targets)!=258:raise ValueError("expected 258 unique target rows")
    edge_keys_by_role={}
    row_to_var={}
    for ri,role in enumerate(ROLES):
        keys=[]
        for j in range(nvar):
            q=ri*nvar+j
            edge=ctrl["inserted_edge_endpoints"][q]
            b=ROLE_BASE[role]
            key=tuple(sorted((int(edge[0])-b,int(edge[1])-b)))
            keys.append(key)
            row=int(ctrl["inserted_node_rows"][q])
            if row!=NEW_BASE[role]+j:raise ValueError(f"new row ordering changed for {role}")
            row_to_var[row]=j
            rec=targets.get(row)
            if rec is None or rec["role"]!=role:raise ValueError(f"target missing/wrong role for row {row}")
            if tuple(map(int,rec["edge"]))!=tuple(map(int,edge)):raise ValueError(f"target edge mismatch at row {row}")
        edge_keys_by_role[role]=keys
    if not (edge_keys_by_role["O"]==edge_keys_by_role["I"]==edge_keys_by_role["D"]):raise ValueError("O/I/D insertion edges not aligned")
    n=len(tets)
    # Baseline coordinates are source002; inserted nodes start as exact edge midpoints.
    ref_source=base["points_reference_world_m_f64"].astype(np.float64)
    cur0=ref_source.astype(np.float32).astype(np.float64)
    ref0=cur0.copy()
    if len(cur0)!=7233 or len(ref0)!=7233:raise ValueError("source002 coordinate row count changed")
    cur=np.empty((nnode,3),dtype=np.float64);ref=np.empty_like(cur)
    cur[:7233]=cur0
    ref[:7233]=ref0
    edges=ctrl["inserted_edge_endpoints"].astype(np.int64)
    cur[7233:]=np.array([(cur0[a]+cur0[b])*.5 for a,b in edges],dtype=np.float64)
    ref[7233:]=np.array([(ref[:7233][a]+ref[:7233][b])*.5 for a,b in edges],dtype=np.float64)
    if not np.array_equal(cur,ctrl["points_reference_f64"]):
        raise ValueError("canonical source002 midpoint geometry differs from pinned 530 control")
    # Retain each qualified 029 projection on the pinned 458 registered
    # surface.  Recompute only its query point from canonical source002 reference
    # edge midpoints; never replace its local face by a donor-wide nearest face.
    if prop.get("base_sha256")!=sha(BASE):raise ValueError("029 proposal base hash mismatch")
    if prop.get("control_sha256")!=sha(CTRL):raise ValueError("029 proposal topology hash mismatch")
    if prop.get("pose_sha256")!=sha(POSE):raise ValueError("029 proposal pose hash mismatch")
    for role in ROLES:
        if prop.get("registration_csv_sha256",{}).get(role)!=sha(SID/f"registered-{role}.csv"):
            raise ValueError(f"029 registered-{role} CSV hash mismatch")
        if prop.get("registration_faces_sha256",{}).get(role)!=sha(SID/f"registered-{role}-faces.csv"):
            raise ValueError(f"029 registered-{role} face hash mismatch")
    fine_cache={}
    from collections import Counter
    for role in ROLES:
        rr=readcsv(SID/f"registered-{role}.csv")
        ff=readcsv(SID/f"registered-{role}-faces.csv")
        xyz=np.array([[float(x[k]) for k in ("wx","wy","wz")] for x in rr],dtype=np.float64)
        faces=np.array([[int(x[k]) for k in ("v0","v1","v2")] for x in ff],dtype=np.int64)
        labels=np.array([[int(x["parent_geometric_donor_body"]),int(x["parent_anatomical_region_id"])] for x in rr],dtype=np.int64)
        matsfine=np.array([int(x["parent_material_index"]) for x in rr],dtype=np.int64)
        facepairs=[]
        incident=[[] for _ in range(len(xyz))]
        for fi,tri in enumerate(faces):
            cnt=Counter(tuple(map(int,labels[v])) for v in tri)
            facepairs.append(sorted(cnt.items(),key=lambda z:(-z[1],z[0]))[0][0])
            for v in tri:incident[int(v)].append(fi)
        fine_cache[role]=(rr,ff,xyz,faces,labels,matsfine,np.asarray(facepairs,dtype=np.int64),incident)
    canonical_target_distances={}
    projection_error={}
    for role in ROLES:
        rr,ff,xyz,faces,labels,matsfine,facepairs,incident=fine_cache[role]
        errs=[];dists=[]
        for j in range(nvar):
            row=NEW_BASE[role]+j
            rec=targets[row]
            query=cur[row]
            pair=tuple(map(int,rec["group"][1:]))
            fi=int(rec["face"])
            ba=np.asarray(rec["bary"],dtype=np.float64)
            tri=faces[fi]
            if rec["role"]!=role or tuple(map(int,rec["edge"]))!=tuple(map(int,edges[row-7233])):
                raise ValueError(f"029 target role/edge mismatch at {role}/{row}")
            if not np.array_equal(tri,np.asarray(rec["vertices"],dtype=np.int64)):
                raise ValueError(f"029 face vertices do not match pinned 458 face row at {role}/{row}")
            if tuple(map(int,facepairs[fi]))!=pair:
                raise ValueError(f"029 face is outside target donor/region at {role}/{row}: {facepairs[fi]} != {pair}")
            if np.any(matsfine[tri]!=ROLE_MAT[role]):
                raise ValueError(f"029 face material mismatch at {role}/{row}")
            if ba.shape!=(3,) or not np.all(np.isfinite(ba)) or abs(float(ba.sum())-1.0)>1e-9 or float(ba.min()) < -1e-9:
                raise ValueError(f"invalid 029 barycentric coordinates at {role}/{row}")
            projected=ba@xyz[tri]
            declared=np.asarray(rec["world"],dtype=np.float64)
            if np.linalg.norm(projected-declared)>2e-7:
                raise ValueError(f"029 barycentric point does not reproduce declared target at {role}/{row}")
            # Keep the original projection query as historical evidence.
            rec["proposal_query"]=list(rec["query"])
            rec["proposal_projection_distance_m"]=float(rec.get("distance",np.nan))
            rec["query"]=query.tolist()
            rec["canonical_query_to_target_distance_m"]=float(np.linalg.norm(projected-query))
            rec["face_meta"]=[int(ff[fi][k]) for k in ("face_source_layer","face_source_body","face_source_bone","face_source_face_id","face_source_original_id","face_source_identity0","face_source_identity1")]
            errs.append(float(np.linalg.norm(projected-declared)))
            dists.append(float(np.linalg.norm(projected-query)))
        projection_error[role]=max(errs)
        canonical_target_distances[role]=maxmin(np.asarray(dists,dtype=np.float64))
        fa=cand[f"coarse_to_fine_face_{role}"]
        ba=cand[f"coarse_to_fine_bary_{role}"]
        if len(fa)!=2411+nvar or ba.shape!=(2411+nvar,3):
            raise ValueError(f"{role} fine-map row domain changed")
        for j in range(nvar):
            rec=targets[NEW_BASE[role]+j]
            fa[2411+j]=int(rec["face"])
            ba[2411+j]=np.asarray(rec["bary"],dtype=np.float64)
    varnode=np.full(nnode,-1,dtype=np.int64)
    dcur=np.zeros_like(cur);dref=np.zeros_like(ref)
    for row,rec in targets.items():
        v=row_to_var[row]
        q=np.asarray(rec["world"],dtype=np.float64)
        dcur[row]=q-cur[row]
        # Explicit initialized-reference rule: carry the current-world correction
        # into the nearby source reference geometry without rotating it through T-pose rest.
        dref[row]=dcur[row]
        varnode[row]=v
    # Preserve pinned original rows; only the 258 new edge nodes are fit variables.
    assert set(row_to_var)==set(range(7233,7491))
    # Load native current matrices only to serialize each new point's current-pose local coordinate.
    pose=json.loads(POSE.read_text());bodies={int(x["body_index"]):x for x in pose["all_bodies"]}
    owner=np.array([int(x["frame_owner_body"]) for x in node_rows],dtype=np.int64)
    owner_new=[]
    for role in ROLES:
        owner_new += [int(targets[NEW_BASE[role]+j]["group"][1]) for j in range(nvar)]
    owner_new=np.asarray(owner_new,dtype=np.int64)
    Rc=np.zeros((258,3,3));tc=np.zeros((258,3));invRc=np.zeros((258,3,3))
    for k,bid in enumerate(owner_new):
        posebody=bodies[int(bid)]
        Rc[k]=np.asarray(posebody["current"]["local_to_world_row_major_f32"],dtype=np.float64).reshape(3,3)
        tc[k]=np.asarray(posebody["current"]["position_world_m"],dtype=np.float64)
        invRc[k]=np.linalg.inv(Rc[k])
    # Keep node ancestry tied to a source parent tet with matching semantic labels.
    # Refined candidate tet rows are mapped through the 534 child-parent table; they
    # are not interchangeable with original 330 parent tet rows.
    parent_label_rows=readcsv(CAP/"compiled-parent-cell-mass.csv.labels.csv")
    parent_tet_rows=readcsv(CAP/"compiled-parent-cell-mass.csv.tets.csv")
    for row,rec in targets.items():
        pt=int(rec["parent_tet"]);group=tuple(map(int,rec["group"]))
        source_group=tuple(int(parent_label_rows[pt][k]) for k in ("material_index","geometric_donor_body","anatomical_region_id"))
        if source_group!=group:raise ValueError(f"fine target source group differs from pinned original parent source at row {row}: {group} != {source_group}")
        pn=np.array([int(parent_tet_rows[pt][f"parent_node{i}"]) for i in range(4)],dtype=np.int64)
        beta=np.asarray(rec["parent_bary"],dtype=np.float64);edge=np.asarray(rec["edge"],dtype=np.int64)
        if abs(beta.sum()-1)>1e-10 or np.min(beta)<-1e-10 or not set(map(int,edge)).issubset(set(map(int,pn))):
            raise ValueError(f"source tet bary/edge ancestry invalid for new row {row}")
        reconstructed=beta@cur0[pn];mid=(cur0[edge[0]]+cur0[edge[1]])*.5
        if np.linalg.norm(reconstructed-mid)>2e-7:raise ValueError(f"source parent bary does not reconstruct edge midpoint for row {row}")
        childrows=np.flatnonzero((ledger["child_parent_rows"]==pt)&np.all(np.stack([ledger[k] for k in ("material_index","geometric_donor_body","anatomical_region_id")],axis=1)==np.asarray(group)[None,:],axis=1)&np.any(tets==row,axis=1))
        if not len(childrows):raise ValueError(f"no incident refined child tet with matching source ancestry for row {row}")
        rec["candidate_parent_tet"]=int(childrows.min())
    mass=ledger["child_mass_kg"].astype(np.float64)
    mats=ledger["material_index"].astype(np.int64)
    rho_min=np.where(mats==15,900.0,850.0);rho_max=np.where(mats==15,1100.0,950.0)
    vlo=mass/rho_max;vhi=mass/rho_min
    if np.any((mats!=15)&(mats!=16)):raise ValueError("unexpected child material")
    # Positive current-volume floor is scale-aware and construction-only.
    cur0v=vol(cur.astype(np.float32).astype(np.float64),tets)
    ref0v=vol(ref,tets)
    if np.any(cur0v<=0) or np.any(ref0v<=0):raise ValueError("refined source002 baseline has nonpositive signed cells")
    rho0=mass/ref0v
    basebad=np.sum((rho0<rho_min)|(rho0>rho_max))
    if basebad:raise ValueError(f"source002 reference baseline violates 534 per-child density bounds in {basebad} cells")
    affected=np.flatnonzero(np.any(varnode[tets]>=0,axis=1))
    unaffected=np.setdiff1d(np.arange(n),affected,assume_unique=True)
    curfloor=np.maximum(cur0v*1e-4,1e-16)
    # Layer ordering/thickness floors for each matched edge triplet.
    layer_rows_by_role={r:sorted([x for x in node_rows if x["role"]==r],key=lambda x:int(x["role_row"])) for r in ROLES}
    pair_specs=[]
    for j in range(nvar):
        on=NEW_BASE["O"]+j;inn=NEW_BASE["I"]+j;dn=NEW_BASE["D"]+j
        for state,pts,dirs in (("current",cur,dcur),("reference",ref,dref)):
            for a,b,thickfield,sign in ((on,inn,"skin_thickness_m",1.0),(inn,dn,"fat_thickness_m",1.0)):
                diff=pts[a]-pts[b]
                length=float(np.linalg.norm(diff))
                if length<=1e-12:raise ValueError(f"zero baseline paired thickness on edge {j}")
                normal=diff/length
                roleA="O" if a==on else "I"
                ep=edge_keys_by_role["O"][j]
                declared=np.mean([float(layer_rows_by_role[roleA][k][thickfield]) for k in ep])
                floor=min(.25*declared,.5*length)
                floor=max(floor,10e-6)
                if length<floor:raise ValueError(f"baseline layer gap smaller than floor at edge {j}")
                coefficient=float(np.dot(normal,dirs[a]-dirs[b]))
                pair_specs.append((state,j,a,b,normal,floor,coefficient))
    # Edge neighborhood smoothness on alpha prevents isolated unsupported spikes.
    adjacency=set()
    edges_local=edge_keys_by_role["O"]
    incident={}
    for j,e in enumerate(edges_local):
        for v in e:incident.setdefault(v,[]).append(j)
    for js in incident.values():
        for ii in range(len(js)):
            for kk in range(ii+1,len(js)):
                a,b=sorted((js[ii],js[kk]))
                if a!=b:adjacency.add((a,b))
    # Variables are the 86 matched O/I/D edge triplets.
    objective=np.zeros(nvar)
    for j in range(nvar):
        objective[j]=sum(float(np.linalg.norm(dcur[NEW_BASE[r]+j])) for r in ROLES)
    if np.any(objective<=0):raise ValueError("zero target direction in edge triplet")
    def exact_metrics(alpha):
        pc=exact_coords(cur,dcur,varnode,alpha,cast32=True)
        pr=exact_coords(ref,dref,varnode,alpha,cast32=True)
        vc=vol(pc,tets);vr=vol(pr,tets)
        rho=mass/vr
        tb=[]
        for j in range(nvar):
            O=NEW_BASE["O"]+j;I=NEW_BASE["I"]+j;D=NEW_BASE["D"]+j
            for a,b,which,pts in ((O,I,"skin_cur",pc),(I,D,"fat_cur",pc),(O,I,"skin_ref",pr),(I,D,"fat_ref",pr)):
                basevec=(cur if which.endswith("cur") else ref)[a]-(cur if which.endswith("cur") else ref)[b]
                norm=np.linalg.norm(basevec);s=np.dot(pts[a]-pts[b],basevec/norm)
                field="skin_thickness_m" if "skin" in which else "fat_thickness_m"
                ra="O" if "skin" in which else "I"
                ep=edges_local[j]
                th=np.mean([float(layer_rows_by_role[ra][k][field]) for k in ep])
                floor=max(10e-6,min(.25*th,.5*norm))
                if s<floor:tb.append((j,which,float(s),float(floor)))
        badcur=affected[vc[affected]<=curfloor[affected]]
        badref=affected[(vr[affected]<vlo[affected])|(vr[affected]>vhi[affected])|(vr[affected]<=0)]
        return pc,pr,vc,vr,rho,badcur,badref,tb
    def linear_constraints(alpha,step):
        pc=exact_coords(cur,dcur,varnode,alpha,cast32=True)
        pr=exact_coords(ref,dref,varnode,alpha,cast32=False)
        vc=vol(pc,tets);vr=vol(pr,tets)
        Jc=volume_jac(pc,tets,varnode,dcur,nvar,affected)
        Jr=volume_jac(pr,tets,varnode,dref,nvar,affected)
        ca=[];cb=[]
        # Current signed volume remains positive with a small numerical floor.
        # Use per-affected cell lower bound only.
        # For A x <= b: -J x <= v - lower - J a.
        A=[-Jc]
        b=[vc[affected]-curfloor[affected]-Jc@alpha]
        # Reference volume is bounded by the immutable child mass and density limits.
        A += [Jr,-Jr]
        b += [vhi[affected]-vr[affected]+Jr@alpha,vr[affected]-vlo[affected]-Jr@alpha]
        # Paired skin/fat thickness stays positive at a declared fraction of the source field.
        tr=[];tc=[];tv=[];tb=[]
        for k,(_,j,a,bn,normal,floor,coef0) in enumerate(pair_specs):
            pts=pc if pair_specs[k][0]=="current" else pr
            dirs=dcur if pair_specs[k][0]=="current" else dref
            basepts=cur if pair_specs[k][0]=="current" else ref
            s=float(np.dot(pts[a]-pts[bn],normal))
            coef=float(np.dot(normal,dirs[a]-dirs[bn]))
            tr.append(k);tc.append(j);tv.append(-coef);tb.append(s-floor-coef*alpha[j])
        Jt=coo_matrix((tv,(tr,tc)),shape=(len(pair_specs),nvar)).tocsr()
        A.append(Jt);b.append(np.asarray(tb))
        # Neighboring edge fit progress may differ, but only within a bounded local band.
        if adjacency:
            rr=[];cc=[];vv=[];bb=[]
            for r,(i,j) in enumerate(sorted(adjacency)):
                rr.extend((2*r,2*r,2*r+1,2*r+1));cc.extend((i,j,i,j));vv.extend((1.,-1.,-1.,1.));bb.extend((.20,.20))
            Js=coo_matrix((vv,(rr,cc)),shape=(2*len(adjacency),nvar)).tocsr()
            A.append(Js);b.append(np.asarray(bb))
        Aall=vstack(A,format="csr");ball=np.concatenate(b)
        bounds=[(float(alpha[j]),float(min(1.0,alpha[j]+step))) for j in range(nvar)]
        return Aall,ball,bounds
    alpha=np.zeros(nvar,dtype=np.float64)
    base_metrics=exact_metrics(alpha)
    if len(base_metrics[5]) or len(base_metrics[6]) or len(base_metrics[7]):raise ValueError("initial 530 edge-mids fail exact positivity/density/thickness gate")
    logs=[];best=alpha.copy();bestscore=0.0
    for it in range(args.iterations):
        titer=time.time()
        A,b,bounds=linear_constraints(alpha,args.trust_step)
        res=linprog(-objective,A_ub=A,b_ub=b,bounds=bounds,method="highs",options={"time_limit":120})
        if not res.success:
            logs.append({"iteration":it,"status":"LP_FAIL","message":str(res.message),"elapsed_s":time.time()-titer,"nnz":int(A.nnz)})
            break
        proposed=np.asarray(res.x,dtype=np.float64)
        if np.max(proposed-alpha)<=1e-7:
            logs.append({"iteration":it,"status":"NO_PROGRESS","max_delta":float(np.max(proposed-alpha)),"elapsed_s":time.time()-titer})
            break
        accepted=None;beta=1.0;why=None
        for ls in range(12):
            trial=alpha+beta*(proposed-alpha)
            metrics=exact_metrics(trial)
            if not len(metrics[5]) and not len(metrics[6]) and not len(metrics[7]):
                accepted=trial;break
            why={"bad_current":len(metrics[5]),"bad_reference":len(metrics[6]),"bad_thickness":len(metrics[7]),"min_current":float(np.min(metrics[2][affected])),"min_reference":float(np.min(metrics[3][affected]))}
            beta*=.5
        if accepted is None:
            logs.append({"iteration":it,"status":"NONLINEAR_REJECT","last":why,"elapsed_s":time.time()-titer})
            break
        stepgain=float(np.dot(objective,accepted-alpha));alpha=accepted
        score=float(np.dot(objective,alpha))
        if score>bestscore:best=alpha.copy();bestscore=score
        metrics=exact_metrics(alpha)
        logs.append({"iteration":it,"status":"ACCEPT","lp_status":int(res.status),"line_beta":beta,"max_alpha":float(alpha.max()),"median_alpha":float(np.median(alpha)),"min_alpha":float(alpha.min()),"progress_m":stepgain,"total_weighted_progress_m":score,"affected":len(affected),"current_bad":len(metrics[5]),"reference_bad":len(metrics[6]),"thickness_bad":len(metrics[7]),"min_current_volume_m3":float(np.min(metrics[2][affected])),"min_reference_volume_m3":float(np.min(metrics[3][affected])),"elapsed_s":time.time()-titer,"constraint_nnz":int(A.nnz)})
        if stepgain<1e-7:break
    alpha=best
    pc,pr,vc,vr,rho,badc,badr,badt=exact_metrics(alpha)
    # Persist optimized geometry before map/receipt serialization. This file is
    # construction evidence only; exact geometry and native admission remain required.
    np.savez_compressed(OUT/"fitted-geometry-checkpoint.npz",
                        points_current_world_m_f32=pc.astype(np.float32),
                        points_reference_world_m_f64=pr,
                        volume_tetrahedra=tets,accepted_alpha=alpha,
                        child_parent_tet_row=ctrl["child_source_tet_rows"])
    # Recompute full state once; unchanged cells remain exact source copies.
    local=np.empty_like(pc)
    owner_all=np.concatenate([owner,owner_new])
    for i,bid in enumerate(owner_all):
        posebody=bodies[int(bid)]
        Rnow=np.asarray(posebody["current"]["local_to_world_row_major_f32"],dtype=np.float64).reshape(3,3)
        tnow=np.asarray(posebody["current"]["position_world_m"],dtype=np.float64)
        local[i]=np.linalg.inv(Rnow)@(pc[i]-tnow)
    # Candidate arrays retain the tested topology/provenance rows from 029; geometry comes from source002 + fitted targets.
    cand["points_current_world_m_f32"]=pc.astype(np.float32)
    cand["points_reference_world_m_f64"]=pc.astype(np.float64)
    cand["points_donor_local_m_f64"]=local
    cand["target_current_world_m"]=np.zeros((nnode,3),dtype=np.float64)
    cand["target_current_world_m"][:7233]=pc[:7233]
    for row,rec in targets.items():cand["target_current_world_m"][row]=np.asarray(rec["world"],dtype=np.float64)
    parent_weights=base["full86_weights"].copy()
    new_weights=np.asarray([(parent_weights[edge[0]]+parent_weights[edge[1]])*.5 for edge in edges],dtype=np.float64)
    cand["full86_weights"]=np.vstack([parent_weights,new_weights])
    # Verify row map output has all 7,491 rows and update the new 258 records.
    node_map=readcsv(TEMPLATE/"coarse-node-to-fine-surface.csv")
    if len(node_map)!=nnode:raise ValueError("029 node map row count changed")
    target_all=pc.copy()
    for r in node_map:
        row=int(r["volume_node_row"]);role=r["role"]
        dest=pc[row];rref=pr[row];dloc=local[row]
        for ax,v in zip("xyz",dest):r[f"candidate_current_{ax}_m"]=repr(float(v))
        for ax,v in zip("xyz",rref):r[f"candidate_reference_{ax}_m"]=repr(float(v))
        for ax,v in zip("xyz",dloc):r[f"candidate_donor_local_{ax}_m"]=repr(float(v))
        if row>=7233:
            j=row_to_var[row];rec=targets[row];fi=int(rec["face"]);ba=np.asarray(rec["bary"],dtype=np.float64)
            r["accepted_alpha"]=repr(float(alpha[j]));r["fine_surface_face_row"]=str(fi)
            for ix,v in enumerate(ba):r[f"fine_bary{ix}"]=repr(float(v))
            r["fine_parent_donor_body"]=str(int(rec["group"][1]));r["fine_parent_region_id"]=str(int(rec["group"][2]))
        else:
            rawfi=r.get("fine_surface_face_row","")
            fi=int(rawfi) if rawfi not in ("","-1","4294967295") else -1
            ba=np.array([float(r.get(f"fine_bary{k}",0.0)) for k in range(3)],dtype=np.float64)
        target=dest.copy()
        if fi>=0:
            rr,ff,xyz,faces,labels,matsfine,facepairs,incident=fine_cache[role]
            if fi>=len(faces):raise ValueError(f"node map fine face row out of range: {role}/{row}/{fi}")
            target=ba@xyz[faces[fi]]
        target_all[row]=target;resid=target-dest
        for ax,v in zip("xyz",resid):r[f"fine_projection_residual_{ax}_m"]=repr(float(v))
        r["fine_projection_distance_m"]=repr(float(np.linalg.norm(resid)))
    cand["target_current_world_m"]=target_all
    writecsv(OUT/"coarse-node-to-fine-surface.csv",node_map)
    writejson(OUT/"requeried-edge-target-proposal.json",{"schema":"numi.matched-fine-surface-edge-targets.canonical-reference.v1","base_sha256":sha(BASE),"registration_receipt_sha256":sha(SID/"receipt.json"),"targets":[targets[row] for row in sorted(targets)]})
    ancestry=readcsv(TEMPLATE/"inserted-node-ancestry.csv")
    if len(ancestry)!=258:raise ValueError("029 inserted ancestry row count changed")
    for r in ancestry:
        row=int(r["volume_node_row"]);j=row_to_var[row];rec=targets[row];resid=np.asarray(rec["world"],dtype=np.float64)-pc[row]
        r["candidate_parent_tet_row"]=str(int(rec["candidate_parent_tet"]))
        r["accepted_alpha"]=repr(float(alpha[j]));r["accepted_residual_m"]=repr(float(np.linalg.norm(resid)))
        r["fine_surface_face_row"]=str(int(rec["face"]))
        for ix,v in enumerate(rec["bary"]):r[f"fine_bary{ix}"]=repr(float(v))
        for key,value in zip(("face_source_layer","face_source_body","face_source_bone","face_source_face_id","face_source_original_id","face_source_identity0","face_source_identity1"),rec["face_meta"]):
            r[key]=str(int(value))
    writecsv(OUT/"inserted-node-ancestry.csv",ancestry)
    for name in ("child-parent-identity.csv","coarse-boundary-faces.csv","candidate.labels.csv","candidate.ele"):
        shutil.copy2(TEMPLATE/name,OUT/name)
    # Emit standard TetGen rows with current F32 coordinates.
    with open(OUT/"candidate.node","w") as f:
        f.write(f"{nnode} 3 0 0\n")
        for i,q in enumerate(pc):f.write(f"{i+1} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g}\n")
    np.savez_compressed(OUT/"candidate.npz",**cand)
    # Diagnostics, per-child density and local preservation.
    old_rows=np.arange(7233);changed_old=np.linalg.norm(pc[:7233]-base["points_current_world_m_f32"].astype(np.float64),axis=1)>0
    affected_cell=affected
    metrics=exact_metrics(alpha)
    report={
      "schema":"numi.connected-constrained-edge-fit.v1","status":"CONSTRUCTION_CANDIDATE_ONLY",
      "reference_mapping":"canonical source002 reference geometry is the initialized operational geometry for both native current and reference; target corrections are direct initialized-world deltas",
      "reference_mapping_basis":{"source002_historical_ref_current_max_m":float(np.max(np.linalg.norm(base["points_reference_world_m_f64"]-base["points_current_world_m_f32"].astype(np.float64),axis=1))),"source002_historical_ref_current_p95_m":float(np.quantile(np.linalg.norm(base["points_reference_world_m_f64"]-base["points_current_world_m_f32"].astype(np.float64),axis=1),.95)),"operational_baseline":"source002 points_reference_world_m_f64 rounded once to canonical F32 and widened; same coordinates used for current and reference","legacy_rest_absolute_path_used":False,"legacy_rest_rotation_applied_to_delta":False},
      "inputs":{str(x):sha(x) for x in (BASE,CTRL,LEDGER,PROP,POSE,*[SID/f"registered-{r}.csv" for r in ROLES],*[SID/f"registered-{r}-faces.csv" for r in ROLES])},
      "topology":{"node_count":nnode,"tet_count":len(tets),"inserted_node_count":258,"edge_triplet_variable_count":nvar,"affected_tets":len(affected),"unaffected_tets":len(unaffected),"old_nodes_equal_canonical_source_reference_baseline":True,"old_rows_differing_from_historical_source002_current":int(np.count_nonzero(changed_old)),"max_old_historical_current_to_operational_delta_m":float(np.max(np.linalg.norm(pc[:7233]-base["points_current_world_m_f32"].astype(np.float64),axis=1)))},
      "projection_reproduction_max_m":projection_error,"canonical_query_to_retained_target_distance_m":canonical_target_distances,
      "optimizer":{"method":"sequential sparse linear programming with analytic tet-volume gradients; exact-coordinate nonlinear backtracking","iterations":logs,"final_alpha":maxmin(alpha),"weighted_target_progress_m":float(np.dot(objective,alpha)),"target_residual_m_by_role":{r:maxmin(np.linalg.norm(np.array([np.asarray(targets[NEW_BASE[r]+j]["world"]) for j in range(nvar)])-pc[NEW_BASE[r]:NEW_BASE[r]+nvar],axis=1)) for r in ROLES}},
      "gates":{"baseline_reference_density_violations":int(basebad),"final_affected_current_nonpositive_or_floor":int(len(badc)),"final_affected_reference_outside_mass_density":int(len(badr)),"final_paired_thickness_failures":int(len(badt)),"current_signed_volume_m3":maxmin(vc[affected]),"reference_signed_volume_m3":maxmin(vr[affected]),"reference_density_kg_m3_by_material":{str(m):maxmin(rho[(mats==m)&np.isin(np.arange(n),affected)]) for m in (15,16)},"current_nonpositive_total":int(np.sum(vc<=0)),"reference_nonpositive_total":int(np.sum(vr<=0)),"min_current_volume_m3":float(np.min(vc)),"min_reference_volume_m3":float(np.min(vr))},
      "source_preservation":{"parent_node_rows":7233,"all_original_node_rows_and_onehot_parent_ids_retained":True,"canonical_source_reference_coordinates_used_as_operational_baseline":True,"historical_source002_current_coordinates_preserved_as_input_diagnostic":True,"cap_and_head_ring_nodes_preserved_in_canonical_reference_state":True,"original_parent_candidate_sha256":sha(BASE),"all_source_labels_retained_from_029_typed_topology":True},
      "runtime_s":time.time()-start
    }
    writejson(OUT/"receipt.json",report)
    writejson(OUT/"optimizer-log.json",{"iterations":logs,"accepted_alpha":alpha.tolist(),"target_weight_m":objective.tolist(),"status":"complete"})
    # Hash receipt is separate so it can pin the exact emitted artifacts.
    artifact_names=("candidate.npz","candidate.node","candidate.ele","candidate.labels.csv","coarse-node-to-fine-surface.csv","inserted-node-ancestry.csv","child-parent-identity.csv","coarse-boundary-faces.csv")
    writejson(OUT/"artifact-hashes.json",{name:sha(OUT/name) for name in artifact_names})
    print(json.dumps({"out":str(OUT),"candidate_sha256":sha(OUT/"candidate.npz"),"node_count":nnode,"tet_count":len(tets),"iterations":len(logs),"alpha":maxmin(alpha),"progress_m":float(np.dot(objective,alpha)),"current_bad":len(badc),"reference_bad":len(badr),"thickness_bad":len(badt),"seconds":time.time()-start},indent=2))
if __name__=="__main__":main()

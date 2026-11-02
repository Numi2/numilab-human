#!/usr/bin/env python3
"""All-witness adaptive source quadrature proposal with explicit current-11 cuts.
Source-only; does not run the native allocator or alter geometry/physics.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
TASK=H/"current-source-carrier-joint-initialization-1054/cut-closure-1054"
BASE=TASK/"actual-031-adaptive-anchor-joint-lp-001/adaptive_anchor_joint_lp_031h.py"
BASE_PREP=TASK/"actual-031-adaptive-anchor-joint-lp-001/run-007/preparation"
BASE_PREP_REPORT=BASE_PREP/"preparation-report.json"
BASE_SITES=BASE_PREP/"adaptive-sites-1046h.npz"
OUTROOT=TASK/"actual-032-all-witness-adaptive-joint-lp-001"
EXPECTED_BASE="e3dc3f7691f78e727a53dc53c2446a0fc16f783db480ac95e2bcde6a4431785b"
EXPECTED_BASE_PREP_REPORT="4f2fbe417d39e23baa954c4b865f434842ddd0313fbe15c7723f6668918a45f5"
EXPECTED_BASE_SITES="140decacd2f9d55627f47af60a84772dc088e6127c145a244ff9afec619c4476"
R=0.019353
PRIOR_MIX=1e-6
CURRENT_NAMES=[f"current027_component_{i:02d}" for i in range(11)]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()


def pin(path,digest,size=None):
    p=Path(path)
    if not p.is_file() or (size is not None and p.stat().st_size!=size) or sha(p)!=digest:
        raise RuntimeError(f"pinned input mismatch: {p}")


def load_base():
    pin(BASE,EXPECTED_BASE)
    sp=importlib.util.spec_from_file_location("h706_031h_for_032a",BASE)
    if sp is None or sp.loader is None: raise RuntimeError("cannot load frozen 031h owner")
    m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
    return m


def load_inputs(m):
    pin(m.MATRIX,m.EXPECTED_MATRIX);pin(m.EXEC,m.EXPECTED_EXEC);pin(m.MANIFEST,m.EXPECTED_RUN)
    pin(m.OWNER030,m.EXPECTED_030_PY_SOURCE);pin(m.OWNER1046H,m.EXPECTED_1046H);pin(m.PROJECTOR,m.EXPECTED_PROJECTOR)
    ex=json.loads(m.EXEC.read_text());mf=json.loads(m.MANIFEST.read_text());named=mf["input_named_pins"]
    mod=m.loadmod(m.OWNER030,"h706_030_preflight_for032a")
    masstab,labeltab,nodetab,tettab,geom=mod.source_arrays(Path(named["source330_mass"]["path"]),Path(named["source330_labels"]["path"]),Path(named["source330_nodes"]["path"]),Path(named["source330_tets"]["path"]))
    mass=masstab["mass_kg"];mat=masstab["material_index"].astype(np.int64)
    donor=labeltab["geometric_donor_body"].astype(np.int64);region=labeltab["anatomical_region_id"].astype(np.int64);tetxyz=geom["tet_xyz"]
    labels=Path(named["child_skin_labels"]["path"])
    nchild=len(mod.read_csv_numeric(labels,("child_tet_row",))["child_tet_row"])
    n=int(ex["matrix_csv"]["rows"])
    ints,fl,wit,dist=m.parse_matrix(m.MATRIX,n)
    parents=ints["parent_input_row"];baseb=np.column_stack([fl[f"bary{i}"] for i in range(4)])
    route=ints["route_mask"].astype(np.uint32);global_route=ints["global_route"]!=0
    if not np.array_equal(ints["source_parent_row"],parents): raise RuntimeError("030 parent row aliases differ")
    if np.any(parents<0) or np.any(parents>=len(mass)): raise RuntimeError("030 parent rows outside source330")
    if not np.array_equal(ints["material"],mat[parents]) or not np.array_equal(ints["donor"],donor[parents]) or not np.array_equal(ints["region"],region[parents]): raise RuntimeError("030 source family labels changed")
    if np.max(np.abs(fl["parent_mass_kg"]-mass[parents]))>1e-15: raise RuntimeError("030 source parent mass mismatch")
    if np.any(~np.isfinite(baseb)) or np.any(baseb<0) or np.max(np.abs(baseb.sum(axis=1)-1))>2e-12: raise RuntimeError("030 base barycentrics invalid")
    prior=np.zeros(len(mass));np.add.at(prior,parents,fl["prior_weight"])
    if np.max(np.abs(prior-1.0))>2e-12: raise RuntimeError("030 base prior weights do not sum to one per parent")
    if np.any((fl["prior_weight"]>0)&~global_route): raise RuntimeError("prior-supported row lacks a valid primary global witness")
    gw=ints["global_witness_child_row"];gd=fl["global_witness_distance_m"]
    if np.any(global_route&((gw<0)|(gw>=nchild)|~np.isfinite(gd)|(gd>R+1e-12))): raise RuntimeError("030 global witness invalid")
    if np.any(~global_route&((gw!=-1)|np.isfinite(gd))): raise RuntimeError("unroutable base row has spurious global witness")
    cutdoc=json.loads(Path(mf["cut_set_manifest"]["path"]).read_text())
    masks=[];demands=[]
    for j,it in enumerate(cutdoc["cut_order"]):
        rr=m.read_targets(mod,Path(it["target"]["path"]),int(it["target"]["rows"]))
        mask=np.zeros(nchild,bool);mask[rr]=True;masks.append(mask)
        d=it["origin"].get("demand_kg")
        if d is None:
            previous=[demands[k] for k in range(j) if cutdoc["cut_order"][k]["target"]["sha256"]==it["target"]["sha256"]]
            if not previous: raise RuntimeError("cut target has no authenticated demand")
            d=previous[0]
        demands.append(float(d))
    if [x["name"] for x in cutdoc["cut_order"]]!=m.CUT_NAMES or len(demands)!=15: raise RuntimeError("pinned 15-cut order changed")
    return {"m":m,"mod":mod,"ex":ex,"manifest":mf,"named":named,"mass":mass,"material":mat,"donor":donor,"region":region,
        "tetxyz":tetxyz,"nchild":nchild,"ints":ints,"fl":fl,"wit":wit,"dist":dist,"parents":parents,"baseb":baseb,
        "route":route,"global_route":global_route,"global_child":gw,"global_dist":gd,"prior_weight":fl["prior_weight"],
        "masks":masks,"demands15":np.asarray(demands,np.float64),"cutdoc":cutdoc}


def current_component_map(data,current_report):
    m=data["m"];mod=data["mod"];nchild=data["nchild"]
    comps=sorted(current_report["current_components"],key=lambda x:int(x["id"]))
    if [int(c["id"]) for c in comps]!=list(range(11)): raise RuntimeError("current027 component IDs are not 0..10")
    comp_by=np.full(nchild,-1,np.int16);dem=[];refs=[]
    for c in comps:
        p=Path(c["path"]);rows=m.read_targets(mod,p,int(c["rows"]))
        pin(p,c["sha256"])
        if np.any(rows<0)|np.any(rows>=nchild)|np.any(comp_by[rows]>=0): raise RuntimeError("current component rows overlap or are out of range")
        comp_by[rows]=int(c["id"]);dem.append(float(c["minimum_demand_kg"]))
        refs.append({"id":int(c["id"]),"path":str(p),"sha256":c["sha256"],"rows":len(rows),"minimum_demand_kg":float(c["minimum_demand_kg"])})
    if not np.array_equal(comp_by>=0,data["masks"][14]): raise RuntimeError("current 11 components do not exactly partition current027/fullcut target")
    return comp_by,np.asarray(dem,np.float64),refs


def matrix_witness_pairs(data,comp_by):
    """Unique, exact-source-parent witness edges from every 030 cut plus global primary.

    Returns (parent, cut-or--1-for-global, child, recorded-distance, matrix-row).
    """
    parents=data["parents"];route=data["route"];wit=data["wit"];dist=data["dist"];masks=data["masks"];nchild=data["nchild"]
    records=[]
    for j in range(15):
        valid=(route&(1<<j))!=0
        if np.any(valid&((wit[:,j]<0)|(wit[:,j]>=nchild)|~np.isfinite(dist[:,j])|(dist[:,j]>R+1e-12))):
            raise RuntimeError(f"030 cut{j} route has missing/out-of-radius witness")
        ix=np.flatnonzero(valid)
        if not len(ix): continue
        if np.any(~masks[j][wit[ix,j]]): raise RuntimeError(f"030 cut{j} witness is outside authenticated target set")
        order=np.lexsort((ix,dist[ix,j],wit[ix,j],parents[ix]))
        ordered=ix[order];pp=parents[ordered];cc=wit[ordered,j]
        first=np.r_[True,(pp[1:]!=pp[:-1])|(cc[1:]!=cc[:-1])]
        for row in ordered[first]: records.append((int(parents[row]),j,int(wit[row,j]),float(dist[row,j]),int(row)))
    valid=np.flatnonzero(data["global_route"])
    if np.any((data["global_child"][valid]<0)|(data["global_child"][valid]>=nchild)|~np.isfinite(data["global_dist"][valid])|(data["global_dist"][valid]>R+1e-12)):
        raise RuntimeError("030 global primary witness invalid")
    # Global primary rows are valid source-child routes; retain a deterministic nearest record per parent/child.
    if len(valid):
        order=np.lexsort((valid,data["global_dist"][valid],data["global_child"][valid],parents[valid]))
        ordered=valid[order];pp=parents[ordered];cc=data["global_child"][ordered]
        first=np.r_[True,(pp[1:]!=pp[:-1])|(cc[1:]!=cc[:-1])]
        for row in ordered[first]: records.append((int(parents[row]),-1,int(data["global_child"][row]),float(data["global_dist"][row]),int(row)))
    if not records: raise RuntimeError("030 matrix contains no authenticated witness edges")
    return np.asarray(records,dtype=np.float64)


def load_p1(data):
    p1={}
    for material,key in ((15,"p1_skin_support"),(16,"p1_fat_support")):
        with np.load(data["named"][key]["path"],allow_pickle=False) as z:
            offsets=z["cell_admissible_region_offsets"];ids=z["cell_admissible_region_ids"]
        if offsets.ndim!=1 or ids.ndim!=1 or len(offsets)<2 or int(offsets[0])!=0 or int(offsets[-1])!=len(ids) or np.any(offsets[1:]<offsets[:-1]):
            raise RuntimeError(f"invalid pinned P1 CSR for material {material}")
        p1[material]=(offsets,ids)
    return p1


def validate_witness_edges(data,records,rows,child_xyz,childmat,rolelocal,p1):
    parent=data["parents"];region=data["region"];material=data["material"]
    rowpos={int(r):i for i,r in enumerate(rows)}
    for rec in records:
        p=int(rec[0]);child=int(rec[2]);local=int(rolelocal[child])
        if int(childmat[child])!=int(material[p]): raise RuntimeError("030 witness crosses source material")
        if local<0: raise RuntimeError("030 witness has invalid role-local child row")
        offsets,ids=p1[int(material[p])]
        if int(region[p]) not in ids[int(offsets[local]):int(offsets[local+1])]: raise RuntimeError("030 witness fails exact P1 one-hop region admission")
    return rowpos


def build_route_maps(records,comp_by):
    children={};cutchildren={}
    for pv,jv,cv,dv,rv in records:
        p=int(pv);j=int(jv);c=int(cv)
        children.setdefault(p,set()).add(c)
        if j>=0: cutchildren.setdefault((p,j),set()).add(c)
    return children,cutchildren


def base_route_expansion(data,comp_by):
    route=data["route"];wit=data["wit"];dist=data["dist"];parents=data["parents"]
    n=len(parents);curw=np.full((n,11),-1,np.int32);curd=np.full((n,11),np.inf,np.float64)
    for j in range(15):
        valid=((route&(1<<j))!=0)&(wit[:,j]>=0)&np.isfinite(dist[:,j])&(dist[:,j]<=R+1e-12)
        ix=np.flatnonzero(valid)
        if not len(ix): continue
        cid=comp_by[wit[ix,j]]
        within=cid>=0;ix=ix[within];cid=cid[within]
        better=dist[ix,j]<curd[ix,cid]
        ix=ix[better];cid=cid[better]
        curw[ix,cid]=wit[ix,j].astype(np.int32);curd[ix,cid]=dist[ix,j]
    valid=data["global_route"]&(data["global_child"]>=0)&np.isfinite(data["global_dist"])&(data["global_dist"]<=R+1e-12)
    ix=np.flatnonzero(valid)
    if len(ix):
        cid=comp_by[data["global_child"][ix]];within=cid>=0;ix=ix[within];cid=cid[within]
        better=data["global_dist"][ix]<curd[ix,cid];ix=ix[better];cid=cid[better]
        curw[ix,cid]=data["global_child"][ix].astype(np.int32);curd[ix,cid]=data["global_dist"][ix]
    mask=data["route"].astype(np.uint32).copy()
    for c in range(11): mask[curw[:,c]>=0]|=np.uint32(1<<(15+c))
    return mask,curw,curd


def route_candidate_sites(cp,xyz,records,comp_by,rows,child_xyz):
    children,cutchildren=build_route_maps(records,comp_by)
    rowpos={int(r):i for i,r in enumerate(rows)}
    route26=np.zeros(len(cp),np.uint32);cutw=np.full((len(cp),15),-1,np.int32);cutd=np.full((len(cp),15),np.inf,np.float64)
    curw=np.full((len(cp),11),-1,np.int32);curd=np.full((len(cp),11),np.inf,np.float64)
    gw=np.full(len(cp),-1,np.int32);gd=np.full(len(cp),np.inf,np.float64)
    order=np.argsort(cp,kind="stable");ps=cp[order]
    starts=np.r_[0,np.flatnonzero(ps[1:]!=ps[:-1])+1,len(ps)]
    for a,b in zip(starts[:-1],starts[1:]):
        p=int(ps[a]);ix=order[a:b]
        kids=np.asarray(sorted(children.get(p,())),np.int64)
        if not len(kids): continue
        kloc=np.asarray([rowpos[int(c)] for c in kids],np.int64)
        delta=xyz[ix,None,:]-child_xyz[kloc][None,:,:]
        dm=np.sqrt(np.einsum("nmk,nmk->nm",delta,delta))
        ci=comp_by[kids]
        for j in range(15):
            ck=sorted(cutchildren.get((p,j),()))
            if not ck: continue
            pos=np.searchsorted(kids,np.asarray(ck,np.int64));sub=dm[:,pos]
            arg=np.argmin(sub,axis=1);dmin=sub[np.arange(len(ix)),arg];good=dmin<=R+1e-12
            if np.any(good):
                target=np.asarray(ck,np.int64)[arg[good]];dst=dmin[good];outix=ix[good]
                route26[outix]|=np.uint32(1<<j);cutw[outix,j]=target.astype(np.int32);cutd[outix,j]=dst
        for c in range(11):
            pos=np.flatnonzero(ci==c)
            if not len(pos): continue
            sub=dm[:,pos];arg=np.argmin(sub,axis=1);dmin=sub[np.arange(len(ix)),arg];good=dmin<=R+1e-12
            if np.any(good):
                target=kids[pos[arg[good]]];dst=dmin[good];outix=ix[good]
                route26[outix]|=np.uint32(1<<(15+c));curw[outix,c]=target.astype(np.int32);curd[outix,c]=dst
        arg=np.argmin(dm,axis=1);dmin=dm[np.arange(len(ix)),arg];good=dmin<=R+1e-12
        if np.any(good):
            outix=ix[good];gw[outix]=kids[arg[good]].astype(np.int32);gd[outix]=dmin[good]
    keep=gw>=0
    return (keep,route26[keep],cutw[keep],cutd[keep],curw[keep],curd[keep],gw[keep],gd[keep],
            {"candidate_rows":len(cp),"global_routeable_rows":int(keep.sum()),"route_mask_counts_26":[int(np.count_nonzero(route26&(1<<j))) for j in range(26)]})


def candidate_edge_readback(data,comp_by,records,cp,cb,cpoint,mask26,cutw,cutd,curw,curd,gw,gd):
    tetxyz=data["tetxyz"];nchild=data["nchild"]
    if cb.shape!=(len(cp),4) or np.any(cp<0)|np.any(cp>=len(data["mass"])): raise RuntimeError("candidate parent/bary shapes invalid")
    xyz=np.einsum("ni,nij->nj",cb,tetxyz[cp])
    if np.any(cb<0)|np.any(np.abs(cb.sum(axis=1)-1)>2e-12)|np.any(np.abs(xyz-cpoint)>2e-12): raise RuntimeError("candidate source point/bary replay failed")
    if mask26.shape!=(len(cp),) or cutw.shape!=(len(cp),15) or cutd.shape!=(len(cp),15) or curw.shape!=(len(cp),11) or curd.shape!=(len(cp),11): raise RuntimeError("candidate route array shapes invalid")
    # Verify every serialized route witness is a matrix-authenticated exact-parent edge,
    # in the corresponding target set, and within the fixed physical transport bound.
    edge={(int(p),int(j),int(c)) for p,j,c,_,_ in records}
    parent_child={(int(p),int(c)) for p,_,c,_,_ in records}
    distchild={}
    for j in range(15):
        ix=np.flatnonzero((mask26&(1<<j))!=0)
        if np.any(cutw[ix,j]<0)|np.any(~np.isfinite(cutd[ix,j]))|np.any(cutd[ix,j]>R+1e-12): raise RuntimeError(f"candidate cut{j} witness invalid")
        for r in ix:
            c=int(cutw[r,j]);p=int(cp[r])
            if (p,j,c) not in edge: raise RuntimeError("candidate historical route lacks exact 030 parent-child witness")
            if not data["masks"][j][c]: raise RuntimeError("candidate historical witness outside exact target set")
    # Current components route via any already authenticated 030 edge from the same exact source parent.
    for c in range(11):
        ix=np.flatnonzero((mask26&(1<<(15+c)))!=0)
        if np.any(curw[ix,c]<0)|np.any(~np.isfinite(curd[ix,c]))|np.any(curd[ix,c]>R+1e-12): raise RuntimeError("candidate current component witness invalid")
        if len(ix) and np.any(comp_by[curw[ix,c]]!=c): raise RuntimeError("candidate current witness is outside its exact component")
        for r in ix:
            if (int(cp[r]),int(curw[r,c])) not in parent_child:
                raise RuntimeError("candidate current route lacks exact 030 parent-child witness")
    if np.any(gw<0)|np.any(~np.isfinite(gd))|np.any(gd>R+1e-12): raise RuntimeError("candidate global witness invalid")
    for r in range(len(cp)):
        if (int(cp[r]),int(gw[r])) not in parent_child:
            raise RuntimeError("candidate global witness lacks exact 030 parent-child witness")
        if not np.isfinite(gd[r]): raise RuntimeError("candidate global distance missing")
    return xyz


LEX_DEFICIT_TOL=1e-8


def lp_solve_lexicographic(m,mass,ap,bary,mask,global_route,demands,prior,time_limit,deficit_tolerance=LEX_DEFICIT_TOL):
    """First minimize normalized cut deficit; then minimize mass-weighted L1 drift from prior.

    The second LP is constrained to the first optimum plus the explicit numerical tolerance.
    Zero-prior rows need no absolute-value auxiliary because all weights are nonnegative.
    """
    res1,w1,rep1=m.lp_solve(mass,ap,bary,mask,global_route,demands,time_limit)
    if res1.x is None or not res1.success:
        rep={**rep1,"lexicographic":{"first_stage_success":bool(res1.success),"first_stage_message":str(res1.message),"secondary_stage_run":False}}
        return res1,np.empty(0,np.float64),rep
    n=len(ap);p=len(mass);k=len(demands);prior=np.asarray(prior,np.float64)
    if prior.shape!=(n,) or np.any(prior<0) or not np.isfinite(prior).all(): raise RuntimeError("lexicographic prior vector invalid")
    req=np.asarray(demands,np.float64)+m.MARGIN
    pos=np.flatnonzero(prior>0);naux=len(pos);nvar=n+k+naux
    # Exact parent mass and first three independent barycentric moments.
    cidx=np.arange(n,dtype=np.int64)
    eqr=np.concatenate([ap,p+3*ap,p+3*ap+1,p+3*ap+2]);eqc=np.tile(cidx,4)
    eqv=np.concatenate([np.ones(n),bary[:,0],bary[:,1],bary[:,2]])
    Aeq1=coo_matrix((eqv,(eqr,eqc)),shape=(4*p,n+k)).tocsr()
    Aeq=hstack([Aeq1,csr_matrix((4*p,naux))],format="csr")
    beq=np.r_[np.ones(p),np.full(3*p,.25)]
    # Same per-cut lower capacities as the primary LP.
    ur=[];uc=[];uv=[]
    for j in range(k):
        ix=np.flatnonzero(mask&(1<<j))
        if len(ix):
            ur.extend([j]*len(ix));uc.extend(ix.tolist());uv.extend((-mass[ap[ix]]/req[j]).tolist())
        ur.append(j);uc.append(n+j);uv.append(-1.)
    Aubcut1=coo_matrix((uv,(ur,uc)),shape=(k,n+k)).tocsr()
    Aubcut=hstack([Aubcut1,csr_matrix((k,naux))],format="csr")
    # Preserve the optimal first-stage sum of normalized shortfalls within the declared tolerance.
    rr=np.zeros(k,dtype=np.int64);cc=np.arange(n,n+k,dtype=np.int64);vv=np.ones(k)
    Aopt1=coo_matrix((vv,(rr,cc)),shape=(1,n+k)).tocsr()
    Aopt=hstack([Aopt1,csr_matrix((1,naux))],format="csr")
    bopt=float(res1.fun)+float(deficit_tolerance)
    # L1 epigraph only for positive-prior rows: u_i >= |x_i - prior_i|.
    if naux:
        ar=np.repeat(np.arange(2*naux,dtype=np.int64),2)
        ac=np.empty(4*naux,dtype=np.int64);av=np.empty(4*naux,dtype=np.float64)
        auxcol=n+k+np.arange(naux,dtype=np.int64)
        ac[0::2]=np.repeat(pos,2)
        ac[1::2]=np.repeat(auxcol,2)
        av[0::4]=1.0;av[1::4]=-1.0;av[2::4]=-1.0;av[3::4]=-1.0
        Al1=coo_matrix((av,(ar,ac)),shape=(2*naux,nvar)).tocsr()
        bl1=np.column_stack([prior[pos],-prior[pos]]).ravel()
    else:
        Al1=csr_matrix((0,nvar));bl1=np.empty(0,np.float64)
    Aub=vstack([Aubcut,Aopt,Al1],format="csr")
    bub=np.r_[-np.ones(k),bopt,bl1]
    obj=np.zeros(nvar,np.float64)
    zero_prior=np.flatnonzero(prior==0)
    obj[zero_prior]=mass[ap[zero_prior]]
    if naux: obj[n+k:]=mass[ap[pos]]
    bounds=np.empty((nvar,2),np.float64);bounds[:,0]=0.;bounds[:,1]=np.inf
    route=np.asarray(global_route,dtype=bool)
    if route.shape!=(n,):raise RuntimeError("lexicographic global route vector length mismatch")
    bounds[np.flatnonzero(~route),1]=0.
    res2=linprog(obj,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds,method="highs",
        options={"time_limit":time_limit,"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10,"presolve":True})
    if res2.x is None or not res2.success:
        rep={**rep1,"lexicographic":{"first_stage_optimal_deficit_sum":float(res1.fun),"deficit_tolerance":float(deficit_tolerance),
            "second_stage_success":bool(res2.success),"second_stage_status":int(res2.status),"second_stage_message":str(res2.message),"secondary_stage_run":True}}
        return res2,np.empty(0,np.float64),rep
    w=np.asarray(res2.x[:n],np.float64);slack=np.asarray(res2.x[n:n+k],np.float64)
    sums=np.bincount(ap,weights=w,minlength=p)
    mom=np.column_stack([np.bincount(ap,weights=w*bary[:,q],minlength=p) for q in range(4)])
    caps=np.array([np.sum(mass[ap[(mask&(1<<j))!=0]]*w[(mask&(1<<j))!=0]) for j in range(k)])
    norm=np.maximum(0.,(req-caps)/req)
    deficit_sum=float(slack.sum());within=bool(deficit_sum<=float(res1.fun)+deficit_tolerance+1e-10)
    rep={"solver_success":bool(res2.success),"solver_status":int(res2.status),"solver_message":str(res2.message),
        "parent_weight_sum_error_max":float(np.max(np.abs(sums-1))),"parent_barycentric_moment_error_max":float(np.max(np.abs(mom-.25))),
        "positive_weight_count":int(np.count_nonzero(w>0)),"normalized_shortfall_variable":slack.tolist(),"normalized_shortfall_replay":norm.tolist(),
        "shortfall_kg":np.maximum(0.,req-caps).tolist(),"capacity_kg":caps.tolist(),"required_kg_with_margin":req.tolist(),
        "shortfall_replay_error_max":float(np.max(np.abs(norm-slack))),
        "passed_parent_moments":bool(np.max(np.abs(sums-1))<=2e-10 and np.max(np.abs(mom-.25))<=2e-10),
        "passed_slack_replay":bool(np.max(np.abs(norm-slack))<=2e-8),
        "lexicographic":{"first_stage_optimal_deficit_sum":float(res1.fun),"second_stage_slack_sum":deficit_sum,
            "deficit_tolerance":float(deficit_tolerance),"within_first_stage_optimum_tolerance":within,
            "second_stage_mass_weighted_l1_objective_kg":float(res2.fun),"positive_prior_rows_with_l1_auxiliaries":int(naux),
            "zero_prior_rows_directly_penalized":int(len(zero_prior)),"secondary_stage_success":True}}
    return res2,w,rep


def preserve_prior_weights(lp_weights,prior_weights,parent_rows,bary,route_masks,parent_count,epsilon=PRIOR_MIX):
    lp=np.asarray(lp_weights,np.float64);prior=np.asarray(prior_weights,np.float64);parents=np.asarray(parent_rows,np.int64)
    b=np.asarray(bary,np.float64);masks=np.asarray(route_masks,np.uint32)
    if lp.shape!=prior.shape or lp.shape!=(len(parents),) or b.shape!=(len(parents),4) or masks.shape!=(len(parents),):
        raise RuntimeError("prior preservation input shapes do not align")
    if not np.isfinite(lp).all() or not np.isfinite(prior).all() or not np.isfinite(b).all() or np.any(lp<0) or np.any(prior<0):
        raise RuntimeError("prior preservation input contains nonfinite or negative values")
    order=np.argsort(parents,kind="stable");sp=parents[order];bounds=np.r_[0,np.flatnonzero(sp[1:]!=sp[:-1])+1,len(sp)]
    fixed=np.zeros(parent_count,bool)
    for a,z in zip(bounds[:-1],bounds[1:]):
        parent=int(sp[a]);fixed[parent]=len(np.unique(masks[order[a:z]]))==1
    result=(1.0-epsilon)*lp+epsilon*prior
    for a,z in zip(bounds[:-1],bounds[1:]):
        parent=int(sp[a])
        if fixed[parent]:
            ix=order[a:z];result[ix]=prior[ix]
    sums=np.bincount(parents,weights=result,minlength=parent_count)
    moments=np.column_stack([np.bincount(parents,weights=result*b[:,q],minlength=parent_count) for q in range(4)])
    finite=bool(np.isfinite(result).all() and np.isfinite(sums).all() and np.isfinite(moments).all())
    nonnegative=bool(np.all(result>=0.0))
    sumerr=float(np.max(np.abs(sums-1.0)))
    momerr=float(np.max(np.abs(moments-.25)))
    passed=bool(finite and nonnegative and sumerr<=2e-10 and momerr<=2e-10)
    stats={"prior_mix_fraction_for_route_variable_parents":float(epsilon),"route_fixed_parents_restored_exact_prior":int(fixed.sum()),
        "route_variable_parents_blended":int(parent_count-fixed.sum()),"finite_weights_masses_moments":finite,"nonnegative_weights":nonnegative,
        "all_parent_weight_sum_error_max":sumerr,"all_parent_barycentric_moment_error_max":momerr,"passed_post_restore_blend_mass_moment_gate":passed}
    return result,fixed,stats


def self_test():
    # Two source parents with distinct current component IDs. For parent 0,
    # the nearest seed child (10) is absent from the candidate neighborhood,
    # while the second authenticated witness (11) restores the route.
    comp=np.full(20,-1,np.int16);comp[[10,11,12,13]]=[2,4,8,9]
    rows=np.array([10,11,12,13],np.int64)
    childxyz=np.array([[0.,0.,0.],[.030,0.,0.],[.010,0.,0.],[.510,0.,0.]])
    records=np.array([[0,2,10,.015,0],[0,2,11,.015,1],[0,5,12,.005,2],[1,2,13,.010,3]],np.float64)
    cp=np.array([0,1,1],np.int64);xyz=np.array([[.030,0.,0.],[.510,0.,0.],[.600,0.,0.]])
    keep,mask,cw,cd,curw,curd,gw,gd,_=route_candidate_sites(cp,xyz,records,comp,rows,childxyz)
    assert keep.tolist()==[True,True,False],"radius gate did not reject over-radius site"
    assert int(cw[0,2])==11,"all-witness route did not recover the alternative child"
    assert int(mask[0])&(1<<2) and int(mask[0])&(1<<(15+4)),"historical/current component route bits missing"
    assert not (int(mask[0])&(1<<(15+2))) and not (int(mask[0])&(1<<(15+8))),"component route leaked across disjoint target IDs"
    assert int(cw[1,2])==13 and int(mask[1])&(1<<(15+9)),"second parent/current component route failed"
    tetxyz=np.array([np.full((4,3),.030),np.array([[.510,0,0],[.870,0,0],[.510,0,0],[.510,0,0]])],np.float64)
    bary=np.array([[.25,.25,.25,.25],[1.,0,0,0],[.25,.25,.25,.25]],np.float64)
    parents=cp;points=np.einsum("ni,nij->nj",bary,tetxyz[parents])
    masks=[np.zeros(20,bool) for _ in range(15)];masks[2][[10,11,13]]=True;masks[5][12]=True
    fake={"tetxyz":tetxyz,"mass":np.ones(2),"nchild":20,"masks":masks}
    rcp=cp[keep];rbary=bary[keep];rpoints=points[keep]
    passed_xyz=candidate_edge_readback(fake,comp,records,rcp,rbary,rpoints,mask,cw,cd,curw,curd,gw,gd)
    assert np.max(np.abs(passed_xyz-rpoints))==0
    badcur=curw.copy();badcur[0,4]=10
    try: candidate_edge_readback(fake,comp,records,rcp,rbary,rpoints,mask,cw,cd,badcur,curd,gw,gd)
    except RuntimeError: pass
    else: raise AssertionError("wrong current-component child witness was accepted")
    badcutd=cd.copy();badcutd[0,2]=np.nan
    try: candidate_edge_readback(fake,comp,records,rcp,rbary,rpoints,mask,cw,badcutd,curw,curd,gw,gd)
    except RuntimeError: pass
    else: raise AssertionError("NaN witness distance was accepted")
    badcutd=cd.copy();badcutd[0,2]=R+1e-3
    try: candidate_edge_readback(fake,comp,records,rcp,rbary,rpoints,mask,cw,badcutd,curw,curd,gw,gd)
    except RuntimeError: pass
    else: raise AssertionError("over-radius witness was accepted")
    prior=np.array([.25]*4+[0.]+[.25]*4,np.float64)
    lp=np.array([.20]*5+[.25]*4,np.float64)
    lpp=np.array([0]*5+[1]*4,np.int64)
    lpbar=np.vstack([np.eye(4),np.full((1,4),.25),np.eye(4)])
    lpmask=np.array([1,1,2,2,2,4,4,4,4],np.uint32)
    wf,fixed,stats=preserve_prior_weights(lp,prior,lpp,lpbar,lpmask,2)
    sums=np.bincount(lpp,weights=wf,minlength=2)
    moments=np.column_stack([np.bincount(lpp,weights=wf*lpbar[:,q],minlength=2) for q in range(4)])
    assert fixed.tolist()==[False,True] and stats["passed_post_restore_blend_mass_moment_gate"]
    assert np.array_equal(wf[5:],prior[5:]) and np.max(np.abs(sums-1))<1e-14 and np.max(np.abs(moments-.25))<1e-14
    assert np.all(wf>=0) and np.all(np.isfinite(wf))
    m=load_base()
    lres,lw,lrep=lp_solve_lexicographic(m,np.array([1.0]),np.zeros(4,np.int64),np.eye(4),np.ones(4,np.uint32),np.ones(4,bool),np.array([1.0]),np.full(4,.25),5.)
    assert lres.success and lrep["lexicographic"]["within_first_stage_optimum_tolerance"]
    assert lrep["passed_parent_moments"] and lrep["passed_slack_replay"] and np.max(np.abs(lw-.25))<1e-9
    print(json.dumps({"status":"PASS_032D_SYNTHETIC_CONTROLS","alternative_witness_child":11,"current_component_bits":[4,9],
        "radius_rejection":True,"tamper_rejections":["wrong_component","NaN_distance","over_radius"],"prior_blend":stats,
        "lexicographic_two_stage":{"within_deficit_tolerance":lrep["lexicographic"]["within_first_stage_optimum_tolerance"],"l1_objective_kg":lrep["lexicographic"]["second_stage_mass_weighted_l1_objective_kg"]}},sort_keys=True))


def prepare(out):
    m=load_base();data=load_inputs(m);mod=data["mod"]
    if out.exists(): raise RuntimeError("refusing to overwrite "+str(out))
    out.mkdir(parents=True)
    preflight=m.run030preflight(mod,out)
    current=m.current11_reconcile(data["manifest"],data["cutdoc"],data["masks"],data["wit"],data["nchild"],out/"current11-target-reconciliation.json")
    comp_by,dem11,comprefs=current_component_map(data,current)
    records=matrix_witness_pairs(data,comp_by)
    allchildren=np.unique(records[:,2].astype(np.int64))
    rows,childxyz,childmat,rolelocal=m.selected_child_geometry(allchildren,Path(data["named"]["child_skin_node"]["path"]),Path(data["named"]["child_skin_elements"]["path"]),Path(data["named"]["child_skin_labels"]["path"]),data["nchild"])
    p1=load_p1(data);validate_witness_edges(data,records,rows,childxyz,childmat,rolelocal,p1)
    base26,basecurw,basecurd=base_route_expansion(data,comp_by)
    base_route_path=out/"base-route-expansion-032.npz"
    np.savez_compressed(base_route_path,route_mask_26=base26,current11_witness_child_rows=basecurw,current11_witness_distance_m=basecurd)
    pair=records[(records[:,1]==2)|(records[:,1]==5)]
    if not len(pair): raise RuntimeError("no residual-cut all-witness pairs")
    pairp=pair[:,0].astype(np.int64);paircut=pair[:,1].astype(np.int64);pairchild=pair[:,2].astype(np.int64);pairdist=pair[:,3].astype(np.float64)
    if len(np.unique(np.column_stack([pairp,paircut,pairchild]),axis=0))!=len(pair): raise RuntimeError("duplicate all-witness pair key")
    rowpos=np.searchsorted(rows,pairchild);queries=childxyz[rowpos]
    gen=m.generate_sites(data["tetxyz"],queries,pairp,paircut,pairchild,pairdist,m.PROJECTOR)
    newp,newb,newseedcut,newseedchild,newkind,newxyz,newseeddist,newproof=gen
    pin(BASE_PREP_REPORT,EXPECTED_BASE_PREP_REPORT);baseprep=json.loads(BASE_PREP_REPORT.read_text())
    baseref=baseprep["adaptive_sites_npz"];pin(BASE_SITES,EXPECTED_BASE_SITES,int(baseref["bytes"]))
    with np.load(BASE_SITES,allow_pickle=False) as z:
        oldp=z["source_parent_row"].astype(np.int64);oldb=z["barycentric"].astype(np.float64);oldseedcut=z["seed_cut_index"].astype(np.int64)
        oldseedchild=z["seed_child_row"].astype(np.int64);oldkind=z["site_kind"].astype(np.int64);oldxyz=z["source_point_world_m"].astype(np.float64)
    cp=np.r_[oldp,newp];cb=np.vstack([oldb,newb]);seedcut=np.r_[oldseedcut,newseedcut];seedchild=np.r_[oldseedchild,newseedchild];kind=np.r_[oldkind,newkind]
    # Exact parent+bary de-dup; old 031h candidates keep deterministic precedence.
    _,ix=np.unique(np.column_stack([cp.astype(np.float64),cb]),axis=0,return_index=True);ix=np.sort(ix)
    cp=cp[ix];cb=cb[ix];seedcut=seedcut[ix];seedchild=seedchild[ix];kind=kind[ix]
    cxyz=np.einsum("ni,nij->nj",cb,data["tetxyz"][cp])
    expanded=route_candidate_sites(cp,cxyz,records,comp_by,rows,childxyz)
    keep,mask26,cutw,cutd,curw,curd,gw,gd,route_summary=expanded
    cp=cp[keep];cb=cb[keep];seedcut=seedcut[keep];seedchild=seedchild[keep];kind=kind[keep];cxyz=cxyz[keep]
    xyz=candidate_edge_readback(data,comp_by,records,cp,cb,cxyz,mask26,cutw,cutd,curw,curd,gw,gd)
    route_path=out/"adaptive-sites-032.npz"
    np.savez_compressed(route_path,source_parent_row=cp.astype(np.int32),barycentric=cb,source_point_world_m=xyz,
        seed_cut_index=seedcut.astype(np.int16),seed_child_row=seedchild.astype(np.int32),site_kind=kind.astype(np.int8),
        route_mask_26=mask26.astype(np.uint32),route_mask=(mask26&((1<<15)-1)).astype(np.uint16),current11_route_mask=(mask26>>15).astype(np.uint16),
        cut_witness_child_rows=cutw.astype(np.int32),cut_witness_distance_m=cutd,
        current11_witness_child_rows=curw.astype(np.int32),current11_witness_distance_m=curd,
        global_route=np.ones(len(cp),np.uint8),global_witness_child_row=gw.astype(np.int32),global_witness_distance_m=gd)
    names26=list(m.CUT_NAMES)+CURRENT_NAMES;dem26=np.r_[data["demands15"],dem11]
    report={"schema":"numi.h706.all-witness-adaptive-current11-joint-lp-032d-preparation.v1","status":"PASS_SOURCE_ONLY_PREPARATION",
        "physical_admission":False,"native_steps":0,"allocator_run":False,"radius_m":R,
        "input_pins":{"031h_owner":{"path":str(BASE),"sha256":EXPECTED_BASE},"031h_preparation":{"path":str(BASE_PREP_REPORT),"sha256":EXPECTED_BASE_PREP_REPORT},
            "031h_sites":{"path":str(BASE_SITES),"sha256":EXPECTED_BASE_SITES},"030_matrix":{"path":str(m.MATRIX),"sha256":m.sha(m.MATRIX)},
            "030_execution":{"path":str(m.EXEC),"sha256":m.sha(m.EXEC)},"030_manifest":{"path":str(m.MANIFEST),"sha256":m.sha(m.MANIFEST)},
            "030_matrix_sha256":m.sha(m.MATRIX),"030_run_manifest_sha256":m.sha(m.MANIFEST),"030_execution_sha256":m.sha(m.EXEC),
            "current_support_sha256":m.EXPECTED_SUPPORT,"current027_manifest_sha256":m.EXPECTED_CURRENT027_MANIFEST,
            "current_component_census_sha256":m.EXPECTED_COMP_CENSUS,"current_fullcut_witness_sha256":m.EXPECTED_SUPPORT_MANIFEST},
        "source_only_030_preflight":{"path":str(preflight),"sha256":sha(preflight),"status":json.loads(preflight.read_text()).get("status"),"physical_admission":False,"native_steps":0},
        "current11_reconciliation":current,"current11_component_targets":comprefs,
        "historical_cut_names":m.CUT_NAMES,"lp_cut_names_26":names26,"lp_demands_kg_26":dem26.tolist(),
        "witness_edge_counts":{"unique_parent_cut_child_triples_all_15":int(sum(1 for r in records if r[1]>=0)),
            "unique_parent_cut_child_triples_cut2":int(np.count_nonzero(records[:,1]==2)),"unique_parent_cut_child_triples_cut5":int(np.count_nonzero(records[:,1]==5)),
            "unique_parent_child_pairs":int(len(np.unique(records[:,[0,2]],axis=0))),
            "unique_residual_seed_pairs_cut2_and_5":int(len(pair)),"source_children_in_witness_pool":int(len(rows))},
        "new_site_generation":newproof,"routed_candidate_summary":route_summary,
        "base_route_expansion_npz":{"path":str(base_route_path),"sha256":sha(base_route_path),"bytes":base_route_path.stat().st_size},
        "adaptive_sites_npz":{"path":str(route_path),"sha256":sha(route_path),"bytes":route_path.stat().st_size,"rows":len(cp)},
        "construction":"source_tetra_anchor_refinement_1046d","provenance_mode":"CurrentCarrierPointProjection",
        "limitations":["Candidate edges are limited to the exact parent-child witness pairs present in pinned 030 matrix; this is not an exhaustive all-parent/all-child graph.",
            "The 26 source-only cut constraints comprise the 15 pinned historical cuts plus all 11 exact current027 components. They are capacity relaxations, not shared-flow proof.",
            "No native allocator, mass-flow, source geometry, density, semantic, radius, or physical-admission result is claimed."]}
    rp=out/"preparation-report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":report["status"],"report":str(rp),"sha256":sha(rp),"sites":len(cp),"cut2_pairs":int(np.count_nonzero(records[:,1]==2)),"cut5_pairs":int(np.count_nonzero(records[:,1]==5))},sort_keys=True))
    return 0


def verify_candidate_arrays(data,comp_by,records,prep,route_path):
    pin(route_path,prep["adaptive_sites_npz"]["sha256"],prep["adaptive_sites_npz"]["bytes"])
    with np.load(route_path,allow_pickle=False) as z:
        cp=z["source_parent_row"].astype(np.int64);cb=z["barycentric"].astype(np.float64);cpoint=z["source_point_world_m"].astype(np.float64)
        mask26=z["route_mask_26"].astype(np.uint32);cutw=z["cut_witness_child_rows"].astype(np.int64);cutd=z["cut_witness_distance_m"].astype(np.float64)
        curw=z["current11_witness_child_rows"].astype(np.int64);curd=z["current11_witness_distance_m"].astype(np.float64)
        gw=z["global_witness_child_row"].astype(np.int64);gd=z["global_witness_distance_m"].astype(np.float64)
        seedcut=z["seed_cut_index"].astype(np.int64);seedchild=z["seed_child_row"].astype(np.int64);kind=z["site_kind"].astype(np.int64)
    xyz=np.einsum("ni,nij->nj",cb,data["tetxyz"][cp])
    if np.max(np.abs(xyz-cpoint))>2e-12 or np.any(cb<0) or np.max(np.abs(cb.sum(axis=1)-1))>2e-12: raise RuntimeError("prepared candidate source point/moment invalid")
    expected=route_candidate_sites(cp,xyz,records,comp_by,prep["_rows"],prep["_child_xyz"])
    keep,em,ecw,ecd,erw,erd,egw,egd,summary=expected
    if not np.all(keep) or not np.array_equal(mask26,em) or not np.array_equal(cutw,ecw) or not np.array_equal(curw,erw) or not np.array_equal(gw,egw): raise RuntimeError("prepared route witnesses/masks do not replay from pinned matrix edges")
    if not np.allclose(cutd,ecd,rtol=0,atol=2e-12) or not np.allclose(curd,erd,rtol=0,atol=2e-12) or not np.allclose(gd,egd,rtol=0,atol=2e-12): raise RuntimeError("prepared route distances do not replay")
    return cp,cb,xyz,mask26,cutw,cutd,curw,curd,gw,gd,seedcut,seedchild,kind


def solve(prepdir,out,time_limit,expected_preparation_sha256):
    m=load_base();data=load_inputs(m)
    if out.exists(): raise RuntimeError("refusing to overwrite "+str(out))
    prepdir=prepdir.resolve();pr=prepdir/"preparation-report.json"
    if sha(pr)!=expected_preparation_sha256: raise RuntimeError("032 preparation report does not match explicit frozen pin")
    prep=json.loads(pr.read_text())
    if prep.get("status")!="PASS_SOURCE_ONLY_PREPARATION" or prep.get("input_pins",{}).get("030_matrix",{}).get("sha256")!=m.sha(m.MATRIX): raise RuntimeError("032 prepared source pins/status mismatch")
    comp_by,dem11,refs=current_component_map(data,prep["current11_reconciliation"])
    records=matrix_witness_pairs(data,comp_by)
    allchildren=np.unique(records[:,2].astype(np.int64));rows,childxyz,childmat,rolelocal=m.selected_child_geometry(allchildren,Path(data["named"]["child_skin_node"]["path"]),Path(data["named"]["child_skin_elements"]["path"]),Path(data["named"]["child_skin_labels"]["path"]),data["nchild"])
    validate_witness_edges(data,records,rows,childxyz,childmat,rolelocal,load_p1(data))
    basefile=prepdir/"base-route-expansion-032.npz";pin(basefile,prep["base_route_expansion_npz"]["sha256"],prep["base_route_expansion_npz"]["bytes"])
    with np.load(basefile,allow_pickle=False) as z: base26=z["route_mask_26"].astype(np.uint32);basecw=z["current11_witness_child_rows"].astype(np.int64);basecd=z["current11_witness_distance_m"].astype(np.float64)
    if base26.shape!=(len(data["parents"]),) or basecw.shape!=(len(data["parents"]),11) or basecd.shape!=(len(data["parents"]),11): raise RuntimeError("base route expansion shape mismatch")
    route_path=prepdir/"adaptive-sites-032.npz"
    pp={"adaptive_sites_npz":prep["adaptive_sites_npz"],"_rows":rows,"_child_xyz":childxyz}
    cp,cb,cxyz,cm,cw,cd,curw,curd,cgw,cgd,seedcut,seedchild,kind=verify_candidate_arrays(data,comp_by,records,pp,route_path)
    nbase=len(data["parents"]);allp=np.r_[data["parents"],cp];allb=np.vstack([data["baseb"],cb])
    allm=np.r_[base26,cm].astype(np.uint32);globalroute=np.r_[data["global_route"],np.ones(len(cp),bool)]
    demands=np.r_[data["demands15"],dem11];names26=list(m.CUT_NAMES)+CURRENT_NAMES
    print(f"LP_START anchors={len(allp)} parents={len(data['mass'])} constraints={len(demands)} adaptive={len(cp)}",flush=True)
    prior=np.zeros(len(allp),np.float64);prior[:nbase]=data["prior_weight"]
    t=time.monotonic();res,w,rep=lp_solve_lexicographic(m,data["mass"],allp,allb,allm,globalroute,demands,prior,time_limit);elapsed=time.monotonic()-t
    out.mkdir(parents=True)
    output=None;blend_rep=None
    if len(w):
        w=np.asarray(w,dtype=np.float64);nvar=len(w)
        # Preserve the exact inherited quadrature for parents whose every available anchor
        # has the same route mask; changing their weights cannot improve any cut.
        order=np.argsort(allp,kind="stable");sp=allp[order];bounds=np.r_[0,np.flatnonzero(sp[1:]!=sp[:-1])+1,len(sp)]
        fixed=np.zeros(len(data["mass"]),bool)
        for a,b in zip(bounds[:-1],bounds[1:]):
            p=int(sp[a]);fixed[p]=len(np.unique(allm[order[a:b]]))==1
        wf,fixed,blend_stats=preserve_prior_weights(w,prior,allp,allb,allm,len(data["mass"]),PRIOR_MIX)
        capacities=np.array([np.sum(data["mass"][allp[(allm&(1<<j))!=0]]*wf[(allm&(1<<j))!=0]) for j in range(len(demands))])
        req=demands+m.MARGIN;short=np.maximum(0,req-capacities)
        finite=bool(blend_stats["finite_weights_masses_moments"] and np.isfinite(capacities).all())
        passed_invariants=bool(blend_stats["passed_post_restore_blend_mass_moment_gate"] and finite)
        blend_rep={**blend_stats,"finite_weights_masses_moments":finite,
            "capacity_after_restore_and_blend_kg":capacities.tolist(),"shortfall_after_restore_and_blend_kg":short.tolist(),
            "all_26_constraints_met_after_restore_and_blend":bool(np.all(short<=1e-10)),
            "prior_global_route_positive_rows_checked":True}
        posnew=np.flatnonzero(wf[nbase:]>0);selected=posnew
        if passed_invariants:
            op=out/"adaptive-joint-weight-proposal-032d.npz"
            np.savez_compressed(op,source_parent_row=cp[selected].astype(np.int32),barycentric=cb[selected],weight=wf[nbase:][selected],source_point_world_m=cxyz[selected],
                seed_cut_index=seedcut[selected].astype(np.int16),seed_child_row=seedchild[selected].astype(np.int32),site_kind=kind[selected].astype(np.int8),
                cut_witness_child_rows=cw[selected].astype(np.int32),cut_witness_distance_m=cd[selected],route_mask=(cm[selected]&((1<<15)-1)).astype(np.uint16),
                current11_witness_child_rows=curw[selected].astype(np.int32),current11_witness_distance_m=curd[selected],current11_route_mask=(cm[selected]>>15).astype(np.uint16),route_mask_26=cm[selected].astype(np.uint32),
                global_route=np.ones(len(selected),np.uint8),global_witness_child_row=cgw[selected].astype(np.int32),global_witness_distance_m=cgd[selected],
                matrix_anchor_row_weights=wf[:nbase],matrix_anchor_current11_witness_child_rows=basecw,matrix_anchor_current11_witness_distance_m=basecd,
                matrix_anchor_route_mask_26=base26,matrix_anchor_current11_route_mask=(base26>>15).astype(np.uint16),matrix_anchor_global_route=data["ints"]["global_route"].astype(np.uint8),
                matrix_anchor_global_witness_child_row=data["ints"]["global_witness_child_row"].astype(np.int32),matrix_anchor_global_witness_distance_m=data["fl"]["global_witness_distance_m"])
            output={"path":str(op),"sha256":sha(op),"bytes":op.stat().st_size,"positive_adaptive_rows":len(selected)}
    cut_results=[]
    for j,name in enumerate(names26):
        cap=float(blend_rep["capacity_after_restore_and_blend_kg"][j]) if blend_rep else None
        sf=float(blend_rep["shortfall_after_restore_and_blend_kg"][j]) if blend_rep else None
        cut_results.append({"index":j,"name":name,"demand_kg":float(demands[j]),"capacity_kg":cap,"shortfall_kg":sf,"route_anchor_rows":int(np.count_nonzero((allm&(1<<j))!=0))})
    status="PASS_SOURCE_ONLY_ADAPTIVE_SLACK_PROPOSAL" if bool(res.success) and rep.get("passed_parent_moments") and rep.get("passed_slack_replay") and blend_rep is not None and blend_rep.get("passed_post_restore_blend_mass_moment_gate") and output is not None else "INCONCLUSIVE_SOURCE_ONLY_LP"
    report={"schema":"numi.h706.all-witness-adaptive-current11-joint-slack-lp-032d.v1","status":status,"physical_admission":False,"native_steps":0,"allocator_run":False,
        "objective":{"constraint_count":26,"names":names26,"demands_kg":demands.tolist(),"margin_kg":m.MARGIN,"minimized":"sum normalized deficits over 15 historical plus 11 current027 component targets"},
        "input_pins":{"031h_owner":{"path":str(BASE),"sha256":EXPECTED_BASE},"preparation_report":{"path":str(pr),"sha256":sha(pr)},
            "adaptive_sites":{"path":str(route_path),"sha256":sha(route_path)},"030_matrix":{"path":str(m.MATRIX),"sha256":sha(m.MATRIX)},
            "030_matrix_sha256":sha(m.MATRIX),"030_run_manifest_sha256":sha(m.MANIFEST),"030_execution_sha256":sha(m.EXEC),
            "expanded_typed_support_sha256":m.EXPECTED_SUPPORT,"current027_run_manifest_sha256":m.EXPECTED_CURRENT027_MANIFEST},
        "solver":{"method":"two-stage SciPy HiGHS sparse LP: minimum 26-cut deficit then mass-weighted L1 prior proximity","status":str(res.message),"status_code":int(res.status),"optimal":bool(res.success),"elapsed_seconds":elapsed,
            "primary_deficit_tolerance":LEX_DEFICIT_TOL,"lexicographic":rep.get("lexicographic"),
            "anchor_variables":len(allp),"base_matrix_rows":nbase,"adaptive_candidate_rows":len(cp),"positive_adaptive_rows":int(output["positive_adaptive_rows"]) if output else 0},
        "lp_replay_before_prior_preservation":rep,"prior_support_preservation":blend_rep,"cut_results":cut_results,"output":output,
        "limitations":["The LP only uses exact parent-child witness edges represented by the pinned 030 matrix; it is not the exhaustive parent-child graph.",
            "The 26 independent capacity inequalities do not enforce shared child flows and are only a source-side relaxation.",
            "Source-only proposal; the actual 1047 shared-capacity allocator remains the deciding flow measurement.",
            "No geometry, source mass, density bounds, material/donor/region policy, radius, or native physics changed."]}
    rp=out/"joint-lp-report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    exe={"status":status,"physical_admission":False,"native_steps":0,"allocator_run":False,"report":{"path":str(rp),"sha256":sha(rp)},"output":output,"elapsed_seconds":elapsed}
    ep=out/"execution.json";ep.write_text(json.dumps(exe,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":status,"report":str(rp),"report_sha256":sha(rp),"output":output,"elapsed_seconds":elapsed,"all26constraints_met_after_blend":None if blend_rep is None else blend_rep["all_26_constraints_met_after_restore_and_blend"]},sort_keys=True))
    return 0 if status.startswith("PASS") else 3


def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=("prepare","solve","self-test"),required=True);p.add_argument("--out-dir",type=Path);p.add_argument("--prepared-dir",type=Path);p.add_argument("--expected-preparation-sha256");p.add_argument("--time-limit-seconds",type=float,default=300.)
    a=p.parse_args()
    if a.mode=="self-test":self_test();return 0
    if a.out_dir is None:raise RuntimeError("prepare/solve requires --out-dir")
    out=a.out_dir.resolve()
    if a.mode=="prepare":return prepare(out)
    if a.prepared_dir is None or not a.expected_preparation_sha256:raise RuntimeError("solve requires --prepared-dir and --expected-preparation-sha256")
    return solve(a.prepared_dir,out,a.time_limit_seconds,a.expected_preparation_sha256)

if __name__=="__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse, csv, hashlib, importlib.util, json, math, os, struct
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
BASE_SRC = HERE / "current-source-carrier-bridge-1046.py"
OLD_RUN = HERE / "current-fractional-semantic-native-1047/actual-027-proof-slot-closure/run-manifest.json"
MATRIX_DEFAULT = HERE / "current-source-carrier-joint-initialization-1054/cut-closure-1054/actual-028-complete-predicate-joint-lp-001/run-001/anchor-cut-matrix.csv"
MATRIX_RUN_DEFAULT = HERE / "current-source-carrier-joint-initialization-1054/cut-closure-1054/actual-028-complete-predicate-joint-lp-001/run-001/run-manifest.json"
MATRIX_EXEC_DEFAULT = HERE / "current-source-carrier-joint-initialization-1054/cut-closure-1054/actual-028-complete-predicate-joint-lp-001/run-001/execution.json"
CLI_DEFAULT = HERE / "current-fractional-semantic-native-1047/optimization-trial-029-canonical-source-tet-sites/bin/current-fractional-mass-mapping-cli-029-o3"
WITNESS_FIELDS = ["record_kind","role","carrier_construction","packet_carrier_row","local_parent_row","global_source330_tet_row","child_local_row","child_global_row","material","donor","region","provenance_mode","p1_group_id","projection_csv_row","bary0","bary1","bary2","bary3","weight","source_x_m","source_y_m","source_z_m","child_centroid_x_m","child_centroid_y_m","child_centroid_z_m","distance_m"]
NEW_CONSTRUCTION = "source_tetra_anchor_refinement_1046d"
EDGE_CONSTRUCTION = "current_source_carrier_edge_support"
MODE = "CurrentCarrierPointProjection"
R = 0.019353

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def load_module(name,path):
    sp=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def file_pin(obj, expected=None):
    p=Path(obj["path"])
    got=sha(p)
    if expected and got!=expected: raise ValueError(f"input pin mismatch {p}: {got} != {expected}")
    if obj.get("sha256") and got!=obj["sha256"]: raise ValueError(f"manifest pin mismatch {p}")
    return got

def csv_iter(path):
    with open(path,newline="",encoding="utf-8") as f:
        yield from csv.DictReader(f)

def role_packet_row(global_flat_row, record_kind, material_packet_offset):
    raw=int(global_flat_row)
    if record_kind=="existing_packet_anchor":
        if raw<material_packet_offset: raise ValueError("global packet row precedes role offset")
        return raw-material_packet_offset
    return raw

def role_local_child(global_child_row, role_global_child_rows):
    child=int(global_child_row); rows=np.asarray(role_global_child_rows,dtype=np.int64)
    i=int(np.searchsorted(rows,child))
    return i if i<len(rows) and int(rows[i])==child else -1

def role_region_allows(ctx, source_region, global_child_row):
    local=role_local_child(global_child_row,ctx["cglobal"])
    return local>=0 and local in ctx["region_children"].get(int(source_region),set())

def require_finite_selected(values, mask, label):
    values=np.asarray(values,dtype=np.float64)
    mask=np.asarray(mask,dtype=bool)
    if values.shape!=mask.shape or not np.isfinite(values[mask]).all():
        raise ValueError(f"{label} contains nonfinite selected witness distances")

def blend_quadrature_weights(lp_base, lp_adaptive, prior, epsilon):
    lp_base=np.asarray(lp_base,dtype=np.float64)
    lp_adaptive=np.asarray(lp_adaptive,dtype=np.float64)
    prior=np.asarray(prior,dtype=np.float64)
    if not math.isfinite(epsilon) or not (0.0 <= epsilon < 1.0):
        raise ValueError("second baseline blend epsilon must be finite and in [0,1)")
    if not np.isfinite(lp_base).all() or not np.isfinite(lp_adaptive).all() or not np.isfinite(prior).all():
        raise ValueError("quadrature blend input contains nonfinite weights")
    if np.any(lp_base < -1e-14) or np.any(lp_adaptive < 0.0) or np.any(prior < 0.0):
        raise ValueError("quadrature blend input contains negative weights")
    return ((1.0-epsilon)*np.maximum(lp_base,0.0)+epsilon*prior,
            (1.0-epsilon)*lp_adaptive)

def validate_preblended_prior_support(weights, prior, prior_report):
    weights=np.asarray(weights,dtype=np.float64)
    prior=np.asarray(prior,dtype=np.float64)
    if not isinstance(prior_report,dict) or prior_report.get("passed_post_restore_blend_mass_moment_gate") is not True:
        raise ValueError("032d report does not pass its prior restoration/blend mass-moment gate")
    if prior_report.get("prior_global_route_positive_rows_checked") is not True:
        raise ValueError("032d report did not validate prior global routes")
    if not np.isfinite(weights).all() or np.any(weights<0) or not np.isfinite(prior).all() or np.any(prior<0):
        raise ValueError("preblended weights/prior are nonfinite or negative")
    prior_positive=prior>0.0
    missing=prior_positive & (weights<=0.0)
    if np.any(missing):
        raise ValueError(f"032d result dropped {int(np.count_nonzero(missing))} prior-positive matrix anchors")
    return int(np.count_nonzero(prior_positive))

def claim_witness_slot(seen, carrier, child):
    child=int(child)
    if child<0:
        return False
    key=(int(carrier),child)
    if key in seen:
        return False
    seen.add(key)
    return True

def validate_witness_arrays(mask, children, distances, count, bits, label):
    mask=np.asarray(mask,dtype=np.uint64)
    children=np.asarray(children,dtype=np.int64)
    distances=np.asarray(distances,dtype=np.float64)
    if mask.shape!=(count,) or children.shape!=(count,bits) or distances.shape!=(count,bits):
        raise ValueError(f"{label} witness/mask shape mismatch")
    expected=((mask[:,None]>>np.arange(bits,dtype=np.uint64))&1).astype(bool)
    if not np.array_equal(children>=0,expected):
        raise ValueError(f"{label} route mask and child witness rows disagree")
    require_finite_selected(distances,expected,label+" distances")
    return expected

def index_controls():
    assert role_packet_row(37436,"existing_packet_anchor",37436)==0
    assert role_packet_row(37440,"existing_packet_anchor",37436)==4
    assert role_packet_row(-1,"implicit_centroid",37436)==-1
    interleaved=np.asarray([0,2,5],np.int64)
    assert role_local_child(5,interleaved)==2 and role_local_child(2,interleaved)==1
    fake={"cglobal":interleaved,"region_children":{69:{2}}}
    assert role_region_allows(fake,69,5)
    assert not role_region_allows(fake,69,2)
    seen={(4,10)}
    assert not claim_witness_slot(seen,4,10)
    assert claim_witness_slot(seen,4,11)
    assert not claim_witness_slot(seen,4,11)
    assert claim_witness_slot(seen,4,12)
    prior=np.asarray([.25,.75],np.float64)
    gate={"passed_post_restore_blend_mass_moment_gate":True,"prior_global_route_positive_rows_checked":True}
    assert validate_preblended_prior_support(np.asarray([.7,.3]),prior,gate)==2
    try:
        validate_preblended_prior_support(np.asarray([.7,0.0]),prior,gate)
    except ValueError:
        pass
    else:
        raise AssertionError("missing prior-positive route was accepted")
    children=np.asarray([[3,-1],[7,9]],np.int64)
    distances=np.asarray([[.001,np.nan],[.002,.003]],np.float64)
    mask=np.asarray([1,3],np.uint16)
    validate_witness_arrays(mask,children,distances,2,2,"fixture")
    try:
        validate_witness_arrays(mask,children,np.asarray([[.001,np.nan],[.002,np.nan]],np.float64),2,2,"fixture")
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite selected current11 witness was accepted")
    return {"status":"PASS","global_packet_to_role_local":True,"implicit_minus_one_preserved":True,
            "interleaved_child_to_role_local":True,"child_membership_uses_role_local_row":True,
            "no_second_blend_prior_support_required":True,"duplicate_slots_deduplicated":True,
            "current11_mask_and_finite_distance_controls":True}

def read_sources(B):
    labels=B.csvread(B.TAB.with_name(B.TAB.name+".labels.csv"))
    tets=B.csvread(B.TAB.with_name(B.TAB.name+".tets.csv"))
    nodes=B.csvread(B.TAB.with_name(B.TAB.name+".nodes.csv"))
    massrows=B.csvread(B.TAB)
    xyz=np.asarray([[float(x["rest_"+a+"_m"]) for a in "xyz"] for x in nodes],np.float64)
    tnodes=np.asarray([[int(x["parent_node"+str(k)]) for k in range(4)] for x in tets],np.int64)
    if len(labels)!=len(tets) or len(massrows)!=len(tets): raise ValueError("source330 parent row domains differ")
    tup=np.asarray([[int(x[k]) for k in ("material_index","geometric_donor_body","anatomical_region_id")] for x in labels],np.int64)
    mass=np.full(len(labels),np.nan,np.float64)
    for r in massrows:
        g=int(r["source_tet_row"])
        if g<0 or g>=len(mass): raise ValueError("source mass parent row range")
        mass[g]=float(r["mass_kg"])
    if not np.isfinite(mass).all(): raise ValueError("source mass table has missing parent rows")
    return labels,tup,mass,tnodes,xyz,xyz[tnodes]

def load_candidate(B,manifest):
    path=B.CAND
    with np.load(path,allow_pickle=False) as z:
        p=z["points_reference_world_m_f64"].astype(np.float64,copy=True)
        tet=z["volume_tetrahedra"].astype(np.int64,copy=True)
        mat=z["material_index"].astype(np.int64,copy=True)
        don=z["geometric_donor_body"].astype(np.int64,copy=True)
        reg=z["anatomical_region_id"].astype(np.int64,copy=True)
    if not np.array_equal(p,p.astype(np.float32).astype(np.float64)):raise ValueError("candidate points are not exact widened F32")
    if tet.shape[1:]!=(4,) or len(mat)!=len(tet) or len(don)!=len(tet) or len(reg)!=len(tet):raise ValueError("candidate typed-volume shape")
    cc=p[tet].mean(axis=1)
    return {"path":path,"points":p,"tets":tet,"mat":mat,"don":don,"reg":reg,"centroids":cc}

def validate_child_mesh_inputs(candidate,named):
    nodepin=named["child_skin_node"];elepin=named["child_skin_elements"];labelpin=named["child_skin_labels"]
    file_pin(nodepin);file_pin(elepin);file_pin(labelpin)
    with open(nodepin["path"],encoding="utf-8") as f:
        head=f.readline().split()
        if len(head)<4 or int(head[0])!=len(candidate["points"]) or int(head[1])!=3: raise ValueError("child node header/domain")
        node_ids=np.empty(len(candidate["points"]),np.int64);node_xyz=np.empty_like(candidate["points"])
        for i,line in enumerate(f):
            if i>=len(node_ids): raise ValueError("child node row overflow")
            q=line.split()
            if len(q)<4: raise ValueError("child node row width")
            node_ids[i]=int(q[0]);node_xyz[i]=[float(q[1]),float(q[2]),float(q[3])]
        if i+1!=len(node_ids): raise ValueError("child node row count")
        if not np.array_equal(node_ids,np.arange(1,len(node_ids)+1)): raise ValueError("child node IDs are not contiguous one-based row IDs")
        if not np.array_equal(node_xyz,candidate["points"]): raise ValueError("typed candidate points differ from exact native child .node")
    with open(elepin["path"],encoding="utf-8") as f:
        head=f.readline().split()
        if len(head)<3 or int(head[0])!=len(candidate["tets"]) or int(head[1])!=4 or int(head[2])<1: raise ValueError("child element header/domain")
        ne=len(candidate["tets"]);cell_ids=np.empty(ne,np.int64);raw_conn=np.empty((ne,4),np.int64);attrs=np.empty((ne,int(head[2])),np.float64)
        for i,line in enumerate(f):
            if i>=ne: raise ValueError("child element row overflow")
            q=line.split()
            if len(q)<5+int(head[2]): raise ValueError("child element row width")
            cell_ids[i]=int(q[0]);raw_conn[i]=[int(v) for v in q[1:5]];attrs[i]=[float(v) for v in q[5:5+int(head[2])]]
        if i+1!=ne or not np.array_equal(cell_ids,np.arange(1,ne+1)): raise ValueError("child element IDs/row count")
        if np.array_equal(raw_conn-1,candidate["tets"]): connbase=1
        elif np.array_equal(raw_conn,candidate["tets"]): connbase=0
        else: raise ValueError("native child .ele connectivity differs from typed candidate")
        if not np.array_equal(attrs[:,0].astype(np.int64),candidate["mat"]): raise ValueError("native child .ele material differs from typed candidate")
    labels=[]
    with open(labelpin["path"],newline="",encoding="utf-8") as f:
        rd=csv.DictReader(f)
        for i,r in enumerate(rd):
            if int(r["child_tet_row"])!=i: raise ValueError("child labels row IDs are not contiguous")
            labels.append((int(r["material_index"]),int(r["geometric_donor_body"]),int(r["anatomical_region_id"])))
    lab=np.asarray(labels,np.int64)
    if not np.array_equal(lab[:,0],candidate["mat"]) or not np.array_equal(lab[:,1],candidate["don"]) or not np.array_equal(lab[:,2],candidate["reg"]):
        raise ValueError("native child labels differ from typed candidate")
    return {"node_sha256":nodepin["sha256"],"elements_sha256":elepin["sha256"],"labels_sha256":labelpin["sha256"],
            "node_rows":len(node_ids),"element_rows":len(cell_ids),"element_connectivity_base":connbase,"exact_typed_candidate_match":True}

def load_role_context(role,manifest,candidate,tups,source_mass,src_xyz,src_tet_nodes):
    mat=15 if role=="Skin" else 16
    nm=manifest["named_input_pins"]
    p1pin=nm["p1_skin_support" if role=="Skin" else "p1_fat_support"]
    file_pin(p1pin)
    with np.load(p1pin["path"],allow_pickle=False) as z:
        ro=z["cell_admissible_region_offsets"].astype(np.int64)
        rid=z["cell_admissible_region_ids"].astype(np.int64)
        maxhop=int(z["admissibility_max_source_region_hops"])
    if maxhop!=1:raise ValueError("expected exact P1 one-hop support")
    cglobal=np.flatnonzero(candidate["mat"]==mat).astype(np.int64)
    if len(cglobal)!=len(ro)-1:raise ValueError(f"{role} candidate/P1 child domain mismatch")
    local_by_region=defaultdict(set)
    for c in range(len(cglobal)):
        a,b=ro[c:c+2]
        for r in rid[a:b]: local_by_region[int(r)].add(c)
    global_to_local={int(g):i for i,g in enumerate(cglobal)}
    grows=np.flatnonzero(tups[:,0]==mat).astype(np.int64)
    global_parent_to_local={int(g):i for i,g in enumerate(grows)}
    if len(grows)!=14316:raise ValueError(f"{role} source parent count changed: {len(grows)}")
    packet_pin=nm["carrier_skin_packet" if role=="Skin" else "carrier_fat_packet"]
    witness_pin=nm["carrier_witness_skin" if role=="Skin" else "carrier_witness_fat"]
    file_pin(packet_pin);file_pin(witness_pin)
    with np.load(packet_pin["path"],allow_pickle=False) as z:
        old={k:z[k].copy() for k in z.files}
    if not np.array_equal(old["global_source330_tet_rows"],grows.astype(np.uint32)):
        raise ValueError(f"{role} legacy packet source map mismatch")
    if not np.array_equal(old["source_parent_tet_rows"],np.arange(len(grows),dtype=np.uint32)):
        raise ValueError(f"{role} legacy packet local parent IDs mismatch")
    oldoff=old["parent_offsets"].astype(np.int64)
    oldw=old["carrier_weights"].astype(np.float64)
    oldb=old["carrier_barycentric"].astype(np.float64)
    if oldoff.shape!=(len(grows)+1,) or oldoff[-1]!=len(oldw) or oldb.shape!=(len(oldw),4):raise ValueError("legacy packet CSR shape")
    return {"role":role,"mat":mat,"grows":grows,"pmap":global_parent_to_local,"cglobal":cglobal,
            "cmap":global_to_local,"region_children":local_by_region,"p1_regions":(ro,rid),
            "packet":old,"oldoff":oldoff,"oldw":oldw,"oldb":oldb,
            "old_packet_path":Path(packet_pin["path"]),"old_witness_path":Path(witness_pin["path"]),
            "typed_candidate_path":candidate["path"],
            "old_packet_sha":packet_pin["sha256"],"old_witness_sha":witness_pin["sha256"]}

def matrix_load(path,n,expected_sha):
    if sha(path)!=expected_sha:raise ValueError("031 matrix hash does not match its report")
    ints={k:np.empty(n,np.int64) for k in ("anchor_index","source_parent_row","parent_input_row","packet_carrier_row","material","donor","region","route_mask","global_route","global_witness_child_row")}
    fl={k:np.empty(n,np.float64) for k in ("parent_mass_kg","prior_weight","global_witness_distance_m")}
    bary=np.empty((n,4),np.float64);source=np.empty((n,3),np.float64)
    cuts=np.full((n,15),-1,np.int64);cutd=np.full((n,15),np.nan,np.float64)
    kinds=[]
    with open(path,newline="",encoding="utf-8") as f:
        rd=csv.DictReader(f)
        i=0
        for row in rd:
            if i>=n:raise ValueError("matrix row overflow")
            for k in ints: ints[k][i]=int(row[k] or -1)
            for k in fl: fl[k][i]=float(row[k]) if row[k] else math.nan
            bary[i]=[float(row[f"bary{k}"]) for k in range(4)]
            source[i]=[float(row[f"source_{a}_m"]) for a in "xyz"]
            for j in range(15):
                v=row[f"cut{j:02d}_witness_child_row"];d=row[f"cut{j:02d}_witness_distance_m"]
                cuts[i,j]=int(v) if v else -1
                cutd[i,j]=float(d) if d else math.nan
            kinds.append(row["record_kind"]);i+=1
    if i!=n:raise ValueError(f"matrix rows {i} != LP rows {n}")
    return ints,fl,bary,source,cuts,cutd,np.asarray(kinds,dtype=object)

def make_row(role,source_g,bary,weight,primary_child,primary_dist,mask,cuts,cutds,
             current11_mask,current11_children,current11_distances,origin,anchor_index,
             old_packet_row,matrix_kind=""):
    return {"role":role,"g":int(source_g),"bary":np.asarray(bary,dtype=np.float64),
            "weight":float(weight),"primary":int(primary_child),"primary_d":float(primary_dist),
            "mask":int(mask),"cuts":np.asarray(cuts,dtype=np.int64),"cutds":np.asarray(cutds,dtype=np.float64),
            "current11_mask":int(current11_mask),"current11":np.asarray(current11_children,dtype=np.int64),
            "current11d":np.asarray(current11_distances,dtype=np.float64),
            "origin":origin,"anchor_index":int(anchor_index),"old_packet_row":int(old_packet_row),"matrix_kind":matrix_kind}

def validate_routes(records,ctx,candidate,src_tet_xyz,tups):
    children=[];parents=[];barys=[];claims=[];distclaims=[];labels=[]
    for r in records:
        g=r["g"];b=r["bary"];mat=ctx["mat"]
        if tups[g,0]!=mat:raise ValueError("proposal parent material mismatch")
        p=b@src_tet_xyz[g]
        if r["primary"]<0:raise ValueError("positive carrier has no global primary witness")
        children.append(r["primary"]);parents.append(g);barys.append(b);claims.append(p);distclaims.append(r["primary_d"]);labels.append(("primary",r))
        for j,ch in enumerate(r["cuts"]):
            bit=(r["mask"]>>j)&1
            if bit != int(ch>=0): raise ValueError(f"route mask/witness disagreement parent={g} cut={j}")
            if bit:
                children.append(int(ch));parents.append(g);barys.append(b);claims.append(p);distclaims.append(float(r["cutds"][j]));labels.append((f"cut{j}",r))
        if r["current11"].shape!=(11,) or r["current11d"].shape!=(11,):
            raise ValueError("current11 witness carrier shape")
        for j,ch in enumerate(r["current11"]):
            bit=(r["current11_mask"]>>j)&1
            if bit != int(ch>=0): raise ValueError(f"current11 route mask/witness disagreement parent={g} cut={j}")
            if bit:
                children.append(int(ch));parents.append(g);barys.append(b);claims.append(p);distclaims.append(float(r["current11d"][j]));labels.append((f"current11_{j}",r))
    children=np.asarray(children,np.int64);parents=np.asarray(parents,np.int64);barys=np.asarray(barys,np.float64)
    if np.any(children<0) or np.any(children>=len(candidate["tets"])):raise ValueError("witness child row range")
    sourcexyz=np.einsum("ni,nij->nj",barys,src_tet_xyz[parents])
    childxyz=candidate["centroids"][children]
    d=np.linalg.norm(sourcexyz-childxyz,axis=1)
    require_finite_selected(np.asarray(distclaims,np.float64),np.ones(len(distclaims),bool),"replayed route claim")
    if not np.isfinite(d).all() or np.max(np.abs(d-np.asarray(distclaims)),initial=0)>2e-12 or np.any(d>R+1e-12):
        raise ValueError("witness distance/radius replay failure")
    if np.any(candidate["mat"][children]!=ctx["mat"]):raise ValueError("witness crosses child material")
    locals_=np.asarray([role_local_child(c,ctx["cglobal"]) for c in children],np.int64)
    if np.any(locals_<0):raise ValueError("witness role-local child map")
    for ix,(g,childlocal) in enumerate(zip(parents,locals_)):
        region=int(tups[g,2])
        if int(childlocal) not in ctx["region_children"].get(region,set()):
            kind=labels[ix][0]
            raise ValueError(f"{kind} source tuple is not in child one-hop region parent={g} child={int(children[ix])}")
    return sourcexyz,locals_

def write_role(role,outdir,records,ctx,candidate,tups,src_tet_xyz,old_side_path):
    mat=ctx["mat"];grows=ctx["grows"];pmap=ctx["pmap"]
    records.sort(key=lambda r:(pmap[r["g"]],r["anchor_index"],r["origin"]))
    n=len(grows)
    parent_counts=np.zeros(n,np.int64)
    for r in records: parent_counts[pmap[r["g"]]]+=1
    if np.any(parent_counts==0):raise ValueError(f"{role} source parents with no positive proposal carriers: {np.count_nonzero(parent_counts==0)}")
    offsets=np.r_[0,np.cumsum(parent_counts)].astype("<u4")
    w=np.asarray([r["weight"] for r in records],dtype="<f8")
    b=np.asarray([r["bary"] for r in records],dtype="<f8").reshape(-1,4)
    primary=np.asarray([r["primary"] for r in records],np.int64)
    cutchild=np.asarray([r["cuts"] for r in records],np.int64).reshape(-1,15)
    cutdist=np.asarray([r["cutds"] for r in records],np.float64).reshape(-1,15)
    current11child=np.asarray([r["current11"] for r in records],np.int64).reshape(-1,11)
    current11dist=np.asarray([r["current11d"] for r in records],np.float64).reshape(-1,11)
    if not np.isfinite(w).all() or np.any(w<=0) or not np.isfinite(b).all() or np.any(b<0) or np.max(np.abs(b.sum(axis=1)-1))>2e-12:
        raise ValueError(f"{role} positive barycentric data invalid")
    sums=np.zeros(n,np.float64);mom=np.zeros((n,4),np.float64)
    loc=np.asarray([pmap[r["g"]] for r in records],np.int64)
    np.add.at(sums,loc,w);np.add.at(mom,loc,w[:,None]*b)
    if np.max(np.abs(sums-1),initial=0)>2e-10 or np.max(np.abs(mom-.25),initial=0)>2e-10:
        raise ValueError(f"{role} parent mass/moment LP replay maxsum={np.max(np.abs(sums-1))} maxmom={np.max(np.abs(mom-.25))}")
    sourcexyz=np.einsum("ni,nij->nj",b,np.asarray([src_tet_xyz[r["g"]] for r in records]))
    childxyz=candidate["centroids"][primary]
    primaryd=np.linalg.norm(sourcexyz-childxyz,axis=1)
    if np.any(primaryd>R+1e-12):raise ValueError(f"{role} primary exceeds unchanged radius")
    # Record indices for old packet primaries/slots whose points remain positive.
    old_to_new={}
    centroid_to_new={}
    matrix_row_to_new={}
    for ix,r in enumerate(records):
        if r["origin"]=="matrix":
            matrix_row_to_new[r["anchor_index"]]=ix
            if r["old_packet_row"]>=0:
                if r["old_packet_row"] in old_to_new:raise ValueError("duplicate surviving old packet row")
                old_to_new[r["old_packet_row"]]=ix
            elif r["origin"]=="matrix" and r["matrix_kind"]=="implicit_centroid":
                if r["g"] in centroid_to_new:raise ValueError("duplicate surviving implicit centroid row")
                centroid_to_new[r["g"]]=ix
    arr={"source_parent_tet_rows":np.arange(n,dtype="<u4"),"global_source330_tet_rows":grows.astype("<u4"),
        "parent_offsets":offsets,"carrier_weights":w,"carrier_barycentric":b,
        "carrier_target_child_rows":np.asarray([ctx["cmap"][int(c)] for c in primary],dtype="<u4"),
        "carrier_target_distance_m":primaryd.astype("<f8")}
    stem=role.lower()+"-adaptive-source-quadrature-032d"
    npz=outdir/(stem+".npz");binp=outdir/(stem+".bin");sidepath=outdir/(role.lower()+"-adaptive-witnesses-032d.csv")
    np.savez_compressed(npz,**arr)
    B=load_module("base_packet_1046",BASE_SRC)
    binp.write_bytes(B.packet(arr))
    loaded,bsha=B.verify_packet(binp,npz,grows.astype("<u4"))
    # Each carrier needs exactly one primary sidecar record.
    seen=set()
    with open(sidepath,"w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=WITNESS_FIELDS);wr.writeheader()
        for ix,r in enumerate(records):
            g=r["g"];tup=tups[g];point=sourcexyz[ix];child=int(primary[ix]);cl=int(ctx["cmap"][child]);d=float(primaryd[ix])
            rec={"record_kind":"carrier","role":role,"carrier_construction":NEW_CONSTRUCTION,"packet_carrier_row":ix,
                 "local_parent_row":pmap[g],"global_source330_tet_row":g,"child_local_row":cl,"child_global_row":child,
                 "material":int(tup[0]),"donor":int(tup[1]),"region":int(tup[2]),"provenance_mode":MODE,
                 "p1_group_id":-1,"projection_csv_row":-1,"weight":float(w[ix]),"source":point,"child":candidate["centroids"][child],
                 "distance":d,"bary":b[ix]}
            wr.writerow(format_row(rec));seen.add((ix,child))
        # Preserve every old primary/slot route for a surviving identical point, remapping it
        # as an exact current-point support slot under the general 1046d mode.
        inherited=0;dropped=0
        for row in csv_iter(old_side_path):
            kind=row["record_kind"]
            if kind not in ("carrier","carrier_child_slot","centroid_route_fallback"):continue
            old_ix=int(row["packet_carrier_row"])
            if old_ix>=0:
                ix=old_to_new.get(old_ix)
                if ix is None:dropped+=1;continue
            else:
                # Implicit centroids map only to an exact positive LP centroid anchor.
                g=int(row["global_source330_tet_row"]);ix=centroid_to_new.get(g)
                if ix is None:dropped+=1;continue
            g=records[ix]["g"];child=int(row["child_global_row"])
            if int(row["global_source330_tet_row"])!=g:raise ValueError("inherited witness parent mismatch")
            if child<0 or child>=len(candidate["tets"]):raise ValueError("inherited witness child range")
            point=b[ix]@src_tet_xyz[g];dst=float(np.linalg.norm(point-candidate["centroids"][child]))
            if candidate["mat"][child]!=mat or dst>R+1e-12 or int(ctx["cmap"][child]) not in ctx["region_children"].get(int(tups[g,2]),set()):
                raise ValueError("inherited witness no longer passes exact material/one-hop/radius replay")
            if not claim_witness_slot(seen,ix,child):continue
            rec={"record_kind":"carrier_child_slot","role":role,"carrier_construction":EDGE_CONSTRUCTION,"packet_carrier_row":ix,
                "local_parent_row":pmap[g],"global_source330_tet_row":g,"child_local_row":ctx["cmap"][child],"child_global_row":child,
                "material":int(tups[g,0]),"donor":int(tups[g,1]),"region":int(tups[g,2]),"provenance_mode":MODE,
                "p1_group_id":-1,"projection_csv_row":-1,"weight":float(w[ix]),"source":point,
                "child":candidate["centroids"][child],"distance":dst,"bary":b[ix]}
            wr.writerow(format_row(rec));inherited+=1
        # Add exact historical and current11 witnesses from the source-only LP.
        cut_slots=0
        current11_slots=0
        for ix,r in enumerate(records):
            g=r["g"];point=sourcexyz[ix]
            for j,child0 in enumerate(cutchild[ix]):
                child=int(child0)
                if child<0:continue
                if not claim_witness_slot(seen,ix,child):continue
                dst=float(np.linalg.norm(point-candidate["centroids"][child]))
                if abs(dst-float(cutdist[ix,j]))>2e-12 or dst>R+1e-12 or candidate["mat"][child]!=mat:
                    raise ValueError(f"cut witness replay failure carrier={ix} cut={j}")
                if not role_region_allows(ctx,int(tups[g,2]),child):
                    raise ValueError(f"cut witness one-hop failure carrier={ix} cut={j}")
                rec={"record_kind":"carrier_child_slot","role":role,"carrier_construction":EDGE_CONSTRUCTION,"packet_carrier_row":ix,
                    "local_parent_row":pmap[g],"global_source330_tet_row":g,"child_local_row":ctx["cmap"][child],"child_global_row":child,
                    "material":int(tups[g,0]),"donor":int(tups[g,1]),"region":int(tups[g,2]),"provenance_mode":MODE,
                    "p1_group_id":-1,"projection_csv_row":-1,"weight":float(w[ix]),"source":point,
                    "child":candidate["centroids"][child],"distance":dst,"bary":b[ix]}
                wr.writerow(format_row(rec));cut_slots+=1
            for j,child0 in enumerate(current11child[ix]):
                child=int(child0)
                if child<0:continue
                if not claim_witness_slot(seen,ix,child):continue
                dst=float(np.linalg.norm(point-candidate["centroids"][child]))
                if abs(dst-float(current11dist[ix,j]))>2e-12 or dst>R+1e-12 or candidate["mat"][child]!=mat:
                    raise ValueError(f"current11 witness replay failure carrier={ix} cut={j}")
                if not role_region_allows(ctx,int(tups[g,2]),child):
                    raise ValueError(f"current11 witness one-hop failure carrier={ix} cut={j}")
                rec={"record_kind":"carrier_child_slot","role":role,"carrier_construction":EDGE_CONSTRUCTION,"packet_carrier_row":ix,
                    "local_parent_row":pmap[g],"global_source330_tet_row":g,"child_local_row":ctx["cmap"][child],"child_global_row":child,
                    "material":int(tups[g,0]),"donor":int(tups[g,1]),"region":int(tups[g,2]),"provenance_mode":MODE,
                    "p1_group_id":-1,"projection_csv_row":-1,"weight":float(w[ix]),"source":point,
                    "child":candidate["centroids"][child],"distance":dst,"bary":b[ix]}
                wr.writerow(format_row(rec));current11_slots+=1
    receipt={"schema":"numi.h706.adaptive-source-anchor-packet-rebind.034.v1","status":"PASS_SOURCE_ONLY_ADAPTIVE_PACKET_READBACK",
        "role":role,"material_index":mat,"native_steps":0,"physical_admission":False,"source_parent_count":n,
        "carrier_count":len(records),"carrierized_parent_count":int(np.count_nonzero(parent_counts)),
        "inherited_route_slots":inherited,"historical_cut_route_slots":cut_slots,"current11_route_slots":current11_slots,
        "dropped_old_routes_for_zero_weight_points":dropped,
        "parent_weight_sum_error_max":float(np.max(np.abs(sums-1),initial=0)),
        "parent_first_moment_error_max":float(np.max(np.abs(mom-.25),initial=0)),
        "max_primary_distance_m":float(primaryd.max(initial=0)),"primary_rows_replayed":len(records),"support_pairs_replayed":len(seen),
        "inputs":{"031_proposal_parent_rows":len(records),"matrix_sha256":sha(MATRIX_PATH),"proposal_sha256":sha(PROPOSAL_PATH),
          "proposal_report_sha256":sha(PROPOSAL_REPORT),"source_run_manifest_sha256":sha(OLD_RUN),
          "old_packet_sha256":ctx["old_packet_sha"],"old_witness_sha256":ctx["old_witness_sha"],
          "typed_candidate_sha256":sha(ctx["typed_candidate_path"])},
        "outputs":{"npz":{"path":str(npz),"sha256":sha(npz),"bytes":npz.stat().st_size},
          "binary":{"path":str(binp),"sha256":bsha,"bytes":binp.stat().st_size},
          "witness_csv":{"path":str(sidepath),"sha256":sha(sidepath),"bytes":sidepath.stat().st_size}},
        "construction":"source_tetra_anchor_refinement_1046d","provenance_mode":"CurrentCarrierPointProjection",
        "witness_rule":"Each primary, historical-cut, and current11 witness independently replays exact source330 tet+bary point, typed parent tuple, same-material current child, exact authenticated one-hop source-region membership and unchanged 19.353mm radius.",
        "native_admission":"not claimed"}
    recpath=outdir/(role.lower()+"-adaptive-packet-receipt.json");recpath.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    verify={"schema":"numi.h706.adaptive-source-anchor-packet-rebind.034.readback.v1","status":"PASS_SOURCE_ONLY_EXACT_PACKET_WITNESS_READBACK",
        "role":role,"receipt_sha256":sha(recpath),"packet_npz_sha256":sha(npz),"packet_binary_sha256":bsha,
        "witness_sidecar_sha256":sha(sidepath),"primary_rows_replayed":len(records),"support_pairs_replayed":len(seen),
        "parent_moments":"PASS","typed_tuple_one_hop_radius":"PASS","native_steps":0,"physical_admission":False}
    vpath=outdir/(role.lower()+"-adaptive-packet-verification.json");vpath.write_text(json.dumps(verify,sort_keys=True,indent=2)+"\n")
    return receipt,verify

def format_row(r):
    out={k:"" for k in WITNESS_FIELDS}
    for k in ("record_kind","role","carrier_construction","packet_carrier_row","local_parent_row","global_source330_tet_row",
              "child_local_row","child_global_row","material","donor","region","provenance_mode","p1_group_id","projection_csv_row"):
        out[k]=r[k]
    for j in range(4):out[f"bary{j}"]=float(r["bary"][j])
    out["weight"]=float(r["weight"])
    for j,a in enumerate("xyz"):
        out[f"source_{a}_m"]=float(r["source"][j]);out[f"child_centroid_{a}_m"]=float(r["child"][j])
    out["distance_m"]=float(r["distance"])
    return out

def build(args):
    global MATRIX_PATH,PROPOSAL_PATH,PROPOSAL_REPORT,MATRIX_RUN_PATH,MATRIX_EXEC_PATH
    MATRIX_PATH=Path(args.matrix);PROPOSAL_PATH=Path(args.proposal);PROPOSAL_REPORT=Path(args.proposal_report)
    MATRIX_RUN_PATH=Path(args.matrix_run_manifest);MATRIX_EXEC_PATH=Path(args.matrix_execution)
    B=load_module("base_source_1046",BASE_SRC)
    H1046=load_module("carrier_h_1046",HERE/"current_source_carrier_bridge_1046h.py")
    oldman=json.loads(Path(args.source_run_manifest).read_text())
    named=oldman["named_input_pins"]
    for key in ("source330_mass","source330_labels","source330_nodes","source330_tets","child_skin_node","child_skin_elements","child_skin_labels",
                "p1_skin_support","p1_fat_support","carrier_skin_packet","carrier_fat_packet","carrier_witness_skin","carrier_witness_fat"):
        if key not in named:raise ValueError("source run manifest missing "+key)
        file_pin(named[key])
    proposal_report=json.loads(PROPOSAL_REPORT.read_text())
    if proposal_report.get("status")!="PASS_SOURCE_ONLY_ADAPTIVE_SLACK_PROPOSAL" or proposal_report.get("physical_admission") is not False or proposal_report.get("native_steps")!=0 or proposal_report.get("allocator_run") is not False:
        raise ValueError("031 proposal is not a source-only pass")
    outpin=proposal_report.get("output")
    if not outpin or Path(outpin["path"])!=PROPOSAL_PATH or file_pin(outpin)!=outpin["sha256"]:
        raise ValueError("031 proposal output pin")
    inp=proposal_report.get("input_pins",{})
    matrix_sha=inp.get("030_matrix_sha256")
    if not matrix_sha or sha(MATRIX_PATH)!=matrix_sha:raise ValueError("031 matrix pin")
    if inp.get("030_run_manifest_sha256")!=sha(MATRIX_RUN_PATH) or inp.get("030_execution_sha256")!=sha(MATRIX_EXEC_PATH):
        raise ValueError("031 proposal does not bind the exact 030 run manifest and execution receipt")
    matrix_run=json.loads(MATRIX_RUN_PATH.read_text());matrix_execution=json.loads(MATRIX_EXEC_PATH.read_text())
    if matrix_run.get("status")!="RUNNING" or matrix_run.get("physical_admission") is not False or matrix_run.get("native_steps")!=0:
        raise ValueError("030 source-only run manifest state")
    if matrix_execution.get("status")!="PASS_SOURCE_ONLY_MATRIX" or matrix_execution.get("exit_code")!=0 or matrix_execution.get("physical_admission") is not False or matrix_execution.get("native_steps")!=0:
        raise ValueError("030 source-only execution receipt state")
    if matrix_execution.get("run_manifest",{}).get("path")!=str(MATRIX_RUN_PATH) or matrix_execution.get("run_manifest",{}).get("sha256")!=sha(MATRIX_RUN_PATH):
        raise ValueError("030 execution-to-manifest binding")
    with np.load(PROPOSAL_PATH,allow_pickle=False) as z: prop={k:z[k].copy() for k in z.files}
    basew=prop["matrix_anchor_row_weights"].astype(np.float64)
    n=len(basew)
    ints,fl,baseb,basexyz,cutchild,cutdist,kinds=matrix_load(MATRIX_PATH,n,matrix_sha)
    if not np.array_equal(ints["anchor_index"],np.arange(n)):raise ValueError("matrix anchor index order")
    if not np.array_equal(ints["source_parent_row"],ints["parent_input_row"]):raise ValueError("matrix parent global keys disagree")
    if not np.array_equal(prop["matrix_anchor_global_route"].astype(np.int64),ints["global_route"]):raise ValueError("proposal/base global route mismatch")
    if not np.array_equal(prop["matrix_anchor_global_witness_child_row"].astype(np.int64),ints["global_witness_child_row"]):raise ValueError("proposal/base global witness mismatch")
    base11child=prop["matrix_anchor_current11_witness_child_rows"].astype(np.int64)
    base11dist=prop["matrix_anchor_current11_witness_distance_m"].astype(np.float64)
    base11mask=prop["matrix_anchor_current11_route_mask"].astype(np.uint64)
    base26=prop["matrix_anchor_route_mask_26"].astype(np.uint64)
    if not np.array_equal((base26&((1<<15)-1)).astype(np.int64),ints["route_mask"]):
        raise ValueError("proposal/base historical 15-bit route-mask mismatch")
    if not np.array_equal(((base26>>15)&((1<<11)-1)).astype(np.uint64),base11mask):
        raise ValueError("proposal/base current11 route-mask mismatch")
    validate_witness_arrays(ints["route_mask"],cutchild,cutdist,n,15,"matrix historical")
    validate_witness_arrays(base11mask,base11child,base11dist,n,11,"matrix current11")
    global_route=ints["global_route"]==1
    require_finite_selected(fl["global_witness_distance_m"],global_route,"matrix global witness")
    require_finite_selected(prop["matrix_anchor_global_witness_distance_m"],global_route,"proposal matrix global witness")
    if np.max(np.abs(prop["matrix_anchor_global_witness_distance_m"][global_route]-fl["global_witness_distance_m"][global_route]),initial=0)>1e-15:
        raise ValueError("proposal/base primary distance mismatch")
    for j in range(15):
        cut_route=(ints["route_mask"]&(1<<j))!=0
        require_finite_selected(cutdist[:,j],cut_route,f"matrix cut {j} witness")
    if np.any(~np.isfinite(basew)) or np.any(basew < -1e-14):raise ValueError("LP returned invalid base weights")
    basew=np.maximum(basew,0.0)
    if np.any((basew>0)&(ints["global_route"]==0)):raise ValueError("LP assigns positive mass to global-route-zero matrix anchor")
    epsilon=float(args.baseline_route_epsilon)
    if epsilon!=0.0:raise ValueError("032d already performs its declared prior blend; bridge must not blend a second time")
    priorw=fl["prior_weight"].copy()
    if not np.isfinite(priorw).all() or np.any(priorw < -1e-14):raise ValueError("030 prior quadrature weights are invalid")
    priorw=np.maximum(priorw,0.0)
    prior_positive=priorw>0.0
    if np.any(prior_positive & ~global_route):raise ValueError("original positive anchor lacks an exact global route")
    require_finite_selected(fl["global_witness_distance_m"],prior_positive,"original positive anchor global witness")
    prior_gate=proposal_report.get("prior_support_preservation",{})
    prior_positive_count=validate_preblended_prior_support(basew,priorw,prior_gate)
    ad_w=prop["weight"].astype(np.float64);ad_g=prop["source_parent_row"].astype(np.int64);ad_b=prop["barycentric"].astype(np.float64)
    ad_route=prop["global_route"].astype(np.int64);ad_child=prop["global_witness_child_row"].astype(np.int64);ad_dist=prop["global_witness_distance_m"].astype(np.float64)
    ad_mask=prop["route_mask"].astype(np.uint64);ad_cuts=prop["cut_witness_child_rows"].astype(np.int64);ad_cutd=prop["cut_witness_distance_m"].astype(np.float64)
    ad11mask=prop["current11_route_mask"].astype(np.uint64)
    ad11child=prop["current11_witness_child_rows"].astype(np.int64)
    ad11dist=prop["current11_witness_distance_m"].astype(np.float64)
    ad26=prop["route_mask_26"].astype(np.uint64)
    if not np.array_equal((ad26&((1<<15)-1)).astype(np.uint64),ad_mask):raise ValueError("adaptive historical mask disagrees with route_mask_26")
    if not np.array_equal(((ad26>>15)&((1<<11)-1)).astype(np.uint64),ad11mask):raise ValueError("adaptive current11 mask disagrees with route_mask_26")
    validate_witness_arrays(ad_mask,ad_cuts,ad_cutd,len(ad_w),15,"adaptive historical")
    validate_witness_arrays(ad11mask,ad11child,ad11dist,len(ad_w),11,"adaptive current11")
    if not (len(ad_w)==len(ad_g)==len(ad_b)==len(ad_route)==len(ad_child)==len(ad_dist)==len(ad_mask)==len(ad_cuts)==len(ad_cutd)==len(ad11mask)==len(ad11child)==len(ad11dist)==len(ad26)):
        raise ValueError("adaptive proposal array length mismatch")
    if np.any(ad_w<=0) or not np.isfinite(ad_w).all() or np.any(ad_route!=1) or np.any(ad_g<0) or np.any(ad_b<0) or not np.isfinite(ad_b).all() or np.max(np.abs(ad_b.sum(axis=1)-1),initial=0)>2e-12:
        raise ValueError("adaptive positive proposal row validity")
    require_finite_selected(ad_dist,np.ones(len(ad_dist),bool),"adaptive global witness")
    if ad_cuts.ndim!=2 or ad_cuts.shape[1]!=15:raise ValueError("adaptive cut witness shape")
    for j in range(15):
        require_finite_selected(ad_cutd[:,j],ad_cuts[:,j]>=0,f"adaptive cut {j} witness")
    labels,tups,mass,tnodes,srcxyz,src_tet_xyz=read_sources(B)
    candidate=load_candidate(B,oldman);candidate["typed_candidate_path"]=B.CAND
    child_mesh_check=validate_child_mesh_inputs(candidate,named)
    if len(tups)!=len(mass) or len(tups)!=len(src_tet_xyz):raise ValueError("source330 arrays mismatch")
    for i in range(n):
        g=int(ints["source_parent_row"][i])
        if g<0 or g>=len(tups):raise ValueError("matrix parent outside source330 domain")
        if tuple(ints[k][i] for k in ("material","donor","region"))!=tuple(tups[g]):raise ValueError("matrix exact parent tuple mismatch")
        if abs(float(fl["parent_mass_kg"][i])-mass[g])>1e-15:raise ValueError("matrix source parent mass mismatch")
        if np.max(np.abs(baseb[i]@src_tet_xyz[g]-basexyz[i]))>2e-12:raise ValueError("matrix source bary point mismatch")
    if np.any(ad_g>=len(tups)):raise ValueError("adaptive source parent row outside source330")
    if np.any(~np.isin(tups[ad_g,0],[15,16])):raise ValueError("adaptive source tuple material invalid")
    if np.max(np.abs(np.einsum("ni,nij->nj",ad_b,src_tet_xyz[ad_g])-prop["source_point_world_m"]),initial=0)>2e-12:raise ValueError("adaptive source point bary replay")
    candcuts=ad_cuts
    if candcuts.shape!=(len(ad_w),15) or ad_cutd.shape!=(len(ad_w),15):raise ValueError("adaptive cut witness shape")
    # Both quadratures close per-parent unit mass and quarter barycentric moment.
    # Their convex mixture restores every positive old anchor without changing
    # either constraint. This is a bounded support-retention sensitivity.
    prior_sum=np.zeros(len(tups),np.float64);prior_moment=np.zeros((len(tups),4),np.float64)
    np.add.at(prior_sum,ints["source_parent_row"],priorw)
    np.add.at(prior_moment,ints["source_parent_row"],priorw[:,None]*baseb)
    if np.max(np.abs(prior_sum-1.0),initial=0)>2e-10 or np.max(np.abs(prior_moment-.25),initial=0)>2e-10:
        raise ValueError("030 prior quadrature does not close exact parent mass and moments")
    restored_route_anchor_count=int(np.count_nonzero(prior_positive & (basew==0.0)))
    if restored_route_anchor_count:
        raise ValueError("032d post-blend matrix unexpectedly dropped prior-positive anchors")
    contexts={r:load_role_context(r,oldman,candidate,tups,mass,srcxyz,tnodes) for r in ("Skin","Fat")}
    matrix_packet_offsets={}
    for mat in (15,16):
        q=(ints["material"]==mat)&(kinds=="existing_packet_anchor")
        ids=np.sort(ints["packet_carrier_row"][q])
        if not len(ids) or not np.array_equal(ids,np.arange(int(ids[0]),int(ids[0])+len(ids))):
            raise ValueError("matrix existing packet rows are not one contiguous global domain")
        matrix_packet_offsets[mat]=int(ids[0])
    # The 70,507 matrix base rows are an exact partition of 46,856 packet
    # anchors and 23,651 empty-parent centroids; verify this before applying LP weights.
    for role in ("Skin","Fat"):
        ctx=contexts[role];old=ctx["packet"];off=ctx["oldoff"];ow=ctx["oldw"];ob=ctx["oldb"]
        for i in range(n):
            if int(ints["material"][i])!=ctx["mat"]: continue
            g=int(ints["source_parent_row"][i]);kind=kinds[i];packetrow=role_packet_row(ints["packet_carrier_row"][i],kind,matrix_packet_offsets[ctx["mat"]]);li=ctx["pmap"][g]
            if kind=="existing_packet_anchor":
                if packetrow<0 or packetrow>=len(ow): raise ValueError("matrix old packet carrier row range")
                if int(old["global_source330_tet_rows"][li])!=g or not (off[li]<=packetrow<off[li+1]): raise ValueError("matrix existing anchor parent/CSR mapping")
                if not np.array_equal(baseb[i],ob[packetrow]) or abs(float(fl["prior_weight"][i])-float(ow[packetrow]))>1e-15: raise ValueError("matrix existing anchor bary/prior differs from current 027 packet")
            elif kind=="implicit_centroid":
                if packetrow!=-1 or off[li]!=off[li+1] or not np.array_equal(baseb[i],np.full(4,.25)) or abs(float(fl["prior_weight"][i])-1.0)>1e-15: raise ValueError("matrix implicit centroid differs from empty-parent base")
            elif kind!="source_tet_candidate": raise ValueError("unknown matrix anchor record kind: "+str(kind))
    records_by_role={"Skin":[],"Fat":[]}
    # Positive matrix anchors (including old packet, implicit centroid, and 030 candidates).
    for i in np.flatnonzero(basew>0):
        g=int(ints["source_parent_row"][i]);role="Skin" if int(tups[g,0])==15 else "Fat" if int(tups[g,0])==16 else None
        if role is None:raise ValueError("positive matrix parent has unknown material")
        records_by_role[role].append(make_row(role,g,baseb[i],basew[i],ints["global_witness_child_row"][i],fl["global_witness_distance_m"][i],
            ints["route_mask"][i],cutchild[i],cutdist[i],base11mask[i],base11child[i],base11dist[i],"matrix",i,
            role_packet_row(ints["packet_carrier_row"][i],kinds[i],matrix_packet_offsets[int(tups[g,0])]) if kinds[i]=="existing_packet_anchor" else -1,kinds[i]))
    # Positive adaptive rows.
    for i in np.flatnonzero(ad_w>0):
        g=int(ad_g[i]);role="Skin" if int(tups[g,0])==15 else "Fat" if int(tups[g,0])==16 else None
        if role is None:raise ValueError("adaptive parent has unknown material")
        records_by_role[role].append(make_row(role,g,ad_b[i],ad_w[i],ad_child[i],ad_dist[i],ad_mask[i],ad_cuts[i],ad_cutd[i],
            ad11mask[i],ad11child[i],ad11dist[i],"adaptive",n+i,-1))
    # Recompute exact global/cut child routes and one-hop membership.
    for role in ("Skin","Fat"):
        validate_routes(records_by_role[role],contexts[role],candidate,src_tet_xyz,tups)
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise ValueError(f"refusing nonempty output directory {out}")
    out.mkdir(parents=True,exist_ok=True)
    result={};verifications={}
    for role in ("Skin","Fat"):
        result[role],verifications[role]=write_role(role,out,records_by_role[role],contexts[role],candidate,tups,src_tet_xyz,contexts[role]["old_witness_path"])
    batch={"schema":"numi.h706.adaptive-source-anchor-packet-rebind.034.batch.v1","status":"PASS_SOURCE_ONLY_ADAPTIVE_PACKET_READBACK",
        "physical_admission":False,"native_steps":0,"allocator_run":False,"source_run_manifest":{"path":str(Path(args.source_run_manifest)),"sha256":sha(Path(args.source_run_manifest))},
        "bridge_source":{"path":str(Path(__file__).resolve()),"sha256":sha(Path(__file__).resolve())},"proposal_report":{"path":str(PROPOSAL_REPORT),"sha256":sha(PROPOSAL_REPORT)},"proposal":{"path":str(PROPOSAL_PATH),"sha256":sha(PROPOSAL_PATH)},
        "matrix":{"path":str(MATRIX_PATH),"sha256":sha(MATRIX_PATH),"rows":n},"matrix_run_manifest":{"path":str(MATRIX_RUN_PATH),"sha256":sha(MATRIX_RUN_PATH)},"matrix_execution":{"path":str(MATRIX_EXEC_PATH),"sha256":sha(MATRIX_EXEC_PATH)},"prior_support_preservation":{"second_blend_epsilon":epsilon,"original_positive_anchor_count":prior_positive_count,
          "retained_positive_anchor_count":prior_positive_count,"restored_zero_lp_anchor_count":restored_route_anchor_count,
          "proposal_gate":prior_gate},"typed_candidate":{"path":str(B.CAND),"sha256":sha(B.CAND)},
        "roles":result,"verifications":verifications,"native_child_mesh_readback":child_mesh_check,"index_controls":index_controls(),
        "prior_support_preservation":{"proposal_gate":prior_gate,"prior_positive_matrix_anchors":prior_positive_count,
            "retained_prior_positive_matrix_anchors":prior_positive_count,"second_blend_epsilon":0.0,
            "statement":"032d already applied its declared prior restoration blend; bridge applies no second blend."},
        "current11_witness_binding":{"base_rows":n,"adaptive_rows":len(ad_w),"unique_slots_deduplicated_per_packet_carrier_child":True,"route_mask_width":11},
        "limitation":"Packet and historical/current11 witnesses are source-only, independently replayed against frozen source/candidate/P1 inputs; no allocator outcome or native admission is claimed."}
    rp=out/"batch-packet-verification.json";rp.write_text(json.dumps(batch,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":batch["status"],"report":str(rp),"report_sha256":sha(rp),
        "roles":{r:{"carriers":result[r]["carrier_count"],"support_slots":verifications[r]["support_pairs_replayed"]-verifications[r]["primary_rows_replayed"],
                    "npz":result[r]["outputs"]["npz"]["sha256"],"witness":result[r]["outputs"]["witness_csv"]["sha256"]} for r in result}},sort_keys=True))

# Global CLI args paths are set before role writing to keep receipts explicit.
MATRIX_PATH=Path()
PROPOSAL_PATH=Path()
PROPOSAL_REPORT=Path()
MATRIX_RUN_PATH=MATRIX_RUN_DEFAULT
MATRIX_EXEC_PATH=MATRIX_EXEC_DEFAULT

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--matrix",type=Path,default=MATRIX_DEFAULT)
    ap.add_argument("--proposal",type=Path)
    ap.add_argument("--proposal-report",type=Path)
    ap.add_argument("--source-run-manifest",type=Path,default=OLD_RUN)
    ap.add_argument("--matrix-run-manifest",type=Path,default=MATRIX_RUN_DEFAULT)
    ap.add_argument("--matrix-execution",type=Path,default=MATRIX_EXEC_DEFAULT)
    ap.add_argument("--baseline-route-epsilon",type=float,default=0.0)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    if args.self_test: print(json.dumps(index_controls(),sort_keys=True))
    else:
        if args.proposal is None or args.proposal_report is None or args.output is None: ap.error("--proposal, --proposal-report and --output are required outside --self-test")
        build(args)


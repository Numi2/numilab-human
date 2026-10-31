from pathlib import Path
import hashlib,json,sys
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
OUT=H/"d-shortedge-quotient-385-run006"
OUT.mkdir(parents=True,exist_ok=False)
PY=Path("/Users/numi/numi-human-prep-venv-20261005/bin/python")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def save_json(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def coord_key(x): return np.ascontiguousarray(np.asarray(x,dtype=np.float32)).tobytes()
def unique_coord_map(points):
    m={}
    dup=set()
    for i,x in enumerate(points):
        k=coord_key(x)
        if k in m: dup.add(k)
        else: m[k]=int(i)
    for k in dup:m[k]=-1
    return m
def map_points(src,dst_lookup):
    return np.asarray([dst_lookup.get(coord_key(x),-1) for x in src],np.int64)
def qface(P,F):
    tri=np.asarray(P,np.float32)[np.asarray(F,np.int64)].astype(np.float64)
    a=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0];c=tri[:,2]-tri[:,1]
    area2=np.linalg.norm(np.cross(a,b),axis=1)
    L=np.maximum.reduce((np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.linalg.norm(c,axis=1)))
    return np.divide(area2,L*L,out=np.zeros_like(area2),where=L>0)
def tet_measure(P,T):
    if not len(T):return np.empty(0),np.empty(0),np.empty(0)
    z=np.asarray(P,np.float32)[np.asarray(T,np.int64)].astype(np.float64)
    a=z[:,1]-z[:,0];b=z[:,2]-z[:,0];c=z[:,3]-z[:,0]
    det=np.einsum("ij,ij->i",a,np.cross(b,c))
    L=np.zeros(len(T),np.float64)
    for i in range(4):
        for j in range(i+1,4):L=np.maximum(L,np.linalg.norm(z[:,i]-z[:,j],axis=1))
    q=np.divide(det,L**3,out=np.zeros_like(det),where=L>0)
    return det,det/6.0,q
def exact_coord_int(x):
    n,d=float(np.float32(x)).as_integer_ratio()
    if (1<<149)%d:raise ValueError("float32 coordinate denominator is not a divisor of 2^149")
    return n*((1<<149)//d)
def exact_det_nodes(P,nodes,cache):
    xyz=[]
    for v in map(int,nodes):
        if v not in cache:cache[v]=tuple(exact_coord_int(c) for c in P[v])
        xyz.append(cache[v])
    a=[xyz[1][i]-xyz[0][i] for i in range(3)]
    b=[xyz[2][i]-xyz[0][i] for i in range(3)]
    c=[xyz[3][i]-xyz[0][i] for i in range(3)]
    cr=(b[1]*c[2]-b[2]*c[1],b[2]*c[0]-b[0]*c[2],b[0]*c[1]-b[1]*c[0])
    return a[0]*cr[0]+a[1]*cr[1]+a[2]*cr[2]

src={
 "O":(H/"joint-native-OI-ears-279-root/O-batch006-dp001/typed-280-003/candidate-O-typed-current.npz","4e0c46662721fed33d4bd4efb390c6668ff67581d2484fcd9ba9058834fa7a20"),
 "I":(H/"paired-I-interface-377-root/qualification378001/typed365/candidate-I-typed-current.npz","8e04da9ad1e9c04216dae1fde854a3d69c8e5bb4c7a7c4a4796754675ce7b747"),
 "D":(H/"d-shortedge-quotient-351-run002/typed-rebind-003/candidate-D-typed-current.npz","18517e7dc0b1b25d67eea9691ad98bdbcfd07d18b9e1fe45f25b95515ee99ade")
}
for r,(p,h) in src.items():
    if sha(p)!=h:raise ValueError(f"{r} candidate hash changed")
parent={r:load(p) for r,(p,h) in src.items()}
# Make explicit identity row-domain wrappers so the 196 generic owner can operate
# directly on these already-typed current candidates without reinterpreting history.
wrapped={}
outpin={}
for r,(p,h) in src.items():
    a=dict(parent[r]);npt=len(a["points_world_m_f32"]);nf=len(a["faces"])
    a["point_output_to_parent_rows"]=np.arange(npt,dtype=np.int64)
    a["point_output_to_input_rows"]=np.arange(npt,dtype=np.int64)
    a["face_output_to_parent_rows"]=np.arange(nf,dtype=np.int64)
    a["face_output_to_input_rows"]=np.arange(nf,dtype=np.int64)
    a["immediate_parent_candidate_sha256"]=np.asarray([h],dtype="U64")
    a["typed_parent_candidate_sha256"]=np.asarray([h],dtype="U64")
    a["source_candidate_sha256"]=np.asarray([h],dtype="U64")
    a["source_candidate_path"]=np.asarray([str(p)],dtype="U1024")
    a["immediate_parent_candidate_path"]=np.asarray([str(p)],dtype="U1024")
    wp=OUT/f"identity-seed-{r}.npz"
    np.savez_compressed(wp,**a)
    wh=sha(wp);wrapped[r]=(wp,wh)
    if not np.array_equal(a["points_world_m_f32"],parent[r]["points_world_m_f32"]) or not np.array_equal(a["faces"],parent[r]["faces"]):
        raise ValueError(f"{r} identity seed changed geometry")
    outpin[r]={"path":str(wp),"sha256":wh,"geometry_parent_path":str(p),"geometry_parent_sha256":h,"point_count":npt,"face_count":nf}
seed_report={
 "schema":"h706.current-typed-candidate-identity-seed.v1",
 "status":"PASS_BOUNDED_EXACT_FIXED_COORDINATE_TOPOLOGY_PROPOSAL",
 "derivation":"metadata-only identity row wrapper; all geometry arrays are bit-identical to the pinned typed current candidate",
 "outputs":{r:{"path":str(wrapped[r][0]),"sha256":wrapped[r][1]} for r in ("O","I","D")},
 "accepted_operations":[],
 "geometry_parents":{r:{"path":str(src[r][0]),"sha256":src[r][1]} for r in ("O","I","D")},
 "coordinates_changed":False,"physical_admission":False,"native_steps":0
}
seed_report_path=OUT/"identity-seed-report.json";save_json(seed_report_path,seed_report)
seed_report_sha=sha(seed_report_path)

# Build the authorized edge subset from the 350 D-only bad-cell witness and map
# all points by exact F32 coordinate identity into both current D and d4906.
census_path=H/"native-boundary-edge-witness-census-350-root/fat-short-surface-edge-witnesses.npz"
census_sha="f5242f1ab6d9f1a74f5d2fd043c21c025dee1ac96d640fc91e4fe60f4063902d"
base336_path=H/"fat-inferred-identity-mask-336-root/fat-run001/fixed-boundary-volume-proposal.npz"
base336_sha="2fc5aa8930fe50f940b071a954c0393b8e41afee514d64ace81e1c419e2b1905"
vol_path=H/"fat614-yy-volume-replay-004/run-014-current-target-rebound-v3/fat-D307-I377-YY-components-current-target-rebound.npz"
vol_sha="8915c3c7a58f8622d0568da5006bf4b7b0f2832f802b65e29bd144e95f11001d"
basevol_path=H/"fat614-yy-volume-replay-002/fat-comp2-comp6-rounded-boundary-points-volume-proposal.npz"
basevol_sha="d4906b176f78283f4073f9bd1e8d202a7301b57550a7b1b71fb101e76c4d236f"
cavity_path=H/"fat614-compact-cavities-004-anchor-stars/connected-cavity-shells.npz"
cavity_sha="5810a3139b619630b8658c407a3b22cde1939ba9c917d273852622743ce5fe44"
current_cavity_path=H/"fat614-yy-volume-replay-004/run-021-current-target-cavities-bounded-closure/connected-cavity-shells.npz"
current_cavity_sha="34dfe3eb2a49f56e036bc5c03ae4e4e72a17ab669dc9bdaf6a299cb905204a85"
current_cavity_report=H/"fat614-yy-volume-replay-004/run-021-current-target-cavities-bounded-closure/report.json"
current_cavity_report_sha="4241341f469c382ea37abe66ce4e78d21df0d52bcbda22246a9182ac225706d7"
i352_report=H/"paired-I-interface-377-root/run002/report.json"
i352_sha="baa09bce6e6995268ec5a932be530573b8de69bb202025568ad41dd9e5325486"
d307_report=H/"d-shortedge-quotient-351-run002/surface-run001/report.json"
d307_sha="b549dd67254f0e44ae69c28f69545c5847235ca12dbdf69b98f7d3295d77156b"
protected_path=H/"verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json"
protected_sha="ed6e15c6dc9324b09cd4514e1fc3a149fc92a07215128a9820d72fca3f350a71"
for p,h in ((census_path,census_sha),(base336_path,base336_sha),(vol_path,vol_sha),(basevol_path,basevol_sha),(cavity_path,cavity_sha),(current_cavity_path,current_cavity_sha),(current_cavity_report,current_cavity_report_sha),(i352_report,i352_sha),(d307_report,d307_sha),(protected_path,protected_sha)):
    if h.startswith("REPLACE_"):continue
    if sha(p)!=h:raise ValueError(f"source pin mismatch {p}: {sha(p)} != {h}")
C=load(census_path);E=np.asarray(C["edge_point_rows"],np.int64);layers=np.asarray(C["layer"],np.int64);lens=np.asarray(C["length_m"],np.float64);inc=np.asarray(C["bad_tet_incidence"],np.int64)
B=load(base336_path);BP=np.asarray(B["points_world_m"],np.float32)
V=load(vol_path);VP=np.asarray(V["points_world_m"],np.float32);VT=np.asarray(V["tetrahedra"],np.int64);VMAT=np.asarray(V["material_indices"],np.int32);VB=np.asarray(V["boundary_faces"],np.int64)
V0=load(basevol_path);VT0=np.asarray(V0["tetrahedra"],np.int64)
D=parent["D"];DP=np.asarray(D["points_world_m_f32"],np.float32);DF=np.asarray(D["faces"],np.int64)
source_to_D=map_points(BP,unique_coord_map(DP))
D_to_V=map_points(DP,unique_coord_map(VP))
V_to_D=map_points(VP,unique_coord_map(DP))
declared_D_to_V=np.asarray(V["fat_current_target_rebind_D_point_to_volume_rows"],np.int64)
if not np.array_equal(D_to_V,declared_D_to_V):raise ValueError("coordinate crosswalk disagrees with authenticated current-target D point map")
if np.any(D_to_V<0) or not np.array_equal(DP,VP[D_to_V]):raise ValueError("current D point rows do not map bitwise to rebound Fat volume")
if len(VT)!=len(VMAT):raise ValueError("Fat volume material/cell rows mismatch")
if len(D_to_V)!=len(DP):raise ValueError("D volume crosswalk malformed")
# Pending cavity points from the recorded cavity sets, plus D split points.
cz=load(cavity_path);po=np.asarray(cz["point_offsets"],np.int64);grow=np.asarray(cz["global_point_rows"],np.int64)
pending_cavity_indices=(2,3,4,5,6,7,8,10,11)
pending_source_rows=np.concatenate([grow[po[c]:po[c+1]] for c in pending_cavity_indices]).astype(np.int64)
pending_d=set(map(int,source_to_D[pending_source_rows][source_to_D[pending_source_rows]>=0]))
current_cavity=load(current_cavity_path)
current_cavity_cells=np.asarray(current_cavity["input_cell_rows"],np.int64)
if np.any(current_cavity_cells<0) or np.any(current_cavity_cells>=len(VT)):raise ValueError("run021 cavity cell rows outside rebound candidate")
current_cavity_volume_points=np.unique(VT[current_cavity_cells].reshape(-1))
current_cavity_mapped=V_to_D[current_cavity_volume_points]
current_cavity_d=set(map(int,current_cavity_mapped[current_cavity_mapped>=0]))
derived=np.asarray(D.get("derived_boundary_split_point_rows",np.empty(0)),np.int64).reshape(-1)
# Include every current point with no original source row: all derived split points.
qsp=np.asarray(D.get("quality_star_point_output_to_input_row",np.arange(len(DP))),np.int64)
derived=np.unique(np.concatenate((derived,np.flatnonzero(qsp<0).astype(np.int64))))
# Protect the complete source-cell/node stars of all accepted I377 operations.
# Their source rows are in the pinned d4906 input domain, so read those cells
# from the exact input rather than applying them to the compacted output domain.
i377=json.loads(i352_report.read_text())
fat_i377=next(x for x in i377["roles"] if x["role"]=="Fat")
i352_cells=set();i377_vnodes=set()
for op in fat_i377["operations"]:
    rr=np.asarray(op["source_cell_rows"],np.int64)
    if np.any(rr<0) or np.any(rr>=len(VT0)):raise ValueError("I377 source cell row outside d4906 domain")
    i352_cells.update(map(int,rr))
    i377_vnodes.update(map(int,VT0[rr].reshape(-1)))
    i377_vnodes.update((int(op["kept_volume_node"]),int(op["retired_volume_node"])))
i352_d=set(map(int,V_to_D[np.fromiter(i377_vnodes,dtype=np.int64)]))
i352_d.discard(-1)
# Protect all D surface vertices in the accepted D307 affected face stars.
d307=json.loads(d307_report.read_text())
d307_seed_record=d307["typed_ancestry_parents"]["D"]
d307_seed_path=Path(d307_seed_record["path"]);d307_seed_sha=str(d307_seed_record["sha256"])
if sha(d307_seed_path)!=d307_seed_sha:raise ValueError("D307 ancestry parent SHA mismatch")
d307_seed=load(d307_seed_path);d307_seed_faces=np.asarray(d307_seed["faces"],np.int64);d307_seed_points=np.asarray(d307_seed["points_world_m_f32"],np.float32)
current_D_lookup=unique_coord_map(DP);d307_source_star_points=set()
for op in d307.get("accepted_operations",[]):
    if op.get("role")!="D":continue
    fr=np.asarray(op["affected_parent_face_rows"],np.int64)
    if np.any(fr<0) or np.any(fr>=len(d307_seed_faces)):raise ValueError("D307 affected face row outside typed parent")
    d307_source_star_points.update(map(int,d307_seed_faces[fr].reshape(-1)))
d307_mapped=map_points(d307_seed_points[np.asarray(sorted(d307_source_star_points),np.int64)],current_D_lookup)
d307_d=set(map(int,d307_mapped[d307_mapped>=0]))
# Protect full current D face-stars of all current target mismatch faces.
# Oriented records are preserved so opposite-winding debt counts as both sides.
from collections import Counter
def oriented_key(tri):
    x=tuple(map(int,tri));s=tuple(sorted(x));rank={v:i for i,v in enumerate(s)}
    inv=int(rank[x[0]]>rank[x[1]])+int(rank[x[0]]>rank[x[2]])+int(rank[x[1]]>rank[x[2]])
    return (s[0],s[1],s[2],-1 if inv%2 else 1)
wanted=Counter(oriented_key(x) for x in np.asarray(V["desired_boundary_faces"],np.int64))
actual=Counter(oriented_key(x) for x in VB)
missing_records=list((wanted-actual).elements());extra_records=list((actual-wanted).elements())
if len(missing_records)!=127 or len(extra_records)!=197:raise ValueError(f"rebound oriented target mismatch changed: missing={len(missing_records)} extra={len(extra_records)}")
mismatch_volume_nodes=set(int(v) for f in missing_records+extra_records for v in f[:3])
mismatch_D_seed=set(int(i) for i,v in enumerate(D_to_V) if int(v) in mismatch_volume_nodes)
mismatch_D_fullstar=set(mismatch_D_seed)
for tri in DF:
    if any(int(v) in mismatch_D_seed for v in tri):mismatch_D_fullstar.update(map(int,tri))
# Protected D route vertices.
pr=json.loads(protected_path.read_text())
protected_D=set()
for rec in pr["surface_routes"]:
    if rec["role"]!="D":continue
    k=coord_key(np.asarray(rec["target_world_m_f32"],np.float32))
    ids=np.flatnonzero(np.asarray([coord_key(x)==k for x in DP],dtype=bool))
    if len(ids)!=1:raise ValueError(f"protected D route maps to {len(ids)} rows")
    protected_D.add(int(ids[0]))
excluded=set(pending_d)|current_cavity_d|set(map(int,derived))|i352_d|d307_d|mismatch_D_fullstar|protected_D
# Exact current D edge and vertex-face topology index.
edge_faces={};neighbors=[set() for _ in range(len(DP))];vfaces=[[] for _ in range(len(DP))]
for fi,tri in enumerate(DF):
    a,b,c=map(int,tri);vfaces[a].append(fi);vfaces[b].append(fi);vfaces[c].append(fi)
    for u,v,w in ((a,b,c),(b,c,a),(c,a,b)):
        e=(u,v) if u<v else (v,u);edge_faces.setdefault(e,[]).append((fi,w))
        neighbors[u].add(v);neighbors[v].add(u)
qD=qface(DP,DF)
# Exact Fat boundary face keys and node-to-cell incidence for cheap local quotient tests.
boundary_keys={tuple(sorted(map(int,f))) for f in VB}
flat=VT.reshape(-1);rows=np.repeat(np.arange(len(VT),dtype=np.int64),4);order=np.argsort(flat,kind="stable")
counts=np.bincount(flat,minlength=len(VP));off=np.empty(len(VP)+1,np.int64);off[0]=0;np.cumsum(counts,out=off[1:]);node_cells=rows[order]
def incident_cells(node):return node_cells[int(off[node]):int(off[node+1])]
def vol_quality(tets):
    det,v,q=tet_measure(VP,tets)
    return det,v,q,(det>0)&(v>1e-18)&(q>=1e-5)
def trial_orientation(dropD,keepD,retired_surface):
    dropV=int(D_to_V[dropD]);keepV=int(D_to_V[keepD])
    if dropV<0 or keepV<0 or dropV==keepV:return None,"surface_endpoint_missing_from_d4906"
    affected=incident_cells(dropV)
    old=VT[affected]
    new=np.where(old==dropV,keepV,old)
    sorted_new=np.sort(new,axis=1);deleted=np.any(sorted_new[:,1:]==sorted_new[:,:-1],axis=1)
    kept=new[~deleted]
    if not len(affected):return None,"no_volume_incidence"
    od,ov,oq,og=vol_quality(old);nd,nv,nq,ng=vol_quality(kept)
    if np.any(nd<=0) or np.any(nv<=1e-18):return None,"volume_nonpositive_or_absolute_gate"
    if np.any(og[~deleted]&~ng):return None,"new_volume_native_quality_failure"
    if int(np.count_nonzero(~ng))>int(np.count_nonzero(~og)):return None,"volume_quality_count_increase"
    # All surface faces removed/replaced by this operation must exist on the current volume boundary.
    for fr in retired_surface:
        tri=DF[int(fr)]
        mapped=D_to_V[tri]
        if np.any(mapped<0) or tuple(sorted(map(int,mapped))) not in boundary_keys:return None,"retired_surface_face_not_in_d4906_boundary"
    # Replacing this endpoint must not duplicate retained tetrahedra within the local star.
    keys=[tuple(sorted(map(int,t))) for t in kept]
    if len(set(keys))!=len(keys):return None,"local_duplicate_tetrahedron"
    tetset=set(map(int,affected))
    material_counts={str(int(m)):int(np.count_nonzero(VMAT[affected]==m)) for m in np.unique(VMAT[affected])}
    return {"drop_surface_row":int(dropD),"keep_surface_row":int(keepD),"drop_volume_row":dropV,"keep_volume_row":keepV,"affected_cell_rows":affected.astype(int).tolist(),"affected_cell_count":len(affected),"deleted_cell_count":int(np.count_nonzero(deleted)),"bad_before":int(np.count_nonzero(~og)),"bad_after":int(np.count_nonzero(~ng)),"volume_delta_m3":float(-ov[deleted].sum()),"material_cell_counts":material_counts},None
# Surface bad-edge witness mapping; edge list is restricted to D layer <=1 micron.
dmask=(layers==2)&(lens<=1e-5)
wedge_to_sources={}
for wi in np.flatnonzero(dmask):
    mapped=source_to_D[E[wi]]
    if np.any(mapped<0) or mapped[0]==mapped[1]:continue
    e=tuple(sorted(map(int,mapped)))
    wedge_to_sources.setdefault(e,[]).append(int(wi))
# Sort witnesses shortest first. Candidate must be a current D edge with a link-valid 2-face star and no pending/protected vertex.
preflight_reasons={}
def reject(k):preflight_reasons[k]=preflight_reasons.get(k,0)+1
candidates=[]
for e,srcrows in wedge_to_sources.items():
    u,v=e
    if u in excluded or v in excluded:reject("excluded_pending_cavity_i352_split_or_protected_endpoint");continue
    ef=edge_faces.get(e,[])
    if len(ef)!=2:reject("current_D_edge_not_two_face_manifold");continue
    opos={int(w) for _,w in ef}
    common=(neighbors[u]&neighbors[v])-{u,v}
    if common!=opos:reject("D_link_condition");continue
    star_rows=sorted(set(vfaces[u])|set(vfaces[v]))
    star_vertices=set(map(int,DF[np.asarray(star_rows,np.int64)].reshape(-1)))
    if star_vertices & excluded:reject("D_star_intersects_excluded_pending_point");continue
    length=float(np.linalg.norm(DP[u].astype(np.float64)-DP[v].astype(np.float64)))
    if length>1e-5:reject("current_D_edge_longer_than_10um");continue
    removed=[int(x[0]) for x in ef]
    changed_u=sorted(int(fr) for fr in vfaces[u] if v not in DF[int(fr)])
    changed_v=sorted(int(fr) for fr in vfaces[v] if u not in DF[int(fr)])
    orientations=[];orientation_cell_union=set();quality=[]
    for drop,keep,changed in ((u,v,changed_u),(v,u,changed_v)):
        retired=sorted(set(changed)|set(removed))
        got,why=trial_orientation(drop,keep,retired)
        if got is None:
            reject("volume_"+why);continue
        orientations.append((drop,keep));orientation_cell_union.update(got["affected_cell_rows"])
        quality.append(got)
    if not orientations:continue
    # Existing positive-cell and measure checks are repeated exactly on selected rows.
    candidates.append({"edge":e,"length_m":length,"source_350_rows":srcrows,"surface_star_rows":star_rows,"surface_star_vertices":sorted(star_vertices),"volume_star_cells":sorted(orientation_cell_union),"orientations":orientations,"local_volume_trials":quality,"seed_face":min(int(x[0]) for x in ef),"seed_qface":float(min(qD[int(x[0])] for x in ef))})
candidates.sort(key=lambda x:(x["length_m"],x["seed_qface"],x["seed_face"],x["edge"]))
# Greedy exact-identity batch: disjoint surface stars and disjoint current volume cell stars.
selected=[];used_surf=set();used_cells=set()
for c in candidates:
    if len(selected)>=2048:break
    sv=set(c["surface_star_vertices"]);vc=set(c["volume_star_cells"])
    if sv&used_surf:reject("overlaps_selected_D_surface_star");continue
    if vc&used_cells:reject("overlaps_selected_Fat_volume_cell_star");continue
    # Exact dyadic integer orientation proof for every retained affected tet in every allowed direction.
    exact_cache={}
    passes_exact=True
    for dropD,keepD in c["orientations"]:
        dropV=int(D_to_V[dropD]);keepV=int(D_to_V[keepD]);affected=incident_cells(dropV);old=VT[affected];new=np.where(old==dropV,keepV,old)
        sorted_new=np.sort(new,axis=1);deleted=np.any(sorted_new[:,1:]==sorted_new[:,:-1],axis=1)
        for tet in old:
            if exact_det_nodes(VP,tet,exact_cache)<=0:passes_exact=False;break
        if not passes_exact:break
        for tet in new[~deleted]:
            if exact_det_nodes(VP,tet,exact_cache)<=0:passes_exact=False;break
        if not passes_exact:break
    if not passes_exact:reject("exact_F32_tet_orientation");continue
    selected.append(c);used_surf|=sv;used_cells|=vc
# Save selected orientation map; the generic exact surface owner will make the final decision.
edge_rows=np.asarray([c["edge"] for c in selected],np.int64).reshape((-1,2))
orient_edge=[];orient_drop=[];orient_keep=[]
for c in selected:
    for dr,kr in c["orientations"]:
        orient_edge.append(c["edge"]);orient_drop.append(dr);orient_keep.append(kr)
npz=OUT/"allowed-D-edge-map.npz"
edge_star_off=[0];edge_star_rows=[];edge_volume_off=[0];edge_volume_rows=[]
for c in selected:
    edge_star_rows.extend(sorted(map(int,c["surface_star_vertices"])));edge_star_off.append(len(edge_star_rows))
    edge_volume_rows.extend(sorted(map(int,c["volume_star_cells"])));edge_volume_off.append(len(edge_volume_rows))
np.savez_compressed(npz,D_edge_rows=edge_rows,D_allowed_orientation_edge_rows=np.asarray(orient_edge,np.int64).reshape((-1,2)),D_allowed_collapsed_point_rows=np.asarray(orient_drop,np.int64),D_allowed_retained_point_rows=np.asarray(orient_keep,np.int64),D_source_350_witness_counts=np.asarray([len(c["source_350_rows"]) for c in selected],np.int32),D_edge_length_m=np.asarray([c["length_m"] for c in selected],np.float64),D_seed_face_rows=np.asarray([c["seed_face"] for c in selected],np.int64),D_seed_face_qface=np.asarray([c["seed_qface"] for c in selected],np.float64),D_edge_surface_star_offsets=np.asarray(edge_star_off,np.int64),D_edge_surface_star_point_rows=np.asarray(edge_star_rows,np.int64),D_edge_volume_star_offsets=np.asarray(edge_volume_off,np.int64),D_edge_volume_star_current_cell_rows=np.asarray(edge_volume_rows,np.int64),D_preflight_protected_target_mismatch_point_rows=np.asarray(sorted(mismatch_D_fullstar),np.int64),D_preflight_protected_current_cavity_point_rows=np.asarray(sorted(current_cavity_d),np.int64),D_preflight_protected_D307_point_rows=np.asarray(sorted(d307_d),np.int64),D_preflight_protected_I377_point_rows=np.asarray(sorted(i352_d),np.int64))
npz_sha=sha(npz)
# Current read-only qface census for the generic owner.
MIN_Q=1e-12
qcounts={r:int(np.count_nonzero(qface(np.asarray(parent[r]["points_world_m_f32"],np.float32),np.asarray(parent[r]["faces"],np.int64))<MIN_Q)) for r in ("O","I","D")}
quality_report={"schema":"h706.current-qbound-witness-census.v1","status":"READ_ONLY_CURRENT_QBOUND_WITNESS_CENSUS","source_batch_report":{"path":str(seed_report_path),"sha256":seed_report_sha},"witness_counts":qcounts,"candidate_faces":{r:{"path":str(wrapped[r][0]),"sha256":wrapped[r][1]} for r in ("O","I","D")},"normalized_det_surface_floor":MIN_Q,"purpose":"degenerate-face prefilter only; actual Fat tet positivity and no-new-native-quality were checked separately on the paired volume"}
quality_path=OUT/"quality-prepass-report.json";save_json(quality_path,quality_report)
back_path=H/"fixed-boundary-quality-topology-196/collapse-batch-004-from-003-001/back412-scope-audit-001/report.json";back_sha="d20f1fd175b699a00dbe7077aae33b787054126a7d6bfef28c6d25894b30db0e"
if sha(back_path)!=back_sha:raise ValueError("back scope report pin mismatch")
cfg={"schema":"h706.fixed-boundary-short-edge-collapse-run-config.v1","batch_id":"385-D-350-witness-contraction-10um","current_candidates":{r:{"path":str(wrapped[r][0]),"sha256":wrapped[r][1]} for r in ("O","I","D")},"typed_ancestry_parents":{r:{"path":str(src[r][0]),"sha256":src[r][1]} for r in ("O","I","D")},"lineage_layers":[{"layer_id":"current-typed-identity-seed","report":{"path":str(seed_report_path),"sha256":seed_report_sha},"candidates":{r:{"path":str(wrapped[r][0]),"sha256":wrapped[r][1]} for r in ("O","I","D")}}],"quality_prepass_report":{"path":str(quality_path),"sha256":sha(quality_path)},"expected_bad_face_counts":qcounts,"back_scope_report":{"path":str(back_path),"sha256":back_sha},"authorized_edge_map":{"path":str(npz),"sha256":npz_sha,"schema":"h706.d-short-edge-quotient-authorized-edges.v1"},"parameters":{"max_edge_length_m":1e-5,"max_accepted":2048,"max_trials":100000,"wall_budget_seconds":90.0,"normalized_det_minimum":MIN_Q,"minimum_altitude_m":20e-9,"minimum_angle_degrees":0.0,"replacement_qbound_policy":"strict_local_net_reduction"}}
cfgp=OUT/"run-config.json";save_json(cfgp,cfg)
# Pin preparation inputs and concise census/selection accounting.
prep={
 "schema":"h706.d-short-edge-quotient-385-preflight.v1",
 "status":"PASS_D_ONLY_FAT_PAIRED_LOCAL_PREFLIGHT",
 "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
 "source_pins":{
  "edge_census":{"path":str(census_path),"sha256":census_sha},
  "source_Fat336":{"path":str(base336_path),"sha256":base336_sha},
  "current_Fat_rebound_8915":{"path":str(vol_path),"sha256":vol_sha},
  "d4906_source_volume":{"path":str(basevol_path),"sha256":basevol_sha},
  "pending_cavities":{"path":str(cavity_path),"sha256":cavity_sha,"excluded_component_ids":list(pending_cavity_indices)},
  "current_target_cavities_021":{"path":str(current_cavity_path),"sha256":current_cavity_sha,"report_path":str(current_cavity_report),"report_sha256":current_cavity_report_sha,"input_cell_count":len(current_cavity_cells),"current_D_point_star_count":len(current_cavity_d)},
  "inner_interface_I377":{"path":str(i352_report),"sha256":i352_sha,"excluded_source_cell_count":len(i352_cells),"excluded_volume_node_count":len(i377_vnodes)},
  "previous_D307":{"path":str(d307_report),"sha256":d307_sha,"excluded_source_star_point_count":len(d307_source_star_points),"excluded_current_D_points":len(d307_d)},
  "current_target_mismatch":{"target_face_count":len(V["desired_boundary_faces"]),"actual_boundary_face_count":len(VB),"missing_oriented_face_count":len(missing_records),"extra_oriented_face_count":len(extra_records),"excluded_D_seed_points":len(mismatch_D_seed),"excluded_D_fullstar_points":len(mismatch_D_fullstar)},
  "current_D":{"path":str(src["D"][0]),"sha256":src["D"][1]},
  "current_O":{"path":str(src["O"][0]),"sha256":src["O"][1]},
  "current_I":{"path":str(src["I"][0]),"sha256":src["I"][1]}
 },
 "mapping":{"D_candidate_points":len(DP),"D_current_points_matching_rebound":int(np.count_nonzero(D_to_V>=0)),"source336_points_matching_D":int(np.count_nonzero(source_to_D>=0)),"pending_cavity_D_points":len(pending_d),"current_target_cavity_D_points":len(current_cavity_d),"derived_split_D_points":len(derived),"i377_excluded_D_points":len(i352_d),"D307_excluded_D_points":len(d307_d),"target_mismatch_D_seed_points":len(mismatch_D_seed),"target_mismatch_D_fullstar_points":len(mismatch_D_fullstar),"protected_D_points":len(protected_D),"excluded_D_point_count":len(excluded)},
 "witness_census":{"D_layer_witness_edges_le_10um":int(np.count_nonzero(dmask)),"unique_current_D_edge_pairs":len(wedge_to_sources),"current_D_edges_present":sum(1 for e in wedge_to_sources if e in edge_faces)},
 "preflight":{"candidate_edges":len(candidates),"selected_disjoint_edges":len(selected),"selected_allowed_orientations":len(orient_drop),"maximum_edge_m":max((c["length_m"] for c in selected),default=0.0),"local_volume_cells_touched":len(used_cells),"rejection_counts":preflight_reasons,"selected_operations":[{"edge":list(c["edge"]),"length_m":c["length_m"],"orientations":[[int(a),int(b)] for a,b in c["orientations"]],"seed_face":c["seed_face"],"volume_trials":c["local_volume_trials"]} for c in selected]},
 "outputs":{"identity_seed_report":{"path":str(seed_report_path),"sha256":seed_report_sha},"quality_prepass":{"path":str(quality_path),"sha256":sha(quality_path)},"authorized_edge_map":{"path":str(npz),"sha256":npz_sha},"run_config":{"path":str(cfgp),"sha256":sha(cfgp)}},
 "limitations":["This is a D-only local quotient prefilter against the current rebound Fat candidate. It does not prove full boundary, mass, typed promotion, native install, or physical admission.","All current D target split points, pending cavity vertices/cells, accepted D307 and I377 operation stars, protected routes, and full D stars of all current target mismatch faces are excluded.","The generic surface owner must independently pass complete O/I/D, both-wall, all185 and topology gates before any contraction is retained."]
}
prepp=OUT/"preflight.json";save_json(prepp,prep)
print(json.dumps({"status":prep["status"],"out":str(OUT),"selected_edges":len(selected),"selected_orientations":len(orient_drop),"candidate_edges":len(candidates),"D_witnesses":int(dmask.sum()),"exclusions":prep["mapping"],"rejections":preflight_reasons,"qface_counts":qcounts,"preflight_sha256":sha(prepp),"config_sha256":sha(cfgp)},indent=2))

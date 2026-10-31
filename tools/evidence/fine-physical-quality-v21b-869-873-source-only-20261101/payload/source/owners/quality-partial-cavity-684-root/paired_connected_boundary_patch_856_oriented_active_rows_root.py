#!/usr/bin/env python3
"""Paired-state boundary-patch quality proposal using connected 2-2 source flips.

This is a geometry proposal generator, not an admission path. It uses target-oriented cavity replacement for inner-boundary winding. It discovers
source-face edge flips from native-bad volume edge-star incidence, greedily
forms one deterministic connected patch, and delegates exact local cavity
retetrahedralization to the pinned 850/785 owner. It emits O->Skin, D->Fat, or
paired I->(Skin,Fat) candidates plus an independent reconstruction receipt.
The full current O/I/D + wall/all185 source audit and mass/native gates remain
separate.
"""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import permutations
import argparse, hashlib, importlib.util, json, math, sys, time
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ENGINE_PATH=HERE/"connected_boundary_patch_milp_852_oriented_root.py"
ENGINE_SHA="50057ba2275fad473d8ce6db935e64141d49f6ca497a18a8cfa60f38526a6ddf"
PROTECTION_OWNER_PATH=BASE/"coupled-anchor-collapse-780-root"/"coupled_anchor_collapse_795_outer_role.py"
PROTECTION_OWNER_SHA="68124ffcffcc2bf5147de081e50f87739089c3be5c2370be8ae3b27bef4e922a"

def sha(p):
    return hashlib.file_digest(Path(p).open("rb"),"sha256").hexdigest()
def pin(p):
    p=Path(p).resolve()
    return {"path":str(p),"sha256":sha(p)}
def imp(name,path,expected=None):
    path=Path(path).resolve()
    if expected and sha(path)!=expected: raise RuntimeError("pinned source SHA mismatch: "+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    if not spec or not spec.loader: raise RuntimeError("cannot import "+str(path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def fkey(x):
    q=np.ascontiguousarray(np.asarray(x,np.float32).reshape(3))
    if not np.isfinite(q).all():raise ValueError("nonfinite point coordinate")
    return q.view(np.uint32).tobytes()
def facekey(f):return tuple(sorted(map(int,f)))
def sign(f):
    a,b,c=map(int,f);n=int(a>b)+int(a>c)+int(b>c)
    return -1 if n%2 else 1
def chain(faces):
    c=Counter()
    for f in np.asarray(faces,np.int64):c[facekey(f)]+=sign(f)
    return {k:v for k,v in c.items() if v}
def edge_chain(faces):
    c=Counter()
    for a,b,d in np.asarray(faces,np.int64):
        for u,v in ((int(a),int(b)),(int(b),int(d)),(int(d),int(a))):
            c[tuple(sorted((u,v)))]+=1 if u<v else -1
    return {k:v for k,v in c.items() if v}
def independent_flip(old):
    old=np.asarray(old,np.int64)
    if old.shape!=(2,3):raise ValueError("not a face pair")
    shared=sorted(set(map(int,old[0]))&set(map(int,old[1])))
    opposite=sorted((set(map(int,old[0]))|set(map(int,old[1])))-set(shared))
    if len(shared)!=2 or len(opposite)!=2:raise ValueError("faces are not a 2-2 quad")
    goal=edge_chain(old);sol={}
    for a in permutations((opposite[0],opposite[1],shared[0])):
        for b in permutations((opposite[0],opposite[1],shared[1])):
            candidate=np.asarray((a,b),np.int64)
            if edge_chain(candidate)==goal:
                cyc=lambda f:min(tuple(f),tuple(f[1:])+tuple(f[:1]),tuple(f[2:])+tuple(f[:2]))
                sol[tuple(sorted((cyc(candidate[0]),cyc(candidate[1]))))]=candidate
    if len(sol)!=1:raise ValueError("independent oriented flip is ambiguous")
    return next(iter(sol.values())),shared,opposite

ENG=imp("paired_boundary_patch_engine850",ENGINE_PATH,ENGINE_SHA)
P=imp("paired_boundary_patch_protection795",PROTECTION_OWNER_PATH,PROTECTION_OWNER_SHA)
M=ENG.M;T=ENG.T
SOURCE_EXPECTED_Q=1e-5
SOURCE_EXPECTED_V=1e-18

def exact_volume_flux(points,faces):
    pts=np.asarray(points,np.float32);fs=np.asarray(faces,np.int64)
    used=np.unique(fs.reshape(-1));lut={int(v):i for i,v in enumerate(used)}
    ip=[ENG.S.S.ip(x) for x in pts[used]]
    return sum(ENG.S.S.triple(*(ip[lut[int(v)]] for v in f)) for f in fs)
def exact_cell_sum(points,tets):
    pts=np.asarray(points,np.float32);ts=np.asarray(tets,np.int64)
    used=np.unique(ts.reshape(-1));lut={int(v):i for i,v in enumerate(used)}
    ip=[ENG.S.S.ip(x) for x in pts[used]]
    return sum(ENG.S.S.det(*(ip[lut[int(v)]] for v in tet)) for tet in ts)
def measurements(points,tets):
    d,q=M.measurements(np.asarray(points,np.float32),np.asarray(tets,np.int64))
    return np.asarray(d),np.asarray(q)
def qscore(d,q):
    r=np.minimum(np.minimum(np.asarray(q)/SOURCE_EXPECTED_Q,np.asarray(d)/(6*SOURCE_EXPECTED_V)),1.)
    return int(np.count_nonzero(~np.asarray(M.passes(d,q),bool))),float(-np.log(np.maximum(r,1e-300)).sum())
def shape(points,faces):
    xyz=np.asarray(points,np.float32)[np.asarray(faces,np.int64)].astype(np.float64)
    a=xyz[:,1]-xyz[:,0];b=xyz[:,2]-xyz[:,0];c=xyz[:,2]-xyz[:,1]
    area=np.linalg.norm(np.cross(a,b),axis=1)
    edge2=np.maximum.reduce((np.sum(a*a,axis=1),np.sum(b*b,axis=1),np.sum(c*c,axis=1)))
    return np.divide(area,edge2,out=np.zeros_like(area),where=edge2>0)
def target_key(v):
    for k in ("desired_boundary_faces","proposed_current_desired_boundary_faces",
              "proposed_current_desired_boundary_faces_rows","boundary_faces"):
        if k in v:return k
    raise KeyError("volume has no current target boundary")
def check_ref(ref,label):
    if not isinstance(ref,dict) or not ref.get("path") or not ref.get("sha256"):
        raise ValueError(label+" has no path/SHA")
    path=Path(ref["path"]).resolve()
    if not path.is_file() or sha(path)!=str(ref["sha256"]):raise ValueError(label+" SHA mismatch")
    return path
def point_key_index(volume_points):
    """Index every volume coordinate, remembering collisions only for queried keys."""
    lookup={};ambiguous=set()
    for i,p0 in enumerate(np.asarray(volume_points,np.float32)):
        k=fkey(p0)
        if k in lookup:ambiguous.add(k)
        else:lookup[k]=int(i)
    return lookup,ambiguous

def exact_point_map(surface_points,volume_points,label,active_rows=None,required_rows=(),volume_index=None):
    """Map active/required source rows exactly; leave retired unreferenced rows at -1."""
    sp=np.asarray(surface_points,np.float32);vp=np.asarray(volume_points,np.float32)
    if sp.ndim!=2 or sp.shape[1]!=3 or vp.ndim!=2 or vp.shape[1]!=3:
        raise ValueError(label+" point arrays must be Nx3")
    active=np.arange(len(sp),dtype=np.int64) if active_rows is None else np.asarray(active_rows,np.int64).reshape(-1)
    required=np.asarray(required_rows,np.int64).reshape(-1)
    if np.any(active<0) or np.any(active>=len(sp)) or np.any(required<0) or np.any(required>=len(sp)):
        raise ValueError(label+" active/required source row outside point domain")
    selected=np.unique(np.r_[active,required]).astype(np.int64)
    lookup,ambiguous=point_key_index(vp) if volume_index is None else volume_index
    out=np.full(len(sp),-1,np.int64);mapped=[]
    for row in selected:
        k=fkey(sp[int(row)])
        if k in ambiguous:
            raise ValueError(label+" active source point has ambiguous exact volume matches at row "+str(int(row)))
        if k not in lookup:
            raise ValueError(label+" active/required source point has no exact volume match at row "+str(int(row)))
        out[int(row)]=lookup[k];mapped.append(lookup[k])
    if len(set(mapped))!=len(mapped):
        raise ValueError(label+" active/required source-to-volume map is not injective")
    return out

def active_surface_point_rows(surface):
    points=np.asarray(surface["points_world_m_f32"],np.float32)
    faces=np.asarray(surface["faces"],np.int64)
    if points.ndim!=2 or points.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or len(faces)==0:
        raise ValueError("typed source point/face arrays malformed")
    if np.any(faces<0) or np.any(faces>=len(points)):
        raise ValueError("typed source face references point outside source domain")
    return np.unique(faces.reshape(-1)).astype(np.int64)

def active_point_map_controls():
    p=np.asarray(((0,0,0),(1,0,0),(0,1,0),(.25,.25,.25),(.75,.75,.75)),np.float32)
    v=np.asarray((p[0],p[1],p[2],p[3],p[4],p[4]),np.float32)
    mapped=exact_point_map(p,v,"active-row control",active_rows=(0,1,2),required_rows=(3,))
    if not np.array_equal(mapped[:4],np.asarray((0,1,2,3),np.int64)) or mapped[4]!=-1:
        raise AssertionError("inactive retired source row was not ignored or protected row was not mapped")
    def must_reject(sp,vp,active,required=()):
        try:exact_point_map(sp,vp,"negative active-row control",active_rows=active,required_rows=required)
        except ValueError:return True
        return False
    if not must_reject(p,np.vstack((v,p[0])),(0,1,2)):
        raise AssertionError("ambiguous active volume coordinate was accepted")
    duplicate_source=np.vstack((p,p[0]))
    if not must_reject(duplicate_source,v,(0,1,2,5)):
        raise AssertionError("duplicate active source coordinate was accepted")
    if not must_reject(p,v[:4],(0,1,2),(4,)):
        raise AssertionError("missing required protected source point was accepted")
    return {"inactive_unreferenced_unmatched_row_ignored":True,"required_protected_retired_row_mapped":True,
            "ambiguous_active_volume_match_rejected":True,"duplicate_active_source_coordinate_rejected":True,
            "missing_required_protected_point_rejected":True}

def protected_scope(state_path,st,role,volumes,outdir,fixture=False):
    """Derive current-row back/rim/route protections from the pinned 795 owner."""
    sources={r:load_npz(check_ref(st["source_surfaces"][r],"current "+r)) for r in ("O","I","D")}
    if fixture:
        result={}
        for vr,vref in volumes.items():
            result[vr]={"surface_rows":np.empty(0,np.int64),"volume_rows":np.empty(0,np.int64),
                        "report":{"status":"SYNTHETIC_FIXTURE_EMPTY_PROTECTION"}}
        return result
    scopep=Path(P.E.BACK_SCOPE_PATH).resolve()
    if sha(scopep)!=P.E.BACK_SCOPE_SHA:raise ValueError("pinned 412 back scope hash mismatch")
    scope=load_npz(scopep);scope_report_path=scopep.parent/"report.json"
    sr=json.loads(scope_report_path.read_text())
    if sr.get("status")!="PASS_READ_ONLY_BACK412_CURRENT_TYPED_SCOPE_AABB_AUDIT":
        raise ValueError("pinned back-scope report status failed")
    if sr.get("protected_mask_asset",{}).get("sha256")!=sha(scopep):
        raise ValueError("back-scope report does not authenticate mask")
    for r in ("O","I","D"):
        candidate=sr.get("current_sources",{}).get("typed_roles",{}).get(r,{}).get("candidate",{})
        masksha=str(np.asarray(scope.get(r+"_candidate_sha256",np.asarray([""]))).reshape(-1)[0])
        if masksha!=candidate.get("sha256"):raise ValueError("back-scope row domain mismatch for "+r)
    current_refs=st["source_surfaces"]
    ref_by_role={r:sr["current_sources"]["typed_roles"][r]["candidate"] for r in ("O","I","D")}
    rows={}
    for r in ("O","I","D"):
        allow=(r=="I")
        rr,proof=P.map_rows_by_lineage(current_refs[r],ref_by_role[r],sources[r],
            np.asarray(scope[r+"_vertex_rows"],np.int64),r+" current back rows",allow_collapsed=allow)
        expected={"O":177,"I":790,"D":1673}[r]
        if len(rr)!=expected:raise ValueError(f"{r} back-scope count {len(rr)} != {expected}")
        rows[r]=np.unique(rr.astype(np.int64))
    route_doc=json.loads(Path(P.A.PROT).read_text())
    route_by_role={}
    for r in ("O","I","D"):
        surf=sources[r];pts=np.asarray(surf["points_world_m_f32"],np.float32);coord={fkey(x):i for i,x in enumerate(pts)}
        route=[]
        for rec in route_doc.get("surface_routes",[]):
            if rec.get("role")==r:
                k=fkey(rec["target_world_m_f32"])
                if k not in coord:raise ValueError(f"{r} registered route point is absent")
                route.append(int(coord[k]))
        route_by_role[r]=np.unique(np.asarray(route,np.int64))
    rims={}
    for r in ("O","I","D"):
        sf=np.asarray(sources[r]["faces"],np.int64)
        e=np.sort(np.vstack((sf[:,[0,1]],sf[:,[1,2]],sf[:,[2,0]])),axis=1)
        ue,counts=np.unique(e,axis=0,return_counts=True)
        if np.any(counts>2):raise ValueError(r+" source surface is nonmanifold at protected rim derivation")
        rims[r]=np.unique(ue[counts==1].reshape(-1)).astype(np.int64)
    role_to_volumes={"O":("Skin",),"D":("Fat",),"I":("Skin","Fat")}
    protected={}
    # Current and historical I actual-back points are protected in both paired volumes.
    scope_i_rows=np.asarray(scope["I_vertex_rows"],np.int64)
    scope_i_ref=ref_by_role["I"];scope_i_refdata=load_npz(check_ref(scope_i_ref,"reference I back source"))
    for vr,vcache in volumes.items():
        vpoints=np.asarray(vcache["points"],np.float32)
        volume_lut,volume_ambiguous=vcache["coordinate_index"]
        src_roles=[r for r,targets in role_to_volumes.items() if vr in targets]
        surface_rows=set();volume_rows=set();protection_records={}
        for r in src_roles:
            sp=np.asarray(sources[r]["points_world_m_f32"],np.float32)
            pr=np.unique(np.r_[rows[r],rims[r],route_by_role[r]]).astype(np.int64)
            active=active_surface_point_rows(sources[r])
            xmap=exact_point_map(sp,vpoints,vr+" "+r,active_rows=active,required_rows=pr,
                                 volume_index=vcache["coordinate_index"])
            surface_rows.update(map(int,pr))
            volume_rows.update(map(int,xmap[pr]))
            protection_records[r]={"back_rows":rows[r].tolist(),"rim_rows":rims[r].tolist(),"route_rows":route_by_role[r].tolist(),
                "active_source_face_vertices":int(len(active)),"required_protected_source_rows":int(len(pr)),
                "unreferenced_unprotected_source_rows_ignored":int(len(sp)-len(np.union1d(active,pr)))}
        if vr=="Fat":
            # Preserve both current I back rows and the historical exact-coordinate union in Fat.
            i_current=rows["I"]
            i_to_fat=exact_point_map(sources["I"]["points_world_m_f32"],vpoints,"current I/Fat",
                active_rows=active_surface_point_rows(sources["I"]),required_rows=i_current,
                volume_index=vcache["coordinate_index"])
            current_fat=np.unique(i_to_fat[i_current])
            hist=[]
            for r in scope_i_rows:
                k=fkey(scope_i_refdata["points_world_m_f32"][int(r)])
                if k in volume_ambiguous:raise ValueError("historical protected I back point has ambiguous Fat matches")
                if k in volume_lut:hist.append(int(volume_lut[k]))
            hist=np.unique(np.asarray(hist,np.int64))
            actual_fat=np.union1d(current_fat,hist).astype(np.int64)
            if len(actual_fat)!=790:raise ValueError("current+historical actual Fat back union is not exactly 790")
            volume_rows.update(map(int,actual_fat))
            protection_records["actual_fat_back_union"]={"current_rows":current_fat.tolist(),"historical_rows":hist.tolist(),"union_count":int(len(actual_fat))}
        else:
            actual_fat=np.empty(0,np.int64)
        protected[vr]={"surface_rows":np.asarray(sorted(surface_rows),np.int64),
                       "volume_rows":np.asarray(sorted(volume_rows),np.int64),
                       "report":{"role":vr,"sources":protection_records,"back_scope":pin(scopep),
                                 "back_scope_report":pin(scope_report_path),"routes_inventory":pin(P.A.PROT),
                                 "policy":"complete cavity is rejected if any cavity point touches a protected back, rim, or 149-route point"}}
    return protected

def volume_cache(volume_ref):
    vp=check_ref(volume_ref,"volume")
    v=load_npz(vp);p=np.asarray(v["points_world_m"],np.float32);t=np.asarray(v["tetrahedra"],np.int64);mat=np.asarray(v["material_indices"])
    if p.ndim!=2 or p.shape[1]!=3 or t.ndim!=2 or t.shape[1]!=4 or mat.shape!=(len(t),):raise ValueError("malformed current volume")
    if not np.isfinite(p).all():raise ValueError("nonfinite current volume point")
    if chain(T.m.boundary(t))!=chain(np.asarray(v[target_key(v)],np.int64)):raise ValueError("volume target is not its exact oriented boundary")
    det,q=measurements(p,t)
    if np.any(det<=0):raise ValueError("input volume contains nonpositive measured tets")
    return {"ref":volume_ref,"path":vp,"arrays":v,"points":p,"tets":t,"mat":mat,"det":det,"q":q,
            "bad":~np.asarray(M.passes(det,q),bool),"target_key":target_key(v),"target":np.asarray(v[target_key(v)],np.int64),
            "incidence":cell_incidence(t,len(p)),"target_lookup":face_lookup(np.asarray(v[target_key(v)],np.int64)),
            "coordinate_index":point_key_index(p)}
def cell_incidence(tets,npoints):
    t=np.asarray(tets,np.int64);order=np.argsort(t.ravel(),kind="stable");off=np.r_[0,np.cumsum(np.bincount(t.ravel(),minlength=npoints))]
    return order,off
def incident_cells(cache,vertices):
    order,off=cache["incidence"];vv=sorted(set(map(int,vertices)),key=lambda x:int(off[x+1]-off[x]))
    if not vv:return np.empty(0,np.int64)
    rows=np.unique(order[off[vv[0]]:off[vv[0]+1]]//4)
    for v in vv[1:]:
        rows=np.intersect1d(rows,order[off[v]:off[v+1]]//4,assume_unique=True)
        if not len(rows):break
    return rows.astype(np.int64)
def full_edge_star(cache,a,b):
    return incident_cells(cache,(int(a),int(b)))
def face_lookup(faces):
    d=defaultdict(list)
    for i,f in enumerate(np.asarray(faces,np.int64)):d[facekey(f)].append(int(i))
    return d

def discover_patch(role,source,volumes,protected,max_pairs,edgecap):
    sf=np.asarray(source["faces"],np.int64);sp=np.asarray(source["points_world_m_f32"],np.float32)
    if sf.ndim!=2 or sf.shape[1]!=3:raise ValueError("typed source faces malformed")
    edge_faces=defaultdict(list)
    for r,f in enumerate(sf):
        for a,b in ((int(f[0]),int(f[1])),(int(f[1]),int(f[2])),(int(f[2]),int(f[0]))):
            edge_faces[tuple(sorted((a,b)))].append(int(r))
    interior={e:rows for e,rows in edge_faces.items() if len(rows)==2}
    if any(len(rows)>2 for rows in edge_faces.values()):raise ValueError("nonmanifold source edge")
    src_edges=set(interior)
    cross={};reverse={};edge_hits=Counter()
    active_rows=active_surface_point_rows(source)
    for vr,c in volumes.items():
        cross[vr]=exact_point_map(sp,c["points"],role+"/"+vr,active_rows=active_rows,
                                  volume_index=c["coordinate_index"])
        reverse[vr]=np.full(len(c["points"]),-1,np.int64)
        mapped_rows=np.flatnonzero(cross[vr]>=0)
        reverse[vr][cross[vr][mapped_rows]]=mapped_rows
        # Count actual failing cells containing a current source boundary edge.
        for tet in c["tets"][c["bad"]]:
            sr=reverse[vr][tet]
            if np.any(sr<0):continue
            for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
                e=tuple(sorted((int(sr[i]),int(sr[j]))))
                if e in interior:edge_hits[e]+=1
    candidates=[];src_prot=set().union(*(set(map(int,protected[vr]["surface_rows"])) for vr in volumes))
    for edge,fr in interior.items():
        hit=int(edge_hits[edge])
        if hit<=0:continue
        old=sf[np.asarray(fr,np.int64)]
        quads=set(map(int,old.reshape(-1)))
        if quads & src_prot:continue
        try:new,shared,opposite=ENG.flip_faces(old)
        except ValueError:continue
        if (min(opposite),max(opposite)) in src_edges:continue
        # Reject a replacement face that duplicates any face outside this pair.
        retired=set(fr)
        face_set={facekey(f):i for i,f in enumerate(sf) if i not in retired}
        if any(facekey(f) in face_set for f in new):continue
        oldshape=shape(sp,old);newshape=shape(sp,new)
        shape_gain=float(newshape.min()-oldshape.min())
        if shape_gain<=max(1e-15,1e-4*float(oldshape.min())):continue
        if np.any(ENG.J.score(sp,new)<1e-5):continue
        length=float(np.linalg.norm(sp[edge[0]].astype(np.float64)-sp[edge[1]].astype(np.float64)))
        if length>edgecap:continue
        stars={};valid=True;bad_hits_by_volume={}
        for vr,c in volumes.items():
            a,b=map(int,cross[vr][np.asarray(edge,np.int64)])
            rr=full_edge_star(c,a,b);stars[vr]=rr
            if not len(rr):valid=False;break
            verts=np.unique(c["tets"][rr].ravel())
            if len(rr)>24 or len(verts)>16 or len(np.unique(c["mat"][rr]))!=1:valid=False;break
            if set(map(int,verts)) & set(map(int,protected[vr]["volume_rows"])):valid=False;break
            bad_hits_by_volume[vr]=int(np.count_nonzero(c["bad"][rr]))
            # Every retired source target face must exist exactly once in every affected volume.
            tl=c["target_lookup"]
            for f in old:
                vf=cross[vr][f]
                if len(tl.get(facekey(vf),[]))!=1:valid=False;break
            if not valid:break
        if not valid:continue
        candidates.append({"edge":edge,"face_rows":tuple(fr),"old_faces":old.copy(),"new_faces":new.copy(),
                           "quad":quads,"bad_edge_hit":hit,"bad_by_volume":bad_hits_by_volume,
                           "length_m":length,"shape_before":oldshape,"shape_after":newshape,
                           "shape_gain":shape_gain,"full_edge_star_rows":{r:stars[r].tolist() for r in stars},
                           "source_shared_edge":list(shared),"source_opposite_vertices":list(opposite)})
    candidates.sort(key=lambda c:(-c["bad_edge_hit"],-c["shape_gain"],c["length_m"],c["edge"],c["face_rows"]))
    selected=[];used_rows=set();patch_vertices=set()
    for c in candidates:
        rows=set(c["face_rows"])
        if rows&used_rows:continue
        if selected and not (c["quad"]&patch_vertices):continue
        selected.append(c);used_rows|=rows;patch_vertices|=c["quad"]
        if len(selected)>=max_pairs:break
    # Preserve only prefixes that remain a valid connected manifold source patch.
    while selected:
        pairs=[list(c["face_rows"]) for c in selected]
        if ENG.connect_pairs(pairs,sf):
            test=sf.copy()
            for c in selected:test[list(c["face_rows"])]=c["new_faces"]
            try:
                ENG.topology_sig(test)
                return {"pairs":pairs,"selected":selected,"candidate_count":len(candidates),
                        "edge_hit_count":len(edge_hits),"source_point_to_volume_rows":cross,
                        "surface_topology_before":ENG.topology_sig(sf),"surface_topology_after":ENG.topology_sig(test)}
            except ValueError:pass
        selected.pop()
    raise ValueError("no deterministic connected source patch satisfies local incidence/protection gates")

def write_protection_masks(out,role,protection):
    out.mkdir(parents=True,exist_ok=False)
    paths={}
    for vr,x in protection.items():
        path=out/f"protected-{role}-surface-volume-{vr}.npz"
        np.savez_compressed(path,protected_surface_point_rows=x["surface_rows"],
            protected_volume_point_rows=x["volume_rows"],protected_volume_cell_rows=np.empty(0,np.int64))
        report_path=out/f"protected-{role}-surface-volume-{vr}.json"
        report={"schema":"h706.paired-connected-boundary-patch-protection.v1","status":"PASS_DERIVED_CURRENT_PROTECTED_ROWS",
                "role":role,"volume_role":vr,"protected_surface_point_rows":int(len(x["surface_rows"])),
                "protected_volume_point_rows":int(len(x["volume_rows"])),"protection":x["report"],
                "mask":pin(path),"physical_admission":False,"native_steps":0}
        report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        paths[vr]={"path":str(path),"sha256":sha(path),"report":pin(report_path)}
    return paths

def get_source_typed(surface):
    a=np.asarray(surface["quality_star_patch_union_offsets"],np.int64)
    b=np.asarray(surface["face_ancestry_offsets"],np.int64)
    if len(a)!=len(surface["faces"])+1 or len(b)!=len(surface["faces"])+1:raise ValueError("typed ancestry CSR shape mismatch")
    return a,b

def independent_replay(parent_surface,parent_volume,proposal_surface,proposal_volume,patch_doc,engine_report,role):
    """Reconstruct a proposal from exact input pins and operation rows; do not trust producer readback."""
    sf0=np.asarray(parent_surface["faces"],np.int64);pts=np.asarray(parent_surface["points_world_m_f32"],np.float32)
    sf1=np.asarray(proposal_surface["faces"],np.int64)
    if not np.array_equal(np.asarray(parent_surface["points_world_m_f32"]).view(np.uint32),pts.view(np.uint32)):
        raise ValueError("parent source F32 point representation invalid")
    pairs=[tuple(map(int,x)) for x in patch_doc["patches"][0]["face_pairs"]]
    expected=sf0.copy();changed=[]
    if len(set(r for pp in pairs for r in pp))!=2*len(pairs):raise ValueError("operation face rows overlap")
    for pp in pairs:
        if min(pp)<0 or max(pp)>=len(sf0):raise ValueError("operation source row out of range")
        new,_,_=independent_flip(sf0[list(pp)]);expected[list(pp)]=new;changed.extend(pp)
    if not np.array_equal(sf1,expected):raise ValueError("independent source-face reconstruction mismatch")
    if ENG.topology_sig(sf0)!=ENG.topology_sig(sf1):raise ValueError("source topology signature changed")
    # Complete typed unions and face ancestry are reconstructed from parent rows.
    union=set();anc=set()
    for r in changed:
        union.update(tuple(map(int,x)) for x in ENG.A.rows_union(parent_surface,int(r)))
        anc.update(tuple(map(int,x)) for x in ENG.A.rows_anc(parent_surface,int(r)))
    if not union or not anc:raise ValueError("source parent ancestry unexpectedly empty")
    qf=np.asarray(proposal_surface["quality_star_face_output_to_input_row"],np.int64)
    if np.any(qf[np.asarray(changed,np.int64)]!=-1):raise ValueError("changed source faces retain exact-parent row claims")
    uoff=np.asarray(proposal_surface["quality_star_patch_union_offsets"],np.int64)
    aoff=np.asarray(proposal_surface["face_ancestry_offsets"],np.int64)
    for r in changed:
        lo,hi=map(int,uoff[r:r+2]);got={tuple(int(proposal_surface[k][j]) for k in ENG.TUN) for j in range(lo,hi)}
        if got!=union:raise ValueError("changed source face union is incomplete")
        lo,hi=map(int,aoff[r:r+2]);got={tuple(int(proposal_surface[k][j]) for k in ENG.ANC) for j in range(lo,hi)}
        if got!=anc:raise ValueError("changed source ancestry is incomplete")
    changed_set=set(changed)
    for r in range(len(sf0)):
        if r in changed_set:continue
        if qf[r]!=np.asarray(parent_surface["quality_star_face_output_to_input_row"],np.int64)[r]:raise ValueError("retained source face map changed")
        if ENG.A.rows_union(proposal_surface,r)!=ENG.A.rows_union(parent_surface,r) or ENG.A.rows_anc(proposal_surface,r)!=ENG.A.rows_anc(parent_surface,r):
            raise ValueError("retained source face ancestry changed")
    vp=np.asarray(parent_volume["points_world_m"],np.float32);tp=np.asarray(proposal_volume["points_world_m"],np.float32)
    if not np.array_equal(vp.view(np.uint32),tp.view(np.uint32)):raise ValueError("volume points changed")
    tin=np.asarray(parent_volume["tetrahedra"],np.int64);tinmat=np.asarray(parent_volume["material_indices"])
    tout=np.asarray(proposal_volume["tetrahedra"],np.int64);toutmat=np.asarray(proposal_volume["material_indices"])
    if toutmat.shape!=(len(tout),):raise ValueError("output material domain malformed")
    origin=np.asarray(proposal_volume["output_cell_to_input_cell"],np.int64)
    if origin.shape!=(len(tout),):raise ValueError("output cell map malformed")
    retained=np.flatnonzero(origin>=0);newrows=np.flatnonzero(origin<0)
    if len(retained)+len(newrows)!=len(tout):raise ValueError("output cell partition malformed")
    if not np.array_equal(tout[retained],tin[origin[retained]]) or not np.array_equal(toutmat[retained],tinmat[origin[retained]]):
        raise ValueError("retained ordered cells/materials changed")
    if not np.array_equal(np.asarray(proposal_volume["retained_input_cell_rows"],np.int64),np.flatnonzero(np.asarray(proposal_volume["input_cell_to_output_cell"])>=0)):
        raise ValueError("retained input cell row map mismatch")
    oldrows=np.asarray(engine_report["edge_star_cell_rows"],np.int64)
    if len(np.unique(oldrows))!=len(oldrows) or np.any(oldrows<0) or np.any(oldrows>=len(tin)):raise ValueError("reported cavity row domain invalid")
    # Independently reconstruct the exact union of complete edge stars.
    cross=exact_point_map(pts,vp,role+" replay",active_rows=np.unique(sf0.reshape(-1)))
    expected_rows=set()
    for pp in pairs:
        oldfaces=sf0[list(pp)];shared=set(map(int,oldfaces[0]))&set(map(int,oldfaces[1]))
        if len(shared)!=2:raise ValueError("face-pair shared edge invalid")
        va,vb=map(int,cross[np.asarray(sorted(shared),np.int64)])
        star=np.flatnonzero(np.any(tin==va,axis=1)&np.any(tin==vb,axis=1))
        expected_rows.update(map(int,star))
    if set(map(int,oldrows))!=expected_rows:raise ValueError("edge-star cavity is incomplete or overinclusive")
    if np.any(origin[newrows]!=-1):raise ValueError("new output cell origin must be -1")
    newt=tout[newrows]
    if not np.array_equal(newt,np.asarray(proposal_volume["replacement_tetrahedra"],np.int64)):raise ValueError("replacement tetrahedron payload mismatch")
    roff=np.asarray(proposal_volume["replacement_parent_offsets"],np.int64);rrows=np.asarray(proposal_volume["replacement_parent_cell_rows"],np.int64)
    if roff.shape!=(len(newt)+1,) or roff[0]!=0 or roff[-1]!=len(rrows):raise ValueError("replacement parent CSR malformed")
    for j in range(len(newt)):
        lo,hi=map(int,roff[j:j+2])
        if set(map(int,rrows[lo:hi]))!=expected_rows:raise ValueError("replacement tetrahedron lacks full cavity parent union")
    # Exact signs and local cavity chain/flux.
    ip=[ENG.S.S.ip(x) for x in vp]
    oldexact=[ENG.S.S.det(*(ip[int(i)] for i in row)) for row in tin[oldrows]]
    newexact=[ENG.S.S.det(*(ip[int(i)] for i in row)) for row in newt]
    if not oldexact or min(oldexact)<=0 or not newexact or min(newexact)<=0:raise ValueError("exact cavity orientation failed")
    b0=np.asarray(T.m.boundary(tin[oldrows]),np.int64);b1=np.asarray(T.m.boundary(newt),np.int64)
    # Rebuild current target face rows and independently prove each retired target
    # facet, with its actual boundary orientation, belongs to the complete cavity.
    tk=target_key(parent_volume);t0=np.asarray(parent_volume[tk],np.int64);t1=np.asarray(proposal_volume[tk],np.int64)
    b0chain=chain(b0);orientation_relations=[];vtarget=t0.copy();lut=face_lookup(t0)
    for pp in pairs:
        oldsrc=sf0[list(pp)];newsrc=expected[list(pp)]
        oldt=[];rels=[]
        for f in oldsrc:
            vm=cross[f];hits=lut.get(facekey(vm),[])
            if len(hits)!=1:raise ValueError("source parent face has no unique physical target row")
            tr=int(hits[0]);old_face=t0[tr].copy();oldt.append(old_face);rel=sign(old_face)*sign(vm);rels.append(rel)
            if b0chain.get(tuple(sorted(map(int,old_face))),0)!=sign(old_face):
                raise ValueError("retired physical target facet is absent or has opposite sign in complete old cavity boundary")
        if len(set(rels))!=1:raise ValueError("parent target orientation relation is inconsistent")
        orientation_relations.append(int(rels[0]))
        newtarget=cross[newsrc]
        if rels[0]<0:newtarget=newtarget[:,[0,2,1]]
        for j,tr in enumerate([lut[facekey(cross[f])] for f in oldsrc]):
            vtarget[int(tr[0])]=newtarget[j]
    if chain(T.m.boundary(tout))!=chain(t1) or chain(t1)!=chain(vtarget):
        raise ValueError("complete output target boundary reconstruction failed")
    oldsum=sum(oldexact);newsum=sum(newexact)
    local_delta=newsum-oldsum
    old_flux=exact_volume_flux(vp,b0);new_flux=exact_volume_flux(vp,b1)
    if old_flux!=oldsum or new_flux!=newsum:raise ValueError("cavity exact volume does not equal its oriented boundary flux")
    global_delta=exact_volume_flux(vp,t1)-exact_volume_flux(vp,t0)
    if global_delta!=local_delta:raise ValueError("local exact delta does not explain global target delta")
    # Whole current volume positive and native thresholds unchanged.
    dd,qq=measurements(vp,tout)
    if np.any(dd<=0):raise ValueError("output has nonpositive measured tetrahedra")
    beforebad,beforedef=qscore(*measurements(vp,tin));afterbad,afterdef=qscore(dd,qq)
    if afterbad>beforebad:raise ValueError("global native bad count regressed")
    # Independent complete target check is geometry/chain only; all185 remains external.
    return {"status":"PASS_INDEPENDENT_CONNECTED_PATCH_REPLAY","role":role,"input_volume":pin(parent_volume["_path"]),
       "output_volume":pin(proposal_volume["_path"]),"input_surface":pin(parent_surface["_path"]),
       "output_surface":pin(proposal_surface["_path"]),"face_pairs":pairs,"retired_volume_cells":len(oldrows),
       "new_volume_cells":len(newt),"source_changed_face_rows":sorted(set(changed)),
       "exact_local_det6_delta":str(local_delta),"exact_global_target_flux_delta_det6":str(global_delta),
       "exact_cavity_boundary_flux":True,"complete_volume_boundary":True,"all_output_cells_measured_positive":True,
       "source_to_target_orientation_relations":orientation_relations,
       "retired_target_facets_verified_on_oriented_cavity_boundary":True,
       "native_bad_before":beforebad,"native_bad_after":afterbad,"native_deficit_before":beforedef,"native_deficit_after":afterdef,
       "all_volume_points_unchanged":True,"full_typed_parent_unions_replayed":True,"full_face_ancestry_replayed":True,
       "physical_admission":False,"native_steps":0,"external_all185_audit":"NOT_RUN"}

def discover_patch_for_state(role,st,volumes,protections,max_pairs,edgecap):
 surface_ref=st["source_surfaces"][role];surface=load_npz(check_ref(surface_ref,"source surface "+role))
 return surface,discover_patch(role,surface,volumes,protections,max_pairs,edgecap)

def copy_volume_metadata(source_volume_path,dest_path,role_to_source,role):
    data=load_npz(source_volume_path)
    for r,ref in role_to_source.items():
        data[f"typed_{r}_candidate_path"]=np.asarray([str(Path(ref["path"]).resolve())])
        data[f"typed_{r}_candidate_sha256"]=np.asarray([str(ref["sha256"])])
    data["paired_boundary_patch_metadata_status"]=np.asarray(["GEOMETRY_PROPOSAL_TYPED_SOURCE_REBOUND_EXTERNAL_TARGET397_PENDING"])
    np.savez_compressed(dest_path,**data)
    return pin(dest_path)

def run_state(args,fixture=False):
    tic=time.monotonic();state_path=Path(args.state).resolve();root=json.loads(state_path.read_text())
    st=root.get("current_verified_state_20261031",root)
    if st.get("physical_admission") is not False or int(st.get("native_steps",-1))!=0:
        raise ValueError("input state is not source-only construction geometry")
    for k in ("source_surfaces","Skin_current","Fat_current"):
        if k not in st:raise ValueError("state lacks "+k)
    if args.role not in ("O","I","D"):raise ValueError("role must be O, I, or D")
    affected=("Skin",) if args.role=="O" else ("Fat",) if args.role=="D" else ("Skin","Fat")
    volumes={}
    for vr in affected:
        volumes[vr]=volume_cache(st[vr+"_current"])
    protections=protected_scope(state_path,st,args.role,volumes,Path(args.output)/"protection",fixture=fixture)
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError("output directory must be fresh")
    out.mkdir(parents=True)
    if getattr(args,"preflight_only",False):
        maskrefs=write_protection_masks(out/"protection",args.role,protections)
        source=load_npz(check_ref(st["source_surfaces"][args.role],"source surface"))
        active=active_surface_point_rows(source)
        inactive=np.setdiff1d(np.arange(len(source["points_world_m_f32"]),dtype=np.int64),active,assume_unique=True)
        report={"schema":"h706.paired-connected-boundary-patch-readonly-preflight.v1",
            "status":"PASS_READONLY_ACTIVE_SOURCE_POINT_MAP_PREFLIGHT","parent_state":pin(state_path),
            "role":args.role,"affected_volume_roles":list(affected),
            "source_surface":pin(check_ref(st["source_surfaces"][args.role],"source surface")),
            "source_active_face_vertex_count":int(len(active)),"inactive_unreferenced_source_point_count":int(len(inactive)),
            "inactive_unreferenced_points_mapped":False,
            "protected_scope":{vr:protections[vr]["report"] for vr in protections},
            "volumes":{vr:{"path":str(c["path"]),"sha256":sha(c["path"]),"cells":int(len(c["tets"])),
              "bad_cells":int(np.count_nonzero(c["bad"])),"target_faces":int(len(c["target"])),
              "protected_surface_point_rows":int(len(protections[vr]["surface_rows"])),
              "protected_volume_point_rows":int(len(protections[vr]["volume_rows"]))} for vr,c in volumes.items()},
            "discovery":"NOT_RUN_READ_ONLY_MAP_PREPARATION_ONLY",
            "protection_masks":maskrefs,"protection_owner":pin(PROTECTION_OWNER_PATH),"engine":pin(ENGINE_PATH),
            "geometry_mutated":False,"solver_invoked":False,"external_all185":"NOT_RUN","mass_mapping":"NOT_RUN",
            "physical_admission":False,"native_steps":0,"elapsed_s":time.monotonic()-tic}
        rp=out/"preflight-report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        report["preflight_report"]=pin(rp)
        print(json.dumps(report,indent=2,sort_keys=True),flush=True);return report
    surface,discovery=discover_patch_for_state(args.role,st,volumes,protections,args.max_pairs,args.maximum_edge_m)
    # The protection directory is created after output root exists.
    # Recreate masks in its canonical, pinned per-volume schema.
    maskdir=out/"protection";maskdir.mkdir()
    maskrefs={}
    for vr,x in protections.items():
        mp=maskdir/f"protected-{args.role}-{vr}.npz"
        np.savez_compressed(mp,protected_surface_point_rows=x["surface_rows"],protected_volume_point_rows=x["volume_rows"],protected_volume_cell_rows=np.empty(0,np.int64))
        mr=maskdir/f"protected-{args.role}-{vr}.json"
        mr.write_text(json.dumps({"status":"PASS_DERIVED_CURRENT_PROTECTED_ROWS","role":args.role,"volume_role":vr,
             "surface_rows":int(len(x["surface_rows"])),"volume_rows":int(len(x["volume_rows"])),"evidence":x["report"],"mask":pin(mp)},indent=2,sort_keys=True)+"\n")
        maskrefs[vr]=pin(mp)
    # Produce at most one connected patch. Shrink deterministically if a larger
    # union cavity is not feasible; preserve all failed trial logs.
    selected=discovery["selected"]
    role_dir=out/"engine-attempts";role_dir.mkdir()
    accepted=None;attempts=[]
    role_sources={r:dict(st["source_surfaces"][r]) for r in ("O","I","D")}
    for n in range(len(selected),0,-1):
        use=selected[:n]
        patch_doc={"schema":"h706.connected-surface-patch-spec.v1","role":args.role,
          "source_surface_parent":st["source_surfaces"][args.role],
          "patches":[{"face_pairs":[list(x["face_rows"]) for x in use],
                      "selection":"deterministic rank by current bad-cell edge-star incidence, source face-shape gain, edge length"}]}
        patchpath=out/f"patch-spec-{n:02d}.json";patchpath.write_text(json.dumps(patch_doc,indent=2,sort_keys=True)+"\n")
        attempt={"pair_count":n,"patch_spec":pin(patchpath),"volume_attempts":{}}
        engine_outputs={};failed=None
        for vr,c in volumes.items():
            vpdir=role_dir/f"attempt-{n:02d}-{vr}";vpdir.mkdir()
            args850=argparse.Namespace(volume_npz=c["path"],surface_npz=check_ref(st["source_surfaces"][args.role],"current source"),
              patches=patchpath,protected_npz=Path(maskrefs[vr]["path"]),role=args.role,output=vpdir/"proposal",
              solve_seconds=args.solve_seconds,maximum_old_cells=args.maximum_old_cells,maximum_points=args.maximum_points,
              maximum_local_volume_fraction=args.maximum_local_volume_fraction,
              maximum_global_volume_fraction=args.maximum_global_volume_fraction)
            try:
                rep=ENG.run(args850)
                engine_outputs[vr]={"report":rep,"volume":Path(rep["volume_candidate"]["path"]),"surface":Path(rep["typed_surface_candidate"]["path"])}
                attempt["volume_attempts"][vr]={"status":"PASS_ENGINE_LOCAL","report":pin(vpdir/"proposal/report.json")}
            except Exception as ex:
                failed=f"{vr}: {type(ex).__name__}: {ex}"
                (vpdir/"failed-attempt.txt").write_text(failed+"\n")
                attempt["volume_attempts"][vr]={"status":"FAIL_ENGINE_LOCAL","failure":failed,"failure_artifact":pin(vpdir/"failed-attempt.txt")}
                break
        if failed is None:
            # I uses the same current typed source parent and same face patch in both volumes.
            if args.role=="I":
                first=load_npz(engine_outputs["Skin"]["surface"]);second=load_npz(engine_outputs["Fat"]["surface"])
                if set(first)!=set(second):failed="paired I typed candidates have different array schemas"
                else:
                    for k in first:
                        if not np.array_equal(first[k],second[k]):failed=f"paired I typed candidate differs at {k}";break
                if failed is None:
                    skin_delta=int(engine_outputs["Skin"]["report"]["global_exact_signed_volume_delta_det6"])
                    fat_delta=int(engine_outputs["Fat"]["report"]["global_exact_signed_volume_delta_det6"])
                    if skin_delta+fat_delta!=0:failed="paired I exact signed-volume flux does not cancel"
                    else:
                        before_bad=sum(int(engine_outputs[v]["report"]["native_bad_before"]) for v in ("Skin","Fat"))
                        after_bad=sum(int(engine_outputs[v]["report"]["native_bad_after"]) for v in ("Skin","Fat"))
                        before_def=sum(float(engine_outputs[v]["report"]["log_deficit_before"]) for v in ("Skin","Fat"))
                        after_def=sum(float(engine_outputs[v]["report"]["log_deficit_after"]) for v in ("Skin","Fat"))
                        if after_bad>before_bad or (after_bad==before_bad and after_def>=before_def-max(1e-8,1e-6*before_def)):
                            failed="paired I combined native quality did not strictly improve without regressions"
            else:
                before_bad=engine_outputs[affected[0]]["report"]["native_bad_before"]
                after_bad=engine_outputs[affected[0]]["report"]["native_bad_after"]
                before_def=engine_outputs[affected[0]]["report"]["log_deficit_before"]
                after_def=engine_outputs[affected[0]]["report"]["log_deficit_after"]
            if failed is None:
                try:
                    source_out=engine_outputs[affected[0]]["surface"]
                    source_arrays=load_npz(source_out)
                    for vr,c in volumes.items():
                        independent_replay(surface|{"_path":check_ref(st["source_surfaces"][args.role],"input source")},
                          c["arrays"]|{"_path":c["path"]},source_arrays|{"_path":source_out},
                          load_npz(engine_outputs[vr]["volume"])|{"_path":engine_outputs[vr]["volume"]},patch_doc,
                          engine_outputs[vr]["report"],args.role)
                except Exception as ex:
                    failed=f"independent replay: {type(ex).__name__}: {ex}"
                    attempt["independent_replay_failure"]=failed
        if failed is not None:
            attempt["status"]="FAIL_PATCH_ATTEMPT";attempt["failure"]=failed
            attempts.append(attempt)
            (out/f"attempt-{n:02d}-failure.json").write_text(json.dumps(attempt,indent=2,sort_keys=True)+"\n")
            continue
        accepted={"pair_count":n,"patch_spec":pin(patchpath),"engine_outputs":engine_outputs,
                  "source_surface_arrays":source_arrays,"quality":{"bad_before":before_bad,"bad_after":after_bad,
                  "deficit_before":before_def,"deficit_after":after_def}}
        attempt["status"]="PASS_ENGINE_AND_INDEPENDENT_REPLAY";attempt["independent_replay"]="PASS"
        attempts.append(attempt);(out/f"attempt-{n:02d}-pass.json").write_text(json.dumps(attempt,indent=2,sort_keys=True)+"\n")
        break
    if accepted is None:raise ValueError("all deterministic connected patch prefixes failed; inspect preserved attempt reports")
    # One typed surface is shared by all changed roles; write a stable copy and
    # update the state and each touched volume's typed-source aliases.
    source_path=out/f"candidate-{args.role}-typed-current.npz"
    np.savez_compressed(source_path,**accepted["source_surface_arrays"]);source_pin=pin(source_path)
    role_sources[args.role]=source_pin
    outstate=json.loads(json.dumps(st))
    outstate["source_surfaces"]=role_sources
    volume_output_refs={}
    for vr in affected:
        original=accepted["engine_outputs"][vr]["volume"]
        meta_path=out/f"{vr.lower()}-typed-source-rebound.npz"
        volume_output_refs[vr]=copy_volume_metadata(original,meta_path,role_sources,args.role)
        voldata=load_npz(meta_path)
        # Readback metadata must point to exact role snapshots.
        for rr,ref in role_sources.items():
            if str(np.asarray(voldata[f"typed_{rr}_candidate_path"]).reshape(-1)[0])!=str(Path(ref["path"]).resolve()):
                raise ValueError("volume typed-source path readback mismatch")
            if str(np.asarray(voldata[f"typed_{rr}_candidate_sha256"]).reshape(-1)[0])!=ref["sha256"]:
                raise ValueError("volume typed-source SHA readback mismatch")
        prior=outstate[vr+"_current"]
        outstate[vr+"_current"]={**prior,**volume_output_refs[vr],
           "cells":int(len(voldata["tetrahedra"])),"native_quality_failures":int(accepted["engine_outputs"][vr]["report"]["native_bad_after"]),
           "nonpositive":0,"boundary_source_binding_status":"PENDING_CURRENT_TARGET397_METADATA_REBIND"}
    # The previous current-context proof is stale after these face updates.
    outstate.pop("exact_context",None);outstate.pop("independent_replay",None);outstate.pop("source_evidence",None)
    outstate.update({"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
       "source_state":pin(state_path),"producer":pin(Path(__file__)),
       "connected_boundary_patch":{"role":args.role,"face_pairs":accepted["pair_count"],
         "patch_spec":accepted["patch_spec"],"selection_evidence":{"candidate_count":discovery["candidate_count"],
         "edge_hit_count":discovery["edge_hit_count"],"selected_pairs":[list(x["face_rows"]) for x in selected[:accepted["pair_count"]]]},
         "input_protection_masks":maskrefs,"attempts":attempts,
         "source_surface_candidate":source_pin,"volume_candidate_outputs":volume_output_refs,
         "internal_independent_replay":"PASS","external_current_OID_both_walls_all185":"REQUIRED_NOT_RUN",
         "target_metadata_binding":"PENDING_397_REBIND","native_thresholds_unchanged":True,
         "regional_volume_fraction_caps":{"local":args.maximum_local_volume_fraction,"global":args.maximum_global_volume_fraction}},
       "status":"SOURCE_ONLY_CONNECTED_BOUNDARY_PATCH_PROPOSAL_PENDING_EXTERNAL_AUDIT"})
    state_path_out=out/"state.json";state_path_out.write_text(json.dumps(outstate,indent=2,sort_keys=True)+"\n")
    report={"schema":"h706.paired-connected-boundary-patch-driver.v1","status":"PASS_PROPOSAL_AND_INDEPENDENT_REPLAY_PENDING_EXTERNAL_AUDIT",
      "parent_state":pin(state_path),"role":args.role,"affected_volume_roles":list(affected),
      "source_surface_parent":pin(check_ref(st["source_surfaces"][args.role],"source parent")),
      "source_surface_candidate":source_pin,"patch_spec":accepted["patch_spec"],
      "candidate_pair_count":accepted["pair_count"],"candidate_edge_count":discovery["candidate_count"],
      "bad_cell_incidence_edge_count":discovery["edge_hit_count"],"quality":accepted["quality"],
      "volume_candidates":volume_output_refs,"attempts":attempts,
      "protection_masks":maskrefs,"protection_owner":pin(PROTECTION_OWNER_PATH),
      "engine":pin(ENGINE_PATH),"native_thresholds_unchanged":True,"source_points_bit_identical":True,
      "volume_points_bit_identical":True,"surface_topology_signature_unchanged":True,
      "full_typed_parent_union_and_ancestry":True,"exact_output_boundary_and_signs_independently_replayed":True,
      "current_OID_walls_all185":"REQUIRED_SEPARATE_FINAL_GATE_NOT_RUN","mass_mapping":"NOT_RUN",
      "physical_admission":False,"native_steps":0,"elapsed_s":time.monotonic()-tic,
      "output_state":pin(state_path_out)}
    rp=out/"driver-report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True),flush=True)
    return report

def _fixture_inputs(out):
    out.mkdir(parents=True,exist_ok=False);inputs=out/"inputs";inputs.mkdir()
    p=np.asarray(((0,0,0),(1,0,0),(.2,.2,0),(.8,-.2,0),(.5,0,1e-7)),np.float32)
    t=M.orient(p,((0,1,2,4),(1,0,3,4)))
    faces=np.asarray(T.m.boundary(t),np.int64)
    surf=ENG.fixture_surface(p,faces)
    sp=inputs/"typed-surface.npz";np.savez_compressed(sp,**surf)
    vp=inputs/"base-volume.npz";np.savez_compressed(vp,points_world_m=p,tetrahedra=t,
       material_indices=np.full(len(t),16,np.int32),boundary_faces=faces,desired_boundary_faces=faces,
       input_node_to_output_node=np.arange(len(p),dtype=np.int64),output_node_to_input_node=np.arange(len(p),dtype=np.int64),
       output_point_to_input_point_row=np.arange(len(p),dtype=np.int64),source_I_vertex_quotient=np.arange(len(p),dtype=np.int64),
       output_point_parent_offsets=np.zeros(len(p)+1,np.int64),output_point_parent_input_cell_rows=np.empty(0,np.int64))
    refS=pin(sp);refV=pin(vp)
    state={"physical_admission":False,"native_steps":0,"source_surfaces":{r:refS for r in ("O","I","D")},
           "Skin_current":{**refV,"cells":len(t),"native_quality_failures":2},
           "Fat_current":{**refV,"cells":len(t),"native_quality_failures":2},
           "status":"SYNTHETIC_SOURCE_ONLY"}
    statepath=inputs/"state.json";statepath.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n")
    return statepath

def fixture(outdir):
    out=Path(outdir).resolve();out.mkdir(parents=True,exist_ok=False);inputs=out/"input-state";statepath=_fixture_inputs(inputs)
    # Run the actual paired-state CLI path for D then I. All generated candidates
    # remain under this fresh synthetic fixture directory.
    reports=[]
    for role in ("D","I"):
        a=argparse.Namespace(state=statepath,role=role,output=out/(role+"-run"),max_pairs=1,maximum_edge_m=1.,
            solve_seconds=3.,maximum_old_cells=16,maximum_points=12,maximum_local_volume_fraction=.01,
            maximum_global_volume_fraction=1e-4)
        reports.append(run_state(a,fixture=True))
    receipt={"schema":"h706.paired-connected-boundary-patch-fixture.v1","status":"PASS_PAIRED_STATE_PATCH_PRODUCER_AND_INDEPENDENT_REPLAY",
      "producer":pin(Path(__file__)),"engine":pin(ENGINE_PATH),"input_state":pin(statepath),
      "roles":[{"role":r["role"],"report":pin(out/(r["role"]+"-run/driver-report.json")),
                "state":r["output_state"]} for r in reports],
      "D_source_to_Fat_exact_replay":True,"I_source_to_Skin_and_Fat_exact_replay":True,
      "paired_I_exact_signed_volume_delta_det6":"0","source_geometry_pins_unchanged":True,
      "active_point_map_controls":active_point_map_controls(),
      "thresholds_unchanged":True,"external_all185":"NOT_RUN","physical_admission":False,"native_steps":0}
    (out/"fixture-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state",type=Path);ap.add_argument("--role",choices=("O","I","D"));ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--max-pairs",type=int,default=3);ap.add_argument("--maximum-edge-m",type=float,default=.005)
    ap.add_argument("--solve-seconds",type=float,default=5.);ap.add_argument("--maximum-old-cells",type=int,default=24)
    ap.add_argument("--maximum-points",type=int,default=16);ap.add_argument("--maximum-local-volume-fraction",type=float,default=.01)
    ap.add_argument("--maximum-global-volume-fraction",type=float,default=1e-4);ap.add_argument("--fixture",action="store_true")
    ap.add_argument("--preflight-only",action="store_true",help="derive current protections/maps without invoking cavity solver")
    a=ap.parse_args()
    if a.fixture:
        if a.output.exists():ap.error("--output must be a fresh path")
        print(json.dumps(fixture(a.output),indent=2,sort_keys=True),flush=True);return
    if a.state is None or a.role is None:ap.error("--state and --role required outside --fixture")
    if not (1<=a.max_pairs<=8 and 1<=a.maximum_old_cells<=128 and 4<=a.maximum_points<=24):ap.error("patch/cavity caps out of range")
    if a.maximum_edge_m<=0 or a.solve_seconds<=0:ap.error("edge cap and solver budget must be positive")
    run_state(a,fixture=False)
if __name__=="__main__":main()

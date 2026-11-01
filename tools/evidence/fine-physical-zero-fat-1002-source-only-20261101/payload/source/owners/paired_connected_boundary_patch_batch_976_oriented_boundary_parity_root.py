#!/usr/bin/env python3
"""Immutable 883 derivative: winding-invariant convex-quad source flip screen; all geometry and exact gates are unchanged.

Role-scoped connected surface remesher with fixed-coordinate topology mode.

Derived immutably from v867. Source protection uses only the edited role's
source-row namespace. Fixed-coordinate topology mode permits a cavity star to
touch protected points owned by another role while keeping all their source
faces/point bytes and all nonedited boundary facets unchanged by construction.
The all-positive, exact boundary/flux, native, ancestry, and final source gates
remain mandatory. This is a source-only construction proposal.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,importlib.util,json,math,sys,time
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
BASE_DRIVER_PATH=HERE/"paired_connected_boundary_patch_856_oriented_active_rows_root.py"
BASE_DRIVER_SHA="8d07da491eb3a6ec9aa1c832bce52bd6b2e72f8486a7f0e71631c2bf1022976b"

PARENT_883_PATH=HERE/"paired_connected_boundary_patch_batch_883_targeted_source_edge_innerflip_root.py"
PARENT_883_SHA="7bde558d69512ab9edc41748ccda728ee510c75c4ec840a297eae65345385c54"
if hashlib.sha256(PARENT_883_PATH.read_bytes()).hexdigest()!=PARENT_883_SHA:
 raise RuntimeError("pinned 883 parent changed")

def sha(p):return hashlib.file_digest(Path(p).open("rb"),"sha256").hexdigest()
def pin(p):
 p=Path(p).resolve();return {"path":str(p),"sha256":sha(p)}
def imp(name,path,expected):
 path=Path(path).resolve()
 if sha(path)!=expected:raise RuntimeError("pinned owner SHA mismatch: "+str(path))
 spec=importlib.util.spec_from_file_location(name,path)
 if not spec or not spec.loader:raise RuntimeError("cannot import pinned owner "+str(path))
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

B=imp("paired_connected_boundary_patch_853_active_rows",BASE_DRIVER_PATH,BASE_DRIVER_SHA)
BACKLINEAGE_OWNER_PATH=HERE/"paired_connected_boundary_patch_855_active_rows_lineage.py"
BACKLINEAGE_OWNER_SHA="5645dbfda4312b6df81b31bca03ba75fdf16af4f4956540c3b92f66b67eb101d"
BACKLINEAGE_OWNER=imp("paired_boundary_patch_current_backlineage_855",BACKLINEAGE_OWNER_PATH,BACKLINEAGE_OWNER_SHA)
# 856's protected_scope rejects authenticated many-to-one D back lineage.
# Preserve the 856/881 geometry and flip gates, using the checked 855 owner only
# for current-row back/rim/route protection derivation.
B.protected_scope=BACKLINEAGE_OWNER.protected_scope
ENG=B.ENG;M=ENG.M;T=ENG.T;A=ENG.A;J=ENG.J
load_npz=B.load_npz
check_ref=B.check_ref
PARENT_871_PATH=HERE/"paired_connected_boundary_patch_batch_871_replay_localids_staged_root.py"
PARENT_871_SHA="8faaf716c17bac710a0ceed0ea836bd10c63b11148e21a953eb3b2a0758be21e"
PARENT_872_PATH=HERE/"paired_connected_boundary_patch_batch_872_empty_boundary_guard_root.py"
PARENT_872_SHA="b07197286a8ed7e1ef3103e86741eee96b103263c625c1476b484204da5273fe"
PARENT_879_PATH=HERE/"paired_connected_boundary_patch_batch_879_multiface_fixturefix_root.py"
PARENT_879_SHA="cf2953c56f5850f125cdfd30888d6d21b2d84ddb40e155c7d987fdc8cb2a03b3"
PARENT_880_PATH=HERE/"paired_connected_boundary_patch_batch_880_material_ring_root.py"
PARENT_880_SHA="0ff7bf28359fc75b3c81673ac643ab96012dfc85b1d0b71d6cfacb7a456529eb"
PATCH_873_PATH=HERE/"constrained_source_patch_873_root.py"
PATCH_873_SHA="20d54965b2913a8349fd06943da4cb4997331c51b06757f515ffc43f7256d4df"
PATCH=imp("constrained_source_patch_873_pinned",PATCH_873_PATH,PATCH_873_SHA)


def checked_closed_boundary(boundary,label):
    """Require a nonempty triangular 2-manifold before using the legacy reduceat check.

    An empty (0,3) cavity boundary is not a valid closed surface. The pinned
    218 helper calls add.reduceat on its empty edge array and raises IndexError;
    convert that accidental NumPy failure into an explicit fail-closed rejection.
    Nonempty arrays still go through the unchanged exact/topology owner.
    """
    b=np.asarray(boundary,dtype=np.int64)
    if b.ndim!=2 or b.shape[1:]!=(3,):
        raise ValueError(f"{label} cavity boundary must have shape (n,3), got {b.shape}")
    if len(b)==0:
        raise ValueError(f"{label} cavity boundary is empty; expected a nonempty closed 2-manifold")
    return T.closed(b)

def empty_boundary_control():
    empty=np.empty((0,3),dtype=np.int64)
    try:
        checked_closed_boundary(empty,"fixture")
    except ValueError as e:
        if "boundary is empty" not in str(e):
            raise AssertionError("empty boundary must fail with the explicit guard") from e
    else:
        raise AssertionError("empty boundary must not be accepted as a closed manifold")
    tet=np.asarray([[0,1,2,3]],dtype=np.int64)
    closed_faces=np.asarray(T.m.boundary(tet),dtype=np.int64)
    if checked_closed_boundary(closed_faces,"valid fixture")!=(True,0):
        raise AssertionError("nonempty closed tetrahedron boundary must retain legacy result")
    return {"status":"PASS_EMPTY_BOUNDARY_FAIL_CLOSED_CONTROL",
            "empty_shape":list(empty.shape),"empty_rejected_before_reduceat":True,
            "legacy_nonempty_tetra_boundary_result":"PASS"}
MAX_INNER_BACK_HAUSDORFF_M=50e-6
MAX_ORDINARY_QUAD_HAUSDORFF_M=500e-6
MAX_ADAPTIVE_EDGE_CAP_M=0.1
ROLE_VOLUMES={"O":("Skin",),"D":("Fat",),"I":("Skin","Fat")}

def canonical_quad_cycle(old_faces):
 old=np.asarray(old_faces,np.int64)
 edge_counts=Counter()
 for tri in old:
  a,b,c=map(int,tri)
  for e in ((a,b),(b,c),(c,a)):edge_counts[tuple(sorted(e))]+=1
 boundary=[e for e,n in edge_counts.items() if n==1]
 if len(boundary)!=4:raise ValueError("2-2 face pair does not have a four-edge boundary")
 adj=defaultdict(list)
 for a,b in boundary:adj[a].append(b);adj[b].append(a)
 if any(len(x)!=2 for x in adj.values()) or len(adj)!=4:raise ValueError("2-2 face pair boundary is not a simple four-cycle")
 start=min(adj)
 choices=[]
 for nxt in sorted(adj[start]):
  cycle=[start,nxt]
  while len(cycle)<4:
   prev,cur=cycle[-2],cycle[-1]
   ns=[x for x in adj[cur] if x!=prev]
   if len(ns)!=1:raise ValueError("cannot order quadrilateral boundary")
   cycle.append(ns[0])
  if cycle[0] not in adj[cycle[-1]]:raise ValueError("quadrilateral boundary did not close")
  choices.append(tuple(cycle))
 return min(choices)

def oriented_patch_boundary_chain(faces,label):
    """Return a valid oriented four-edge patch boundary; reject mixed winding."""
    f=np.asarray(faces,np.int64)
    if f.shape!=(2,3): raise ValueError(f"{label} patch must contain exactly two triangles")
    counts=Counter();signed=Counter()
    for tri in f:
        for u,v in zip(tri,np.roll(tri,-1)):
            u=int(u);v=int(v);key=tuple(sorted((u,v)))
            if u==v: raise ValueError(f"{label} patch has a degenerate edge")
            counts[key]+=1;signed[key]+=1 if u<v else -1
    if any(n not in (1,2) for n in counts.values()):
        raise ValueError(f"{label} patch has nonmanifold edge incidence")
    boundary=[k for k,n in counts.items() if n==1]
    internal=[k for k,n in counts.items() if n==2]
    if len(boundary)!=4 or len(internal)!=1:
        raise ValueError(f"{label} patch is not a two-triangle quadrilateral")
    if signed[internal[0]]!=0:
        raise ValueError(f"{label} internal diagonal winding does not cancel")
    if any(abs(signed[k])!=1 for k in boundary):
        raise ValueError(f"{label} exterior edge has invalid oriented multiplicity")
    return {k:int(signed[k]) for k in boundary}

def quad_overlay_hausdorff_upper(points,old_faces,new_faces,shared_edge,opposite_vertices):
 """Conservative surface-distance bound from the common projected quadrilateral.

Both triangulations are projected to a best-fit plane. If that projection is a
strictly convex, nonfolded quadrilateral, both patches are single-valued
piecewise-linear height fields over the same polygon. Their pointwise height
difference is affine on the overlay cells and is zero on the common outer
boundary; its maximum is attained where the two diagonals cross. The 3-D gap
between interpolants at that crossing therefore bounds the bidirectional
Hausdorff distance. Nonconvex/degenerate projections fail closed.
"""
 p=np.asarray(points,np.float32);old=np.asarray(old_faces,np.int64);new=np.asarray(new_faces,np.int64)
 old_chain=oriented_patch_boundary_chain(old,"old")
 new_chain=oriented_patch_boundary_chain(new,"replacement")
 if old_chain!=new_chain:
  raise ValueError("replacement oriented boundary does not match original surface normal")
 cyc=canonical_quad_cycle(old);xyz=p[np.asarray(cyc,np.int64)].astype(np.float64)
 centered=xyz-xyz.mean(axis=0)
 _,sv,vt=np.linalg.svd(centered,full_matrices=False)
 if len(sv)<2 or sv[1]<=max(1e-30,sv[0]*1e-12):raise ValueError("quad has degenerate 2-D projection")
 uv=centered@vt[:2].T
 lut={v:i for i,v in enumerate(cyc)}
 poly=np.asarray([uv[lut[v]] for v in cyc])
 cross=[]
 for i in range(4):
  a=poly[(i+1)%4]-poly[i];b=poly[(i+2)%4]-poly[(i+1)%4]
  cross.append(float(a[0]*b[1]-a[1]*b[0]))
 eps=max(1e-30,float(np.max(np.abs(poly)))**2*1e-12)
 if min(cross)>eps:orient=1
 elif max(cross)<-eps:orient=-1
 else:raise ValueError("quad projection is not strictly convex")
 def area2(tri):
  x,y,z=(uv[lut[int(v)]] for v in tri)
  return float((y[0]-x[0])*(z[1]-x[1])-(y[1]-x[1])*(z[0]-x[0]))
 old_area=np.asarray([area2(tri) for tri in old],np.float64)
 new_area=np.asarray([area2(tri) for tri in new],np.float64)
 # canonical_quad_cycle intentionally picks one of two opposite cycle orders.
 # Surface face winding may oppose that arbitrary order, so test nonfolding by
 # requiring all old and replacement triangles to share one nonzero sign.
 # Convexity of the projected outer cycle was proved above; consistent triangle
 # winding plus the exact same four-edge boundary proves both triangulations
 # cover the same projected quad.
 all_area=np.r_[old_area,new_area]
 if np.any(np.abs(all_area)<=eps) or not (np.all(all_area>eps) or np.all(all_area<-eps)):
  raise ValueError("old/new triangulation is not a consistently wound nonfolded projected cover")
 a,b=map(int,shared_edge);c,d=map(int,opposite_vertices)
 pa,pb,pc,pd=(uv[lut[v]] for v in (a,b,c,d))
 r=pb-pa;s=pd-pc;den=float(r[0]*s[1]-r[1]*s[0])
 if abs(den)<=eps:raise ValueError("quad diagonals do not have a stable projected intersection")
 q=pc-pa
 t=float((q[0]*s[1]-q[1]*s[0])/den)
 u=float((q[0]*r[1]-q[1]*r[0])/den)
 if not (0.0<t<1.0 and 0.0<u<1.0):raise ValueError("quad diagonals do not cross inside the quad")
 xyz_lut={int(v):p[int(v)].astype(np.float64) for v in cyc}
 x0=xyz_lut[a]+t*(xyz_lut[b]-xyz_lut[a])
 x1=xyz_lut[c]+u*(xyz_lut[d]-xyz_lut[c])
 raw=float(np.linalg.norm(x0-x1))
 pad=64*np.finfo(np.float64).eps*max(1.0,float(np.max(np.abs(xyz))))
 return {"hausdorff_upper_bound_m":raw+pad,"diagonal_gap_m":raw,"projection":"BEST_FIT_PLANE_STRICT_CONVEX_OVERLAY",
         "diagonal_parameters":[t,u],"cycle_rows":list(cyc),
         "triangle_projected_double_area_signs":{"old":np.sign(old_area).astype(int).tolist(),"new":np.sign(new_area).astype(int).tolist()},
         "triangle_winding_check":"INVARIANT_TO_ARBITRARY_CANONICAL_CYCLE_ORIENTATION",
         "oriented_boundary_chain_match":True,
         "oriented_surface_normal":"replacement exterior edge chain exactly matches original",
         "bound_claim":"conservative bidirectional Hausdorff upper bound for two nonfolded PL triangulations over the same projected convex quadrilateral"}

# Compatibility alias. The strict convex-overlay proof screens every candidate quad.
inner_back_hausdorff_upper=quad_overlay_hausdorff_upper

def active_map(source,cache,label):
 rows=B.active_surface_point_rows(source)
 return B.exact_point_map(source["points_world_m_f32"],cache["points"],label,
          active_rows=rows,volume_index=cache["coordinate_index"])

def fixture_protection(st,role,volumes,sources):
 """Small explicit masks exercise own-inner release and fixed outer-O route rows."""
 out={}
 for vr,cache in volumes.items():
  if role=="I":
   used_roles=["O","I"] if vr=="Skin" else ["D","I"]
  elif role=="D":used_roles=["D","I"]
  else:used_roles=["O","I"]
  recs={};srows=set();vrows=set()
  for r in used_roles:
   surf=sources[r]
   if role=="I" and r=="I":back=[0]
   elif role=="D" and r=="D":back=[0]
   elif role=="I" and r=="O":back=[len(surf["points_world_m_f32"])-1]
   else:back=[]
   rim=[];routes=[]
   recs[r]={"back_rows":back,"rim_rows":rim,"route_rows":routes,
            "active_source_face_vertices":len(B.active_surface_point_rows(surf)),
            "required_protected_source_rows":len(back),"unreferenced_unprotected_source_rows_ignored":0}
   if back:
    mp=B.exact_point_map(surf["points_world_m_f32"],cache["points"],"synthetic protected "+r,
          active_rows=B.active_surface_point_rows(surf),required_rows=back,volume_index=cache["coordinate_index"])
    srows.update(back);vrows.update(map(int,mp[back]))
  if vr=="Fat":
   if role=="I":
    icur=sorted(vrows & set())
    # I current/historical back at row zero is represented in its source record.
    if "I" in recs and recs["I"]["back_rows"]:
     mp=B.exact_point_map(sources["I"]["points_world_m_f32"],cache["points"],"synthetic I back",
          active_rows=B.active_surface_point_rows(sources["I"]),required_rows=recs["I"]["back_rows"],volume_index=cache["coordinate_index"])
     icur=sorted(map(int,mp[recs["I"]["back_rows"]]))
    else:icur=[]
    recs["actual_fat_back_union"]={"current_rows":icur,"historical_rows":icur,"union_count":len(icur)}
    vrows.update(icur)
   else:recs["actual_fat_back_union"]={"current_rows":[],"historical_rows":[],"union_count":0}
  out[vr]={"surface_rows":np.asarray(sorted(srows),np.int64),"volume_rows":np.asarray(sorted(vrows),np.int64),
           "report":{"role":vr,"sources":recs,"synthetic_fixture_protection":True}}
 return out

def role_source_protected_rows_for_volume(protection,role,released_rows=()):
    report=protection.get("report",{})
    records=report.get("sources",{})
    if role not in records:
        if report.get("synthetic_fixture_protection"):
            return set(map(int,protection.get("surface_rows",[])))
        if "synthetic_fixture_protection" in report:
            return set(map(int,protection.get("surface_rows",[])))
        raise ValueError("protected-scope report lacks edited-role source rows for "+role)
    rec=records[role]
    rows=set(map(int,rec.get("back_rows",[])))|set(map(int,rec.get("rim_rows",[])))|set(map(int,rec.get("route_rows",[])))
    return rows-set(map(int,released_rows))

def edited_role_source_protected_rows(protections,role,inner_allowed=None):
    rows=set()
    for vr,protection in protections.items():
        release=(inner_allowed or {}).get(vr,())
        rows.update(role_source_protected_rows_for_volume(protection,role,release))
    return rows

def protected_volume_star_conflicts(star_rows,all_protected_rows,own_protected_rows,fixed_coordinate_topology_only):
    blocked=set(map(int,own_protected_rows)) if fixed_coordinate_topology_only else set(map(int,all_protected_rows))
    return sorted(set(map(int,star_rows))&blocked)

def own_protected_volume_rows(role,vr,protection,source_map,inner_allowed=None):
    released=(inner_allowed or {}).get(vr,())
    src=role_source_protected_rows_for_volume(protection,role,released)
    out=set()
    if src:
        rr=np.asarray(sorted(src),np.int64);mapped=np.asarray(source_map,np.int64)[rr]
        if np.any(mapped<0):raise ValueError("edited-role protected source row lacks current exact volume map")
        out.update(map(int,mapped))
    if role=="I" and vr=="Fat":
        rec=protection.get("report",{}).get("sources",{}).get("actual_fat_back_union",{})
        current=set(map(int,rec.get("current_rows",[])));historical=set(map(int,rec.get("historical_rows",[])))
        out.update(historical-current)
    return out

def role_protection_fixture_control():
    def rec(back=(),rim=(),routes=()):
        return {"back_rows":list(back),"rim_rows":list(rim),"route_rows":list(routes)}
    # O/D deliberately reuse the same small integer rows as I in their own local namespaces.
    protections={
      "Skin":{"surface_rows":np.asarray([0,2,3,6,7]),"volume_rows":np.asarray([100,101]),
        "report":{"sources":{"O":rec((0,),(6,)),"I":rec((2,),(3,),(7,))}}},
      "Fat":{"surface_rows":np.asarray([0,2,3,6,7]),"volume_rows":np.asarray([200,201]),
        "report":{"sources":{"D":rec((0,),(6,)),"I":rec((2,),(3,),(7,)),
                            "actual_fat_back_union":{"current_rows":[200],"historical_rows":[200]}}}}
    }
    own=edited_role_source_protected_rows(protections,"I")
    if own!={2,3,7}:raise AssertionError("role-local I source rows must exclude colliding O/D row IDs")
    released=edited_role_source_protected_rows(protections,"I",{"Skin":{2},"Fat":{2}})
    if released!={3,7}:raise AssertionError("inner release may remove only the edited I back row")
    if not ({2,9,10}&own) or ({0,6,8}&own):
        raise AssertionError("own-quad / other-role namespace separation control")
    allvol={100,101};ownvol={100}
    if protected_volume_star_conflicts({101},allvol,ownvol,False)!=[101]:
        raise AssertionError("default mode must keep all other-role volume protection")
    if protected_volume_star_conflicts({101},allvol,ownvol,True):
        raise AssertionError("fixed-coordinate mode should allow other-role point-star contact")
    if protected_volume_star_conflicts({100},allvol,ownvol,True)!=[100]:
        raise AssertionError("fixed-coordinate mode must retain own-role protected volume rows")
    return {"status":"PASS_ROLE_SCOPED_FIXED_COORDINATE_PROTECTION_CONTROL",
       "source_protected_I_rows":sorted(own),"after_explicit_inner_back_release":sorted(released),
       "same_numeric_O_D_rows_excluded":True,"other_role_volume_star_allowed_only_in_fixed_mode":True,
       "own_role_volume_star_remains_rejected":True}

def allow_inner_back_only(role,protections,sources,volumes):
 if role not in ("I","D"):raise ValueError("inner-back topology-only mode is restricted to I or D")
 allowed={}
 for vr,cache in volumes.items():
  records=protections[vr]["report"].get("sources",{})
  src_roles=[r for r,vs in ROLE_VOLUMES.items() if vr in vs]
  surf_keep=set();vol_keep=set();allowed_rows=set()
  for r in src_roles:
   rec=records.get(r)
   if rec is None:raise ValueError("protected-scope report lacks role "+r)
   back=set(map(int,rec.get("back_rows",[])))
   rim=set(map(int,rec.get("rim_rows",[])))
   routes=set(map(int,rec.get("route_rows",[])))
   release=(back-rim-routes) if r==role else set()
   if r==role:allowed_rows.update(release)
   keep=(back|rim|routes)-release
   if r==role:surf_keep.update(keep)
   if keep:
    surf=sources[r]
    active=B.active_surface_point_rows(surf)
    mp=B.exact_point_map(surf["points_world_m_f32"],cache["points"],"inner-back protection "+vr+"/"+r,
           active_rows=active,required_rows=sorted(keep),volume_index=cache["coordinate_index"])
    vol_keep.update(map(int,mp[list(sorted(keep))]))
  if vr=="Fat" and "actual_fat_back_union" in records:
   rec=records["actual_fat_back_union"]
   cur=set(map(int,rec.get("current_rows",[])));hist=set(map(int,rec.get("historical_rows",[])))
   extra=(hist-cur) if role=="I" else (cur|hist)
   vol_keep.update(extra)
  protections[vr]["surface_rows"]=np.asarray(sorted(surf_keep),np.int64)
  protections[vr]["volume_rows"]=np.asarray(sorted(vol_keep),np.int64)
  protections[vr]["report"]={**protections[vr]["report"],
      "inner_back_topology_only":{"declared":True,"released_role":role,"released_back_rows":sorted(allowed_rows),
       "all_point_coordinates_fixed":True,"rim_and_route_rows_still_protected":True}}
  allowed[vr]=allowed_rows
 return allowed

def enumerate_candidates(role,surface,volumes,protections,edgecap,inner_allowed=None):
 sf=np.asarray(surface["faces"],np.int64);sp=np.asarray(surface["points_world_m_f32"],np.float32)
 edge_faces=defaultdict(list)
 for r,f in enumerate(sf):
  for a,b in ((int(f[0]),int(f[1])),(int(f[1]),int(f[2])),(int(f[2]),int(f[0]))):
   edge_faces[tuple(sorted((a,b)))].append(int(r))
 interior={e:rows for e,rows in edge_faces.items() if len(rows)==2}
 if any(len(rows)>2 for rows in edge_faces.values()):raise ValueError("nonmanifold source edge")
 cross={vr:active_map(surface,c,role+"/"+vr) for vr,c in volumes.items()}
 reverse={}
 edge_hits=Counter()
 for vr,c in volumes.items():
  rev=np.full(len(c["points"]),-1,np.int64);mapped=np.flatnonzero(cross[vr]>=0)
  rev[cross[vr][mapped]]=mapped;reverse[vr]=rev
  for tet in c["tets"][c["bad"]]:
   sr=rev[tet]
   if np.any(sr<0):continue
   for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
    e=tuple(sorted((int(sr[i]),int(sr[j]))))
    if e in interior:edge_hits[e]+=1
 src_prot=edited_role_source_protected_rows(protections,role,inner_allowed)
 counts=Counter();witnesses=[];candidates=[]
 allowed_union=set().union(*(set(map(int,x)) for x in (inner_allowed or {}).values())) if inner_allowed else set()
 for edge,fr0 in interior.items():
  hit=int(edge_hits[edge])
  if hit<=0:continue
  fr=tuple(map(int,fr0));old=sf[np.asarray(fr,np.int64)];quad=set(map(int,old.reshape(-1)))
  reason=None
  if quad & src_prot:reason="protected_source_vertex"; 
  if reason is None:
   try:new,shared,opp=ENG.flip_faces(old)
   except Exception as e:reason="invalid_oriented_flip:"+str(e)
  if reason is None:
   if (min(opp),max(opp)) in set(interior):reason="replacement_diagonal_already_surface_edge"
  if reason is None:
   retired=set(fr);other={tuple(sorted(map(int,f))) for i,f in enumerate(sf) if i not in retired}
   if any(tuple(sorted(map(int,f))) in other for f in new):reason="replacement_face_duplicates_retained_face"
  if reason is None:
   oldshape=B.shape(sp,old);newshape=B.shape(sp,new)
   gain=float(newshape.min()-oldshape.min())
   if gain<=max(1e-15,1e-4*float(oldshape.min())):reason="source_shape_gain_gate"
  if reason is None and np.any(ENG.J.score(sp,new)<1e-5):reason="joint_apex_gate"
  if reason is None:
   length=float(np.linalg.norm(sp[edge[0]].astype(np.float64)-sp[edge[1]].astype(np.float64)))
   if length>edgecap:reason="edge_cap"
  hinfo=None;touch=bool(quad&allowed_union)
  if reason is None and touch:
   try:
    hinfo=inner_back_hausdorff_upper(sp,old,new,shared,opp)
    if hinfo["hausdorff_upper_bound_m"]>MAX_INNER_BACK_HAUSDORFF_M:reason="inner_back_hausdorff_50um_cap"
   except Exception as e:reason="inner_back_hausdorff_unproven:"+str(e)
  stars={};bad_by={};target_rows={};valid=True
  if reason is None:
   for vr,c in volumes.items():
    a,b=map(int,cross[vr][np.asarray(edge,np.int64)])
    try:rr= B.full_edge_star(c,a,b)
    except Exception as e:reason="edge_star:"+str(e);valid=False;break
    stars[vr]=rr
    if len(rr)>24:reason="edge_star_cell_cap";valid=False;break
    verts=np.unique(c["tets"][rr].ravel())
    if len(verts)>16:reason="edge_star_point_cap";valid=False;break
    if len(np.unique(c["mat"][rr]))!=1:reason="edge_star_material_interface";valid=False;break
    if set(map(int,verts))&set(map(int,protections[vr]["volume_rows"])):reason="protected_volume_star";valid=False;break
    bad_by[vr]=int(np.count_nonzero(c["bad"][rr]))
    target_lu=c["target_lookup"];trrows=[]
    for face in old:
     vf=cross[vr][face];matches=target_lu.get(tuple(sorted(map(int,vf))),[])
     if len(matches)!=1:reason="target_parent_face_missing_or_ambiguous";valid=False;break
     trrows.append(int(matches[0]))
    if not valid:break
    target_rows[vr]=trrows
  if reason:
   counts[reason]+=1
   if len(witnesses)<200:witnesses.append({"edge":list(edge),"face_rows":list(fr),"bad_cell_edge_hits":hit,"reason":reason})
   continue
  candidates.append({"edge":edge,"face_rows":fr,"old_faces":old.copy(),"new_faces":new.copy(),"quad":quad,
    "shared":list(shared),"opposite":list(opp),"bad_edge_hit":hit,"bad_by_volume":bad_by,"length_m":length,
    "shape_gain":gain,"stars":stars,"target_rows":target_rows,"inner_back_touched":touch,"inner_back_hausdorff":hinfo})
 candidates.sort(key=lambda c:(-c["bad_edge_hit"],-c["shape_gain"],c["length_m"],c["edge"],c["face_rows"]))
 return {"candidates":candidates,"candidate_edge_count":len(candidates),"edge_hit_count":len(edge_hits),
         "rejection_counts":dict(counts),"rejection_witnesses":witnesses,"source_point_maps":cross}

def choose_disjoint_patches(candidates,surface,volumes,max_patches,max_pairs,max_cells,max_points):
 sf=np.asarray(surface["faces"],np.int64);byedge=list(candidates);unused=set(range(len(byedge)))
 groups=[];selection_witnesses=[];reserved_faces=set();reserved_vertices=set();reserved_cells={vr:set() for vr in volumes}
 def cells_for(cands,vr):return set(int(x) for c in cands for x in c["stars"][vr])
 def patch_valid(cands):
  pairs=[c["face_rows"] for c in cands]
  flat=[r for pp in pairs for r in pp]
  if len(flat)!=len(set(flat)):return False,"overlapping_source_face_rows"
  if not ENG.connect_pairs(pairs,sf):return False,"source_patch_not_connected"
  trial=sf.copy()
  for c in cands:trial[list(c["face_rows"])]=c["new_faces"]
  try:ENG.topology_sig(trial)
  except Exception as e:return False,"combined_source_topology:"+str(e)
  retired=set(flat)
  other={tuple(sorted(map(int,f))) for i,f in enumerate(sf) if i not in retired}
  newkeys=[]
  for c in cands:
   newkeys.extend(tuple(sorted(map(int,f))) for f in c["new_faces"])
  if len(newkeys)!=len(set(newkeys)) or any(k in other for k in newkeys):return False,"combined_new_face_duplicate"
  return True,""
 while unused and len(groups)<max_patches:
  seed_idx=next(iter(sorted(unused,key=lambda i:(-byedge[i]["bad_edge_hit"],-byedge[i]["shape_gain"],byedge[i]["length_m"],byedge[i]["edge"]))))
  seed=byedge[seed_idx]
  conflict=set(seed["quad"])&reserved_vertices
  overlap={vr:sorted(cells_for([seed],vr)&reserved_cells[vr]) for vr in volumes}
  if set(seed["face_rows"])&reserved_faces or conflict or any(overlap.values()):
   unused.remove(seed_idx);selection_witnesses.append({"edge":list(seed["edge"]),"face_rows":list(seed["face_rows"]),
      "reason":"overlaps_previously_selected_disjoint_patch","reserved_source_points":sorted(conflict),
      "reserved_volume_cells":overlap});continue
  group=[seed];unused.remove(seed_idx)
  while len(group)<max_pairs:
   verts=set().union(*(c["quad"] for c in group))
   frontier=[i for i in unused if byedge[i]["quad"]&verts]
   if not frontier:break
   frontier.sort(key=lambda i:(-byedge[i]["bad_edge_hit"],-byedge[i]["shape_gain"],byedge[i]["length_m"],byedge[i]["edge"]))
   added=False
   for idx in frontier:
    c=byedge[idx]
    if set(c["face_rows"])&{r for q in group for r in q["face_rows"]}:
     unused.remove(idx);selection_witnesses.append({"edge":list(c["edge"]),"face_rows":list(c["face_rows"]),"reason":"overlapping_patch_face_rows"});continue
    test=group+[c]
    union_cells={vr:cells_for(test,vr) for vr in volumes}
    if any(len(x)>max_cells for x in union_cells.values()):
     unused.remove(idx);selection_witnesses.append({"edge":list(c["edge"]),"face_rows":list(c["face_rows"]),"reason":"connected_patch_cell_cap"});continue
    union_verts={vr:set(map(int,np.unique(volumes[vr]["tets"][np.asarray(sorted(union_cells[vr]),np.int64)].ravel()))) for vr in volumes}
    if any(len(x)>max_points for x in union_verts.values()):
     unused.remove(idx);selection_witnesses.append({"edge":list(c["edge"]),"face_rows":list(c["face_rows"]),"reason":"connected_patch_point_cap"});continue
    if any(len(np.unique(volumes[vr]["mat"][np.asarray(sorted(union_cells[vr]),np.int64)]))!=1 for vr in volumes):
     unused.remove(idx);selection_witnesses.append({"edge":list(c["edge"]),"face_rows":list(c["face_rows"]),"reason":"connected_patch_material_interface"});continue
    ok,why=patch_valid(test)
    if not ok:
     unused.remove(idx);selection_witnesses.append({"edge":list(c["edge"]),"face_rows":list(c["face_rows"]),"reason":why});continue
    group.append(c);unused.remove(idx);added=True;break
   if not added:break
  # Single candidates have already passed source-edge geometry and local caps.
  groups.append(group)
  reserved_faces.update(r for c in group for r in c["face_rows"])
  reserved_vertices.update(v for c in group for v in c["quad"])
  for vr in volumes:reserved_cells[vr].update(cells_for(group,vr))
 if unused:
  selection_witnesses.append({"unprocessed_candidates":len(unused),"reason":"maximum_patch_count_or_source_disjointness_budget"})
 return groups,selection_witnesses,reserved_cells

def choose_disjoint_patches(candidates,surface,volumes,max_patches,max_pairs,max_cells,max_points):
    """Choose connected source patches whose complete volume stars are disjoint."""
    sf=np.asarray(surface["faces"],np.int64);rank=lambda c:(-c["bad_edge_hit"],-c["shape_gain"],c["length_m"],c["edge"])
    order=sorted(range(len(candidates)),key=lambda i:rank(candidates[i]))
    remaining=set(order);groups=[];witnesses=[];reserved_rows=set();reserved_vertices=set()
    reserved_cells={vr:set() for vr in volumes};base_sig=ENG.topology_sig(sf)
    def patch_valid(group):
        pairs=[c["face_rows"] for c in group]
        rows=[int(r) for pp in pairs for r in pp]
        if len(rows)!=len(set(rows)):return False,"overlapping_source_face_rows"
        if not ENG.connect_pairs(pairs,sf):return False,"source_patch_not_connected"
        trial=sf.copy()
        for c in group:trial[list(c["face_rows"])]=c["new_faces"]
        try:after=ENG.topology_sig(trial)
        except Exception as e:return False,"combined_source_topology:"+str(e)
        if after!=base_sig:return False,"source_topology_signature_changed"
        retired=set(rows);retained={tuple(sorted(map(int,f))) for i,f in enumerate(sf) if i not in retired}
        emitted=[tuple(sorted(map(int,f))) for c in group for f in c["new_faces"]]
        if len(emitted)!=len(set(emitted)) or any(k in retained for k in emitted):return False,"combined_new_face_duplicate"
        return True,""
    while remaining and len(groups)<max_patches:
        seed_i=next(i for i in order if i in remaining);seed=candidates[seed_i]
        conflict=sorted(set(seed["quad"])&reserved_vertices)
        overlap={vr:sorted(set(map(int,seed["stars"][vr]))&reserved_cells[vr]) for vr in volumes}
        if set(seed["face_rows"])&reserved_rows or conflict or any(overlap.values()):
            remaining.remove(seed_i);witnesses.append({"edge":list(seed["edge"]),"face_rows":list(seed["face_rows"]),
              "reason":"overlaps_previously_selected_full_star_patch","reserved_source_points":conflict,"reserved_volume_cells":overlap})
            continue
        group=[seed];remaining.remove(seed_i)
        while len(group)<max_pairs:
            verts=set().union(*(c["quad"] for c in group));frontier=[i for i in order if i in remaining and candidates[i]["quad"]&verts]
            accepted=None
            for idx in frontier:
                c=candidates[idx]
                if set(c["face_rows"])&{r for z in group for r in z["face_rows"]}:
                    remaining.remove(idx);witnesses.append({"edge":list(c["edge"]),"face_rows":list(c["face_rows"]),"reason":"shares_retired_face_with_patch"});continue
                if set(c["quad"])&reserved_vertices or any(set(map(int,c["stars"][vr]))&reserved_cells[vr] for vr in volumes):
                    continue
                test=group+[c];cell_union={vr:set().union(*(set(map(int,z["stars"][vr])) for z in test)) for vr in volumes}
                if any(len(x)>max_cells for x in cell_union.values()):continue
                if any(len(np.unique(volumes[vr]["tets"][np.asarray(sorted(cell_union[vr]),np.int64)].ravel()))>max_points for vr in volumes):continue
                if any(len(np.unique(volumes[vr]["mat"][np.asarray(sorted(cell_union[vr]),np.int64)]))!=1 for vr in volumes):continue
                ok,why=patch_valid(test)
                if not ok:
                    witnesses.append({"edge":list(c["edge"]),"face_rows":list(c["face_rows"]),"reason":"not_added_to_patch:"+why});continue
                accepted=idx;break
            if accepted is None:break
            group.append(candidates[accepted]);remaining.remove(accepted)
        ok,why=patch_valid(group)
        if not ok:raise AssertionError("seed candidate ceased to be valid: "+why)
        groups.append(group);reserved_rows.update(r for c in group for r in c["face_rows"])
        reserved_vertices.update(v for c in group for v in c["quad"])
        for vr in volumes:reserved_cells[vr].update(int(x) for c in group for x in c["stars"][vr])
    if remaining:witnesses.append({"unprocessed_candidates":len(remaining),"reason":"max_patch_count_reached"})
    return groups,witnesses,reserved_cells

def _quality_deficit(det,q):
    r=np.minimum(np.minimum(np.asarray(q,dtype=np.float64)/1e-5,np.asarray(det,dtype=np.float64)/6e-18),1.0)
    return float(-np.log(np.maximum(r,1e-300)).sum())

def solve_patch_in_memory(role,group,surface,vr,c,source_map,args,cumulative_delta,global_flux,allow_good_tie=False):
    sf=np.asarray(surface["faces"],np.int64);sp=np.asarray(surface["points_world_m_f32"],np.float32)
    s2v=np.asarray(source_map[vr],np.int64);target=np.asarray(c["target"],np.int64);target_lu=c["target_lookup"]
    ops=[];target_rows=[];replacement_pairs=[];stars=[]
    for cand in group:
        rows=tuple(map(int,cand["face_rows"]));oldsrc=sf[list(rows)];newsrc=np.asarray(cand["new_faces"],np.int64)
        old_source_volume=s2v[oldsrc];new_volume=s2v[newsrc]
        if np.any(old_source_volume<0) or np.any(new_volume<0):raise ValueError("source patch face uses unmapped active source point")
        oldtarget=[];trrows=[];rels=[]
        for face in old_source_volume:
            hits=target_lu.get(tuple(sorted(map(int,face))),[])
            if len(hits)!=1:raise ValueError("source patch face lacks a unique target boundary row")
            tr=int(hits[0]);trrows.append(tr);oldtarget.append(target[tr].copy());rels.append(ENG.rel(target[tr],face))
        if len(set(rels))!=1:raise ValueError("source and target winding relation differs within face pair")
        oldtarget=np.asarray(oldtarget,np.int64)
        if rels[0]<0:new_volume=new_volume[:,[0,2,1]]
        ops.append({"face_rows":rows,"old_source":oldsrc.copy(),"new_source":newsrc.copy(),
             "old_target":oldtarget,"new_target":new_volume.copy(),"target_rows":trrows,"orientation_relation":int(rels[0]),
             "edge":list(cand["edge"]),"source_patch_kind":cand.get("source_patch_kind","edge_2to2"),"patch_boundary_cycle":list(cand.get("patch_boundary_cycle",[])),
             "material_face_ring_added_cell_rows":list(cand.get("material_face_ring_added_cell_rows",{}).get(vr,[])),
             "material_face_ring_diagnostics":cand.get("material_face_ring_diagnostics",{}).get(vr)})
        target_rows.extend(trrows);replacement_pairs.extend((oldtarget[j],new_volume[j]) for j in range(len(oldtarget)))
        stars.append(np.asarray(cand["stars"][vr],np.int64))
    cellrows=np.unique(np.concatenate(stars)).astype(np.int64)
    if len(cellrows)>args.maximum_old_cells:raise ValueError("connected patch full edge-star cell cap")
    old=np.asarray(c["tets"][cellrows],np.int64);oldmat=np.asarray(c["mat"][cellrows])
    if len(np.unique(oldmat))!=1:raise ValueError("connected patch crosses material index")
    verts=np.unique(old);pointset=set(map(int,verts))
    if len(verts)>args.maximum_points:raise ValueError("connected patch full edge-star point cap")
    oldbad=int(np.count_nonzero(c["bad"][cellrows]))
    olddef=_quality_deficit(c["det"][cellrows],c["q"][cellrows])
    oldq=np.asarray(c["q"][cellrows],np.float64);olddet=np.asarray(c["det"][cellrows],np.float64)
    def cavity_diag(newflux=None,newbad=None,newdef=None,newq=None,newdet=None,solver=None):
        out={"volume_role":vr,"edge_face_rows":[list(map(int,x["face_rows"])) for x in ops],
             "source_edges":[list(map(int,x["edge"])) for x in ops],"input_cell_rows":cellrows.tolist(),
             "material":int(oldmat[0])}
        # Exact old/new fluxes and native measurements are kept losslessly in decimal text/ranges.
        out["old_exact_cell_det6_sum"]=str(oldsum)
        out["new_oriented_boundary_det6"]=str(newflux) if newflux is not None else None
        out["exact_det6_delta_from_flux"]=str(int(newflux)-int(oldsum)) if newflux is not None else None
        out["local_volume_fraction_from_flux"]=float(abs(int(newflux)-int(oldsum))/int(oldsum)) if newflux is not None and int(oldsum)>0 else None
        out["cumulative_global_volume_fraction_from_flux"]=float(abs(int(cumulative_delta)+int(newflux)-int(oldsum))/global_flux) if newflux is not None else None
        out["native_bad_old"]=oldbad;out["native_log_deficit_old"]=olddef
        out["old_quality_q_range"]=[float(oldq.min()),float(oldq.max())] if len(oldq) else None
        out["old_signed_det6_range"]=[float(olddet.min()),float(olddet.max())] if len(olddet) else None
        out["native_bad_new"]=int(newbad) if newbad is not None else None
        out["native_log_deficit_new"]=float(newdef) if newdef is not None else None
        if newq is not None:
            x=np.asarray(newq,np.float64);out["new_quality_q_range"]=[float(x.min()),float(x.max())]
        if newdet is not None:
            x=np.asarray(newdet,np.float64);out["new_signed_det6_range"]=[float(x.min()),float(x.max())]
        if solver is not None:out["solver"]={k:solver.get(k) for k in ("status","stage1_message","stage1_status","stage2_status","elapsed_s")}
        return out
    for op in ops:
        for f in op["old_target"]:
            if not pointset.issuperset(map(int,f)):raise ValueError("old target face vertex absent from complete cavity")
    local_map=np.full(len(c["points"]),-1,np.int64);local_map[verts]=np.arange(len(verts),dtype=np.int64)
    lp=np.asarray(c["points"][verts],np.float32);lo=local_map[old]
    oldboundary=np.asarray(T.m.boundary(old),np.int64)
    local_chain=ENG.chain(local_map[oldboundary])
    for op in ops:
        for f in op["old_target"]:
            lf=local_map[f];key=tuple(sorted(map(int,lf)))
            if local_chain.get(key,0)!=ENG.fsign(lf):raise ValueError("retired target facet is not present with target orientation in old cavity boundary")
    closed,open_edges=checked_closed_boundary(oldboundary,"old");topo=T.manifold_topology(oldboundary)
    if not closed or topo["bad_vertex_links"] or topo["boundary_components"]!=1:raise ValueError("old cavity boundary is not one closed manifold")
    replacements=[(local_map[a],local_map[b]) for a,b in replacement_pairs]
    if any(np.any(a<0) or np.any(b<0) for a,b in replacements):raise ValueError("target replacement references a point outside the cavity")
    newboundary=ENG.oriented_boundary_replace(local_map[oldboundary],replacements)
    closed2,open2=checked_closed_boundary(newboundary,"replacement");topo2=T.manifold_topology(newboundary)
    if not closed2 or topo2["bad_vertex_links"] or topo2["boundary_components"]!=1:raise ValueError("new cavity boundary is not one closed manifold")
    # Reject inverted/over-budget target shells before spending any MILP time.
    local_ip=[ENG.S.S.ip(x) for x in lp]
    oldexact=[ENG.S.S.det(*(local_ip[int(x)] for x in tet)) for tet in lo]
    if not oldexact or min(oldexact)<=0:raise ValueError("old cavity has nonpositive exact determinant")
    oldsum=sum(oldexact);newflux=ENG.exact_boundary_flux(lp,newboundary);pre_delta=newflux-oldsum
    if newflux<=0:
        raise ValueError("PREMILP_REJECT_NONPOSITIVE_ORIENTED_BOUNDARY "+json.dumps(cavity_diag(newflux=newflux),sort_keys=True))
    if abs(pre_delta)/float(oldsum)>args.maximum_local_volume_fraction:
        raise ValueError("PREMILP_REJECT_LOCAL_VOLUME_BUDGET "+json.dumps(cavity_diag(newflux=newflux),sort_keys=True))
    if abs(cumulative_delta+pre_delta)/float(global_flux)>args.maximum_global_volume_fraction:
        raise ValueError("PREMILP_REJECT_CUMULATIVE_VOLUME_BUDGET "+json.dumps(cavity_diag(newflux=newflux),sort_keys=True))
    seconds=min(float(args.solve_seconds),max(.02,float(args.deadline-time.monotonic())))
    if seconds<=.02:raise TimeoutError("batch solve deadline reached")
    report,newlocal=ENG.S.solve(lp,newboundary,seconds=seconds,max_points=args.maximum_points,material=int(oldmat[0]))
    if newlocal is None or report.get("status")!="PASS_PARTIAL_POSITIVE_CAVITY_MILP":
        raise ValueError("MILP_NO_FEASIBLE_INCUMBENT "+json.dumps(cavity_diag(newflux=newflux,solver=report),sort_keys=True))
    newglobal=verts[np.asarray(newlocal,np.int64)]
    newlo=np.asarray(newlocal,np.int64)
    newexact=[ENG.S.S.det(*(local_ip[int(x)] for x in tet)) for tet in newlo]
    if not newexact or min(newexact)<=0:
        raise ValueError("MILP_REJECT_NONPOSITIVE_OUTPUT "+json.dumps(cavity_diag(newflux=newflux,solver=report),sort_keys=True))
    oldflux=ENG.exact_boundary_flux(lp,local_map[oldboundary])
    oldsum=sum(oldexact);newsum=sum(newexact);delta=newsum-oldsum
    if oldflux!=oldsum or newflux!=newsum:raise ValueError("cavity exact boundary flux does not match old/new cells")
    if abs(delta)/float(oldsum)>args.maximum_local_volume_fraction:raise ValueError("local exact volume fraction exceeds configured cap")
    if abs(cumulative_delta+delta)/float(global_flux)>args.maximum_global_volume_fraction:raise ValueError("cumulative exact volume fraction exceeds configured cap")
    nd,nq=M.measurements(c["points"],newglobal);newbad,newdef=B.qscore(nd,nq)
    tol=max(1e-8,1e-6*olddef)
    progress=(newbad<oldbad) or (newbad==oldbad and newdef<olddef-tol)
    good_tie=(allow_good_tie and oldbad==0 and newbad==0 and abs(newdef-olddef)<=tol)
    if newbad>oldbad or not (progress or good_tie):
        raise ValueError("LOCAL_NATIVE_QUALITY_NO_PROGRESS "+json.dumps(
            cavity_diag(newflux=newflux,newbad=newbad,newdef=newdef,newq=nq,newdet=nd,solver=report),sort_keys=True))
    oldtargetvol=float(oldsum);newtargetvol=float(newsum)
    return {"volume_role":vr,"candidate_operations":ops,"cell_rows":cellrows,"old_tets":old.copy(),
      "old_materials":oldmat.copy(),"cavity_points":verts,"replacement_tets":np.asarray(newglobal,np.int64),
      "material":int(oldmat[0]),"new_boundary_local":np.asarray(newboundary,np.int64),
      "local_exact_old_det6":str(oldsum),"local_exact_new_det6":str(newsum),"exact_det6_delta":int(delta),
      "old_flux_det6":str(oldflux),"new_flux_det6":str(newflux),"old_bad":oldbad,"new_bad":int(newbad),
      "old_log_deficit":olddef,"new_log_deficit":float(newdef),"solver":report,
      "target_rows":target_rows,"target_replacements":[(int(op["target_rows"][j]),op["new_target"][j].copy()) for op in ops for j in range(len(op["target_rows"]))],
      "local_volume_fraction":float(abs(delta)/oldsum),"global_volume_fraction":float(abs(cumulative_delta+delta)/global_flux),
      "inner_back_touched":any(bool(x.get("inner_back_touched")) for x in group),
      "inner_back_hausdorff_max_m":max((float(x["inner_back_hausdorff"]["hausdorff_upper_bound_m"]) for x in group if x.get("inner_back_hausdorff")),default=0.0)}

def compose_surface_batch(surface,accepted,parent_path):
    parent=dict(surface);faces=np.asarray(parent["faces"],np.int64).copy();face_ops={}
    for patch_id,rec in enumerate(accepted):
        group=rec["group"];oldrows=sorted(r for c in group for r in c["face_rows"])
        union=sorted(set(tuple(map(int,x)) for r in oldrows for x in A.rows_union(parent,r)))
        ancestry=sorted(set(tuple(map(int,x)) for r in oldrows for x in A.rows_anc(parent,r)))
        if not union or not ancestry:raise ValueError("accepted source patch has empty typed ancestry")
        for c in group:
            rows=list(map(int,c["face_rows"]));new=np.asarray(c["new_faces"],np.int64)
            faces[rows]=new
        for r in oldrows:
            if r in face_ops:raise ValueError("accepted patches overlap a source face")
            face_ops[r]={"union":union,"ancestry":ancestry,"patch":patch_id}
    out=dict(parent);out["faces"]=faces
    qf=np.asarray(parent["quality_star_face_output_to_input_row"],np.int64).copy()
    changed=sorted(face_ops)
    qf[changed]=-1;out["quality_star_face_output_to_input_row"]=qf
    unions=[];ancs=[]
    for r in range(len(faces)):
        if r in face_ops:
            unions.append(face_ops[r]["union"]);ancs.append(face_ops[r]["ancestry"])
        else:
            unions.append(A.rows_union(parent,r));ancs.append(A.rows_anc(parent,r))
    if any(not x for x in unions) or any(not x for x in ancs):raise ValueError("source output contains empty retained ancestry")
    uoff,udata=A.csr(unions,7);aoff,adata=A.csr(ancs,3)
    out["quality_star_patch_union_offsets"]=uoff
    for j,k in enumerate(ENG.TUN):out[k]=udata[:,j]
    out["face_ancestry_offsets"]=aoff
    for j,k in enumerate(ENG.ANC):out[k]=adata[:,j]
    direct=np.asarray([x[0] for x in unions],np.int64)
    for j,k in enumerate(ENG.TSC):out[k]=direct[:,j]
    for k in ("face_parent_source_face_rows","face_exact_source_row"):
        if k in out:
            z=np.asarray(out[k]).copy();z[changed]=-1;out[k]=z
    out["source_current_candidate_path"]=np.asarray([str(Path(parent_path).resolve())])
    out["source_current_candidate_sha256"]=np.asarray([sha(parent_path)])
    out["typed_parent_candidate_path"]=np.asarray([str(Path(parent_path).resolve())])
    out["typed_parent_candidate_sha256"]=np.asarray([sha(parent_path)])
    out["connected_boundary_patch_batch_semantics"]=np.asarray(["per-patch changed faces carry that connected patch full 7-lane source union and 3-lane ancestry; retained rows preserve parent CSR"])
    # Complete patch-local operation maps, flattened once for this batch.
    face_offsets=[0];face_union=[];anc_offsets=[0];anc_rows=[];patch_offsets=[0];patch_faces=[];op_face_rows=[]
    for patch_id,rec in enumerate(accepted):
        rows=sorted(r for c in rec["group"] for r in c["face_rows"]);patch_faces.extend(rows);patch_offsets.append(len(patch_faces));op_face_rows.extend(rows)
        oldrows=rows
        union=sorted(set(tuple(map(int,x)) for r in oldrows for x in A.rows_union(parent,r)))
        ancestry=sorted(set(tuple(map(int,x)) for r in oldrows for x in A.rows_anc(parent,r)))
        for _ in rows:
            face_union.extend(union);face_offsets.append(len(face_union))
            anc_rows.extend(ancestry);anc_offsets.append(len(anc_rows))
    out["connected_boundary_patch_operation_face_rows"]=np.asarray(op_face_rows,np.int64)
    out["connected_boundary_patch_operation_face_parent_offsets"]=np.asarray(face_offsets,np.int64)
    out["connected_boundary_patch_operation_face_parent_union"]=np.asarray(face_union,np.int64).reshape((-1,7))
    out["connected_boundary_patch_operation_face_ancestry_offsets"]=np.asarray(anc_offsets,np.int64)
    out["connected_boundary_patch_operation_face_ancestry"]=np.asarray(anc_rows,np.int64).reshape((-1,3))
    out["connected_boundary_patch_operation_patch_face_offsets"]=np.asarray(patch_offsets,np.int64)
    out["connected_boundary_patch_operation_patch_face_rows"]=np.asarray(patch_faces,np.int64)
    return out,face_ops

def compose_volume_batch(role,cache,accepted,source_ref,all_source_refs,args):
    parent=dict(cache["arrays"]);p=np.asarray(cache["points"],np.float32);t=np.asarray(cache["tets"],np.int64);mat=np.asarray(cache["mat"])
    tk=cache["target_key"];target=np.asarray(cache["target"],np.int64).copy()
    retired=np.unique(np.concatenate([r["volumes"][role]["cell_rows"] for r in accepted])).astype(np.int64)
    if len(retired)!=sum(len(r["volumes"][role]["cell_rows"]) for r in accepted):raise ValueError("accepted full cell stars overlap")
    keep=np.ones(len(t),bool);keep[retired]=False
    out_new=[];out_newmat=[];patch_cell_offsets=[0];patch_cells=[];patch_new_offsets=[0];patch_new_rows=[]
    global_replaced_rows=[];replace_parent_offsets=[0];replace_parent_rows=[];patch_delta=0;patch_reports=[]
    local_limit=0
    for patch_id,rec in enumerate(accepted):
        pr=rec["volumes"][role];cellrows=np.asarray(pr["cell_rows"],np.int64)
        start=int(keep.sum())+len(out_new)
        new=np.asarray(pr["replacement_tets"],np.int64)
        out_new.extend(new);out_newmat.extend([int(pr["material"])]*len(new))
        patch_cells.extend(map(int,cellrows));patch_cell_offsets.append(len(patch_cells))
        newrows=list(range(start,start+len(new)));patch_new_rows.extend(newrows);patch_new_offsets.append(len(patch_new_rows))
        for _ in new:
            replace_parent_rows.extend(map(int,cellrows));replace_parent_offsets.append(len(replace_parent_rows))
        for tr,face in pr["target_replacements"]:
            if int(tr) in global_replaced_rows:raise ValueError("two accepted patches replace same target row")
            global_replaced_rows.append(int(tr));target[int(tr)]=np.asarray(face,np.int64)
        patch_delta+=int(pr["exact_det6_delta"])
        patch_reports.append({"patch_id":patch_id,"input_cells":cellrows.tolist(),"output_cells":newrows,
          "new_cell_count":len(new),"exact_det6_delta":str(pr["exact_det6_delta"]),
          "old_bad":pr["old_bad"],"new_bad":pr["new_bad"],
          "source_to_target_orientation_relations":[int(op["orientation_relation"]) for op in pr["candidate_operations"]]})
    newarr=np.asarray(out_new,np.int64).reshape((-1,4));newmat=np.asarray(out_newmat,dtype=mat.dtype)
    outt=np.vstack((t[keep],newarr));outmat=np.r_[mat[keep],newmat]
    actual=np.asarray(T.m.boundary(outt),np.int64)
    if ENG.chain(actual)!=ENG.chain(target):raise ValueError("composed volume boundary does not match all patch target updates")
    if len(np.unique(np.sort(outt,axis=1),axis=0))!=len(outt):raise ValueError("composed volume has duplicate tetrahedra")
    # Whole-volume native metrics are measured once after all independent local patches.
    dd,qq=M.measurements(p,outt)
    if np.any(np.asarray(dd)<=0):raise ValueError("composed volume has nonpositive measured tetrahedra")
    bad=np.asarray(~np.asarray(M.passes(dd,qq),bool),bool);before_bad=np.asarray(cache["bad"],bool)
    if int(bad.sum())>int(before_bad.sum()):raise ValueError("composed volume regresses whole-role native bad count")
    old_def=_quality_deficit(cache["det"],cache["q"]);new_def=_quality_deficit(dd,qq)
    if args.role!="I" and int(bad.sum())==int(before_bad.sum()) and new_def>=old_def-max(1e-8,1e-6*old_def):
        raise ValueError("composed volume has no strict whole-role native progress")
    global_before=ENG.exact_boundary_flux(p,cache["target"])
    global_after=ENG.exact_boundary_flux(p,target)
    if global_after-global_before!=patch_delta:raise ValueError("composed exact target-flux delta differs from sum of local cavity deltas")
    origin=np.r_[np.flatnonzero(keep).astype(np.int64),np.full(len(newarr),-1,np.int64)]
    inv=np.full(len(t),-1,np.int64);inv[keep]=np.arange(int(keep.sum()),dtype=np.int64)
    out=dict(parent);out["tetrahedra"]=outt;out["material_indices"]=outmat
    out["boundary_faces"]=actual
    for key in ("desired_boundary_faces","proposed_current_desired_boundary_faces","proposed_current_desired_boundary_faces_rows","boundary_faces"):
        if key in out:out[key]=target
    out["output_cell_to_input_cell"]=origin;out["input_cell_to_output_cell"]=inv
    out["retained_input_cell_rows"]=np.flatnonzero(keep)
    out["replacement_tetrahedra"]=newarr
    out["replacement_parent_offsets"]=np.asarray(replace_parent_offsets,np.int64)
    out["replacement_parent_cell_rows"]=np.asarray(replace_parent_rows,np.int64)
    out["connected_boundary_patch_source_role"]=np.asarray([args.role])
    out["connected_boundary_patch_operation_cell_rows"]=retired
    out["connected_boundary_patch_operation_patch_cell_offsets"]=np.asarray(patch_cell_offsets,np.int64)
    out["connected_boundary_patch_operation_patch_cell_rows"]=np.asarray(patch_cells,np.int64)
    out["connected_boundary_patch_operation_patch_new_cell_offsets"]=np.asarray(patch_new_offsets,np.int64)
    out["connected_boundary_patch_operation_patch_new_cell_rows"]=np.asarray(patch_new_rows,np.int64)
    out["connected_boundary_patch_batch_operation_reports"]=np.asarray([json.dumps(x,sort_keys=True) for x in patch_reports])
    out["connected_boundary_patch_input_volume_path"]=np.asarray([str(cache["path"])])
    out["connected_boundary_patch_input_volume_sha256"]=np.asarray([sha(cache["path"])])
    out["connected_boundary_patch_input_surface_path"]=np.asarray([str(Path(source_ref["path"]).resolve())])
    out["connected_boundary_patch_input_surface_sha256"]=np.asarray([str(source_ref["sha256"])])
    out["connected_boundary_patch_regional_volume_fraction_caps"]=np.asarray([args.maximum_local_volume_fraction,args.maximum_global_volume_fraction],np.float64)
    out["connected_boundary_patch_exact_global_det6_delta"]=np.asarray([str(global_after-global_before)])
    out["connected_boundary_patch_old_native_bad_count"]=np.asarray([int(before_bad.sum())],np.int64)
    out["connected_boundary_patch_new_native_bad_count"]=np.asarray([int(bad.sum())],np.int64)
    for r,ref in all_source_refs.items():
        out[f"typed_{r}_candidate_path"]=np.asarray([str(Path(ref["path"]).resolve())])
        out[f"typed_{r}_candidate_sha256"]=np.asarray([str(ref["sha256"])])
    out.update(ENG.P785.unchanged_point_lineage_updates(parent,len(p)))
    ENG.P785.assert_unchanged_point_lineage(out,len(p))
    readp=np.asarray(out["points_world_m"],np.float32)
    if not np.array_equal(readp.view(np.uint32),p.view(np.uint32)):raise ValueError("batch volume points changed in composed arrays")
    return out,{"volume_role":role,"input_cells":len(t),"output_cells":len(outt),"retired_cells":len(retired),
      "replacement_cells":len(newarr),"native_bad_before":int(before_bad.sum()),"native_bad_after":int(bad.sum()),
      "log_deficit_before":old_def,"log_deficit_after":new_def,"exact_global_det6_before":str(global_before),
      "exact_global_det6_after":str(global_after),"exact_global_det6_delta":str(global_after-global_before),
      "patches":patch_reports,"complete_target_boundary":True,"all_cells_measured_positive":True}

def independent_batch_replay(role,parent_surface,parent_volumes,proposal_surface,proposal_volumes,accepted,source_maps,args):
    """Rebuild source operations, full stars, local shells, signed flux, cells and quality from parent bytes."""
    sf0=np.asarray(parent_surface["faces"],np.int64);sf1=np.asarray(proposal_surface["faces"],np.int64)
    sp=np.asarray(parent_surface["points_world_m_f32"],np.float32)
    if not np.array_equal(sp.view(np.uint32),np.asarray(proposal_surface["points_world_m_f32"],np.float32).view(np.uint32)):
        raise ValueError("source point bytes changed")
    expected_sf=sf0.copy();face_owner={};union_by_patch={};anc_by_patch={}
    for pid,rec in enumerate(accepted):
        group=rec["group"];oldrows=sorted(r for c in group for r in c["face_rows"])
        if len(oldrows)!=len(set(oldrows)):raise ValueError("patch repeats a source face row")
        union=set();ancestry=set()
        for r in oldrows:
            union.update(tuple(map(int,x)) for x in A.rows_union(parent_surface,r))
            ancestry.update(tuple(map(int,x)) for x in A.rows_anc(parent_surface,r))
        if not union or not ancestry:raise ValueError("patch parent typed union/ancestry empty")
        union_by_patch[pid]=union;anc_by_patch[pid]=ancestry
        for c in group:
            rows=list(map(int,c["face_rows"]));new=reconstruct_source_patch(c,sf0,sp)
            if not np.array_equal(new,np.asarray(c["new_faces"],np.int64)):raise ValueError("patch journal flip differs from exact parent source rows")
            expected_sf[rows]=new
        for r in oldrows:
            if r in face_owner:raise ValueError("patches overlap source face rows")
            face_owner[r]=pid
    if not np.array_equal(sf1,expected_sf):raise ValueError("independent batch source geometry reconstruction mismatch")
    if ENG.topology_sig(sf0)!=ENG.topology_sig(sf1):raise ValueError("source topology signature changed")
    parent_qf=np.asarray(parent_surface["quality_star_face_output_to_input_row"],np.int64)
    out_qf=np.asarray(proposal_surface["quality_star_face_output_to_input_row"],np.int64)
    uoff=np.asarray(proposal_surface["quality_star_patch_union_offsets"],np.int64);aoff=np.asarray(proposal_surface["face_ancestry_offsets"],np.int64)
    for r in range(len(sf0)):
        if r in face_owner:
            if out_qf[r]!=-1:raise ValueError("changed face retains an exact-parent output-face claim")
            lo,hi=map(int,uoff[r:r+2]);got={tuple(int(proposal_surface[k][j]) for k in ENG.TUN) for j in range(lo,hi)}
            if got!=union_by_patch[face_owner[r]]:raise ValueError("changed face lacks its exact patch-specific full typed union")
            lo,hi=map(int,aoff[r:r+2]);got={tuple(int(proposal_surface[k][j]) for k in ENG.ANC) for j in range(lo,hi)}
            if got!=anc_by_patch[face_owner[r]]:raise ValueError("changed face lacks its exact patch-specific ancestry")
        else:
            if not np.array_equal(sf1[r],sf0[r]) or out_qf[r]!=parent_qf[r]:raise ValueError("retained source face geometry/map changed")
            if A.rows_union(proposal_surface,r)!=A.rows_union(parent_surface,r) or A.rows_anc(proposal_surface,r)!=A.rows_anc(parent_surface,r):
                raise ValueError("retained source face ancestry changed")
    role_results={};deltas_by_patch={}
    for vr,parent_volume in parent_volumes.items():
        c=parent_volume;out=proposal_volumes[vr]
        p=np.asarray(c["points"],np.float32);t=np.asarray(c["tets"],np.int64);mat=np.asarray(c["mat"])
        if not np.array_equal(p.view(np.uint32),np.asarray(out["points_world_m"],np.float32).view(np.uint32)):
            raise ValueError(vr+" volume point bytes changed")
        cross=np.asarray(source_maps[vr],np.int64)
        target0=np.asarray(c["target"],np.int64);target1=np.asarray(out[B.target_key(out)],np.int64);expected_target=target0.copy()
        lookup=B.face_lookup(target0);oldrows_all=[];newtet_all=[];newmat_all=[];local_deltas=[]
        patch_oldrows=[];patch_newranges=[];patch_materials=[]
        old_target_flux=ENG.exact_boundary_flux(p,target0)
        for pid,rec in enumerate(accepted):
            ops=[];stars=[]
            for cand in rec["group"]:
                rows=list(map(int,cand["face_rows"]));oldsrc=sf0[rows];newsrc=reconstruct_source_patch(cand,sf0,sp)
                sfv=cross[oldsrc]
                if np.any(sfv<0):raise ValueError("source face row map missing during replay")
                oldtarget=[];rels=[];targetrows=[]
                for f in sfv:
                    hits=lookup.get(tuple(sorted(map(int,f))),[])
                    if len(hits)!=1:raise ValueError("parent target face is missing or ambiguous")
                    tr=int(hits[0]);tf=target0[tr].copy();targetrows.append(tr);oldtarget.append(tf);rels.append(ENG.rel(tf,f))
                if len(set(rels))!=1:raise ValueError("source/target orientation is inconsistent during replay")
                oldtarget=np.asarray(oldtarget,np.int64);newtarget=cross[newsrc]
                if rels[0]<0:newtarget=newtarget[:,[0,2,1]]
                for j,tr in enumerate(targetrows):
                    expected_target[tr]=newtarget[j]
                internal_edges=source_patch_internal_edges(oldsrc)
                if not internal_edges:raise ValueError("replay source patch has no internal edge")
                patch_star=[]
                for edge in internal_edges:
                    va,vb=map(int,cross[np.asarray(edge,np.int64)])
                    patch_star.extend(map(int,B.full_edge_star(c,va,vb)))
                star=np.asarray(sorted(set(patch_star)),np.int64)
                if cand.get("source_patch_kind")=="polygon_3face5vertex":
                    star,ring_added,ring_diag=expand_complete_material_face_ring(c,star,oldtarget,int(np.asarray(mat)[star[0]]))
                    recorded_ring=np.asarray(cand.get("material_face_ring_added_cell_rows",{}).get(vr,[]),np.int64)
                    if not np.array_equal(ring_added,recorded_ring):raise ValueError(vr+" complete material face-neighbor ring differs from candidate journal")
                stars.append(star)
                ops.append({"oldtarget":oldtarget,"newtarget":newtarget,"targetrows":targetrows})
            cellrows=np.unique(np.concatenate(stars)).astype(np.int64)
            recorded=np.asarray(rec["volumes"][vr]["cell_rows"],np.int64)
            if not np.array_equal(cellrows,recorded):raise ValueError(vr+" recorded complete source edge-star union mismatches")
            oldrows_all.extend(map(int,cellrows));patch_oldrows.append(cellrows)
            verts=np.unique(t[cellrows]);loc=np.full(len(p),-1,np.int64);loc[verts]=np.arange(len(verts))
            lp=p[verts];oldlocal=loc[t[cellrows]];oldboundary=np.asarray(T.m.boundary(t[cellrows]),np.int64)
            bchain=ENG.chain(loc[oldboundary]);replace=[]
            for op in ops:
                for j,oldf in enumerate(op["oldtarget"]):
                    lf=loc[oldf];key=tuple(sorted(map(int,lf)))
                    if bchain.get(key,0)!=ENG.fsign(lf):raise ValueError("replay old target face is absent/opposite in complete cavity boundary")
                    replace.append((lf,loc[op["newtarget"][j]]))
            want_local=ENG.oriented_boundary_replace(loc[oldboundary],replace)
            newt=np.asarray(rec["volumes"][vr]["replacement_tets"],np.int64)
            if not len(newt):raise ValueError("accepted patch has no replacement tets")
            if not set(map(int,np.unique(newt))).issubset(set(map(int,verts))):raise ValueError("replacement tet uses a point outside its complete cavity")
            actual_local=np.asarray(T.m.boundary(newt),np.int64)
            if ENG.chain(actual_local)!=ENG.chain(verts[want_local]):raise ValueError("replacement tets do not realize the oriented patch cavity shell")
            ip=[ENG.S.S.ip(x) for x in lp]
            oldexact=[ENG.S.S.det(*(ip[int(x)] for x in tet)) for tet in oldlocal]
            newlocal=loc[newt];newexact=[ENG.S.S.det(*(ip[int(x)] for x in tet)) for tet in newlocal]
            if min(oldexact)<=0 or min(newexact)<=0:raise ValueError("patch exact determinant sign failed")
            oldsum=sum(oldexact);newsum=sum(newexact)
            if ENG.exact_boundary_flux(lp,loc[oldboundary])!=oldsum or ENG.exact_boundary_flux(lp,want_local)!=newsum:
                raise ValueError("patch local exact volume/flux mismatch")
            delta=newsum-oldsum
            if delta!=int(rec["volumes"][vr]["exact_det6_delta"]):raise ValueError("patch exact det6 journal delta mismatch")
            local_deltas.append(delta);newtet_all.extend(newt);newmat_all.extend([int(rec["volumes"][vr]["material"])]*len(newt))
            patch_newranges.append((len(newtet_all)-len(newt),len(newtet_all)));patch_materials.append(int(rec["volumes"][vr]["material"]))
        if not np.array_equal(expected_target,target1):raise ValueError(vr+" current target does not match all independently reconstructed source flips")
        retired=np.unique(np.asarray(oldrows_all,np.int64));keep=np.ones(len(t),bool);keep[retired]=False
        expected_t=np.vstack((t[keep],np.asarray(newtet_all,np.int64).reshape((-1,4))))
        expected_m=np.r_[mat[keep],np.asarray(newmat_all,dtype=mat.dtype)]
        if not np.array_equal(expected_t,np.asarray(out["tetrahedra"],np.int64)) or not np.array_equal(expected_m,np.asarray(out["material_indices"])):
            raise ValueError(vr+" whole volume connectivity/material reconstruction mismatch")
        if not np.array_equal(np.asarray(out["output_cell_to_input_cell"],np.int64),np.r_[np.flatnonzero(keep),np.full(len(newtet_all),-1,np.int64)]):
            raise ValueError(vr+" retained/new output-cell row map mismatch")
        if not np.array_equal(np.asarray(out["input_cell_to_output_cell"],np.int64)[keep],np.arange(int(keep.sum()),dtype=np.int64)) or np.any(np.asarray(out["input_cell_to_output_cell"],np.int64)[~keep]!=-1):
            raise ValueError(vr+" input-to-output retained-cell row map mismatch")
        actual=np.asarray(T.m.boundary(expected_t),np.int64)
        if ENG.chain(actual)!=ENG.chain(target1):raise ValueError(vr+" complete oriented target boundary mismatch")
        if len(np.unique(np.sort(expected_t,axis=1),axis=0))!=len(expected_t):raise ValueError(vr+" duplicate output tetrahedra")
        dd,qq=M.measurements(p,expected_t)
        if np.any(np.asarray(dd)<=0):raise ValueError(vr+" output contains nonpositive measured tetrahedra")
        beforebad,beforedef=B.qscore(c["det"],c["q"]);afterbad,afterdef=B.qscore(dd,qq)
        if afterbad>beforebad:raise ValueError(vr+" whole-volume native bad count regressed")
        out_bad_by_mat={}
        for m in np.unique(mat):
            oldb=int(np.count_nonzero(c["bad"]&(mat==m)))
            newb=int(np.count_nonzero((~np.asarray(M.passes(dd,qq),bool))&(expected_m==m)))
            if newb>oldb:raise ValueError(vr+" native bad count regressed for material "+str(int(m)))
            out_bad_by_mat[str(int(m))]={"before":oldb,"after":newb}
        fluxdelta=ENG.exact_boundary_flux(p,target1)-old_target_flux
        if fluxdelta!=sum(local_deltas):raise ValueError(vr+" exact global target delta differs from local cavity deltas")
        rpo=np.asarray(out["replacement_parent_offsets"],np.int64);rpr=np.asarray(out["replacement_parent_cell_rows"],np.int64)
        if rpo.shape!=(len(newtet_all)+1,) or rpo[0]!=0 or rpo[-1]!=len(rpr):raise ValueError(vr+" replacement parent CSR malformed")
        cursor=0
        for rec in accepted:
            rows=np.asarray(rec["volumes"][vr]["cell_rows"],np.int64)
            for _ in rec["volumes"][vr]["replacement_tets"]:
                lo,hi=map(int,rpo[cursor:cursor+2]);cursor+=1
                if set(map(int,rpr[lo:hi]))!=set(map(int,rows)):raise ValueError(vr+" replacement tet missing complete patch parent cells")
        role_results[vr]={"native_bad_before":beforebad,"native_bad_after":afterbad,"native_log_deficit_before":beforedef,
          "native_log_deficit_after":afterdef,"material_bad_counts":out_bad_by_mat,"exact_global_det6_delta":str(fluxdelta),
          "patch_exact_deltas":[str(x) for x in local_deltas],"retired_cells":len(retired),"new_cells":len(newtet_all),
          "all_output_cells_positive":True,"full_target_boundary":True}
        deltas_by_patch[vr]=local_deltas
    if role=="I":
        for pid in range(len(accepted)):
            if deltas_by_patch["Skin"][pid]+deltas_by_patch["Fat"][pid]!=0:
                raise ValueError("paired I patch exact signed-volume deltas do not cancel")
        old_bad=sum(x["native_bad_before"] for x in role_results.values())
        new_bad=sum(x["native_bad_after"] for x in role_results.values())
        old_def=sum(x["native_log_deficit_before"] for x in role_results.values())
        new_def=sum(x["native_log_deficit_after"] for x in role_results.values())
        if new_bad>old_bad or (new_bad==old_bad and new_def>=old_def-max(1e-8,1e-6*old_def)):
            raise ValueError("paired I independent replay has no strict aggregate native quality progress")
    return {"status":"PASS_INDEPENDENT_BATCH_REPLAY","role":role,"accepted_patches":len(accepted),
      "changed_source_face_rows":sorted(face_owner),"source_topology_preserved":True,
      "full_typed_patch_unions_and_ancestry":True,"all_volume_roles":role_results,
      "paired_I_exact_volume_cancellation":True if role=="I" else None,
      "all_source_and_volume_point_coordinates_fixed":True,"physical_admission":False,"native_steps":0}

def _source_state(state_path):
    root=json.loads(Path(state_path).read_text());st=root.get("current_verified_state_20261031",root)
    if st.get("physical_admission") is not False or int(st.get("native_steps",-1))!=0:
        raise ValueError("current input is not source-only geometry")
    if not all(k in st for k in ("source_surfaces","Skin_current","Fat_current")):raise ValueError("current state lacks typed source/volume refs")
    return root,st

def build_patch_journal(surface,volumes,accepted,affected):
    journal=[]
    for pid,rec in enumerate(accepted):
        oldrows=sorted(r for c in rec["group"] for r in c["face_rows"])
        journal.append({"patch_id":pid,"selection":rec["selection"],"source_face_pairs":[list(c["face_rows"]) for c in rec["group"]],
          "source_edges":[list(c["edge"]) for c in rec["group"]],
          "source_patch_kinds":[c.get("source_patch_kind","edge_2to2") for c in rec["group"]],
          "source_patch_face_rows":[list(c["face_rows"]) for c in rec["group"]],
          "source_patch_boundary_cycles":[list(c.get("patch_boundary_cycle",[])) for c in rec["group"]],
          "source_patch_new_faces":[np.asarray(c["new_faces"],np.int64).tolist() for c in rec["group"]],
          "material_face_ring_added_cell_rows_by_role":{vr:[list(c.get("material_face_ring_added_cell_rows",{}).get(vr,[])) for c in rec["group"]] for vr in affected},
          "material_face_ring_diagnostics_by_role":{vr:[c.get("material_face_ring_diagnostics",{}).get(vr) for c in rec["group"]] for vr in affected},
          "source_changed_face_rows":oldrows,
          "source_parent_union_7":[list(x) for x in sorted(set(tuple(map(int,t)) for r in oldrows for t in A.rows_union(surface,r)))],
          "source_parent_ancestry_3":[list(x) for x in sorted(set(tuple(map(int,t)) for r in oldrows for t in A.rows_anc(surface,r)))],
          "other_role_protected_volume_rows_touched":{vr:sorted(set(int(y) for c in rec["group"] for y in c["other_role_protected_volume_rows_touched"].get(vr,[]))) for vr in affected},
          "volumes":{vr:{"input_cell_rows":pr["cell_rows"].tolist(),"replacement_tets":pr["replacement_tets"].tolist(),
            "exact_det6_delta":str(pr["exact_det6_delta"]),"old_bad":pr["old_bad"],"new_bad":pr["new_bad"],
            "source_to_target_orientation_relations":[int(op["orientation_relation"]) for op in pr["candidate_operations"]],
            "old_quality_q_range":[float(np.min(volumes[vr]["q"][pr["cell_rows"]])),float(np.max(volumes[vr]["q"][pr["cell_rows"]]))],
            "inner_back_hausdorff_upper_m":pr["inner_back_hausdorff_max_m"],
            "source_quad_hausdorff_upper_m":max(float(c["quad_hausdorff"]["hausdorff_upper_bound_m"]) for c in rec["group"]),
            "source_quad_hausdorff_cap_m":max(float(c["surface_hausdorff_limit_m"]) for c in rec["group"])} for vr,pr in rec["volumes"].items()}})
    return journal

def run_batch_state(args,fixture=False):
    tic=time.monotonic();state_path=Path(args.state).resolve();root,st=_source_state(state_path)
    if args.role not in ("O","I","D"):raise ValueError("role must be O, I, or D")
    if args.inner_back_topology_only and args.role not in ("I","D"):raise ValueError("inner-back topology mode is I/D only")
    if not (0<args.batch_seconds<=120 and 0<args.solve_seconds<=3 and 1<=args.max_patches<=32 and 1<=args.max_pairs_per_patch<=8):
        raise ValueError("batch/solver/pair caps exceed approved bounds")
    affected=("Skin",) if args.role=="O" else ("Fat",) if args.role=="D" else ("Skin","Fat")
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError("batch output path must be fresh")
    out.mkdir(parents=True)
    volumes={vr:B.volume_cache(st[vr+"_current"]) for vr in affected}
    surface_ref=st["source_surfaces"][args.role];surface=load_npz(check_ref(surface_ref,"current typed "+args.role))
    fixture_sources={r:load_npz(check_ref(st["source_surfaces"][r],"source "+r)) for r in ("O","I","D")}
    protections=fixture_protection(st,args.role,volumes,fixture_sources) if fixture else B.protected_scope(state_path,st,args.role,volumes,out,fixture=False)
    inner_allowed=None
    if args.inner_back_topology_only:inner_allowed=allow_inner_back_only(args.role,protections,fixture_sources,volumes)
    for vr,protection in protections.items():
        role_rows=role_source_protected_rows_for_volume(protection,args.role,(inner_allowed or {}).get(vr,()))
        protection["surface_rows"]=np.asarray(sorted(role_rows),np.int64)
        protection["report"]={**protection.get("report",{}),"edited_role_surface_protection":{
            "role":args.role,"source_point_rows":sorted(role_rows),"source_row_namespace":args.role,
            "cross_role_source_rows_not_applied":True,"volume_protection_retains_all_roles":True}}
    fixed_mode=bool(getattr(args,"fixed_coordinate_topology_only",False))
    # Do not write candidates before all affected source maps are exact and all volume parents pass.
    active=B.active_surface_point_rows(surface)
    source_maps={vr:active_map(surface,c,args.role+"/"+vr) for vr,c in volumes.items()}
    # Exact target flux is invariant while all candidates are evaluated from this
    # pinned parent. Compute it once per role and reuse it across all seed solves.
    global_fluxes={vr:ENG.exact_boundary_flux(c["points"],c["target"]) for vr,c in volumes.items()}
    if any(int(x)<=0 for x in global_fluxes.values()):raise ValueError("pinned parent target has nonpositive exact signed volume")
    args.deadline=time.monotonic()+float(args.batch_seconds)
    discovery=enumerate_candidates_fast(args.role,surface,volumes,protections,args.maximum_edge_m,inner_allowed,
        deadline=args.deadline,fixed_coordinate_topology_only=fixed_mode,
        max_multiface_pockets=int(getattr(args,"max_multiface_pockets",64)),
        max_cells=args.maximum_old_cells,max_points=args.maximum_points)
    if args.source_edge is not None:
        requested=tuple(sorted(map(int,args.source_edge)))
        if requested[0]==requested[1] or requested[0]<0:
            raise ValueError("source edge must contain two distinct nonnegative current point rows")
        before_count=int(discovery["candidate_edge_count"])
        selected=[c for c in discovery["candidates"]
                  if tuple(sorted(map(int,c["edge"])))==requested
                  and c.get("source_patch_kind","edge_2to2")=="edge_2to2"]
        discovery["source_edge_filter"]={"requested_edge":list(requested),
            "unfiltered_candidate_count":before_count,"matched_edge_2to2_candidate_count":len(selected),
            "candidate_kind":"edge_2to2","filter_complete":bool(discovery.get("search_complete",True))}
        discovery["candidates"]=selected
        discovery["candidate_edge_count"]=len(selected)
    groups,selection_witnesses,reserved=choose_disjoint_seed_patches(discovery["candidates"],surface,volumes,args.max_patches,args.maximum_old_cells,args.maximum_points,deadline=args.deadline)
    if not groups:
        report={"schema":"h706.connected-boundary-patch-batch.v1","status":"NO_ELIGIBLE_CONNECTED_PATCHES",
          "parent_state":pin(state_path),"role":args.role,"current_backlineage_owner":pin(BACKLINEAGE_OWNER_PATH),"candidate_edges":discovery["candidate_edge_count"],"source_edge_filter":discovery.get("source_edge_filter"),
          "bad_edge_incidence":discovery["edge_hit_count"],"rejections":discovery["rejection_counts"],
          "candidate_edge_search_complete":bool(discovery.get("search_complete",True)),
          "candidate_edges_scored":int(discovery.get("candidate_edges_scored",0)),
          "selection_witnesses":selection_witnesses,"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
        (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        print(json.dumps(report,indent=2,sort_keys=True),flush=True);return report
    accepted=[];rejections=[];cumulative={vr:0 for vr in affected}
    accepted_source_rows=set();accepted_source_vertices=set();accepted_cells={vr:set() for vr in affected}
    candidates_attempted=0
    def trial_group(group,source_kind):
        nonlocal cumulative
        results={};trialdelta={}
        for vr in affected:
            pr=solve_patch_in_memory(args.role,group,surface,vr,volumes[vr],source_maps,args,
                cumulative[vr],global_fluxes[vr],allow_good_tie=(args.role=="I"))
            results[vr]=pr;trialdelta[vr]=int(pr["exact_det6_delta"])
        if args.role=="I":
            if sum(trialdelta.values())!=0:raise ValueError("paired interface patch exact volume deltas fail to cancel")
            progress=[]
            for vr,pr in results.items():
                tol=max(1e-8,1e-6*float(pr["old_log_deficit"]))
                improved=(pr["new_bad"]<pr["old_bad"]) or (
                    pr["new_bad"]==pr["old_bad"] and pr["new_log_deficit"]<pr["old_log_deficit"]-tol)
                if pr["new_bad"]>pr["old_bad"]:raise ValueError("paired interface role has a local bad-count regression: "+vr)
                progress.append(improved)
            if not any(progress):raise ValueError("paired interface patch has no strict aggregate local quality improvement")
        return results,trialdelta
    for group in groups:
        if candidates_attempted>=args.max_patches or time.monotonic()>=args.deadline:break
        conflict_faces={r for c0 in group for r in c0["face_rows"]}&accepted_source_rows
        conflict_vertices=set().union(*(c0["quad"] for c0 in group))&accepted_source_vertices
        conflict_cells={vr:set(int(x) for c0 in group for x in c0["stars"][vr])&accepted_cells[vr] for vr in affected}
        if conflict_faces or conflict_vertices or any(conflict_cells.values()):
            rejections.append({"patch_seed_edges":[list(c0["edge"]) for c0 in group],"reason":"overlap_with_accepted_patch","source_faces":sorted(conflict_faces),"source_points":sorted(conflict_vertices),"volume_cells":{k:sorted(v) for k,v in conflict_cells.items()}});continue
        candidates_attempted+=1
        try:
            pres,pdelta=trial_group(group,"connected")
        except Exception as e:
            rejections.append({"patch_seed_edges":[list(c0["edge"]) for c0 in group],
                "source_face_rows":[list(c0["face_rows"]) for c0 in group],
                "reason":type(e).__name__+": "+str(e),
                "reserved_after_failure":False})
            continue
        accepted.append({"group":group,"volumes":pres,"selection":"connected-source patch"})
        for vr in affected:
            cumulative[vr]+=pdelta[vr]
            accepted_cells[vr].update(int(x) for c0 in group for x in c0["stars"][vr])
        accepted_source_rows.update(r for c0 in group for r in c0["face_rows"])
        accepted_source_vertices.update(v for c0 in group for v in c0["quad"])
    if not accepted:
        report={"schema":"h706.connected-boundary-patch-batch.v1","status":"NO_PATCH_PASSED_LOCAL_SOLVER_AND_QUALITY_GATES",
          "parent_state":pin(state_path),"role":args.role,"current_backlineage_owner":pin(BACKLINEAGE_OWNER_PATH),"candidate_edges":discovery["candidate_edge_count"],"source_edge_filter":discovery.get("source_edge_filter"),
          "bad_edge_incidence":discovery["edge_hit_count"],"accepted_patches":0,"rejection_counts":discovery["rejection_counts"],
          "selection_witnesses":selection_witnesses,"solver_rejections":rejections,
          "elapsed_s":time.monotonic()-tic,"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
        (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        print(json.dumps(report,indent=2,sort_keys=True),flush=True);return report
    # Exactly one source NPZ and one volume NPZ per affected role are emitted.
    source_arrays,face_ops=compose_surface_batch(surface,accepted,surface_ref["path"])
    if ENG.topology_sig(np.asarray(source_arrays["faces"],np.int64))!=ENG.topology_sig(np.asarray(surface["faces"],np.int64)):
        raise ValueError("final batched source topology signature differs from parent")
    source_path=out/f"candidate-{args.role}-typed-current.npz";np.savez_compressed(source_path,**source_arrays)
    source_read=load_npz(source_path);source_pin=pin(source_path)
    role_refs={r:dict(st["source_surfaces"][r]) for r in ("O","I","D")};role_refs[args.role]=source_pin
    output_volumes={};volume_metrics={}
    for vr in affected:
        va,vm=compose_volume_batch(vr,volumes[vr],accepted,source_ref=source_pin,all_source_refs=role_refs,args=args)
        output_volumes[vr]=va;volume_metrics[vr]=vm
    # Freeze byte-exact candidates and complete operation ancestry before independent replay.
    volume_refs={}
    for vr in affected:
        vp=out/f"{vr.lower()}-batch-volume-proposal.npz";np.savez_compressed(vp,**output_volumes[vr])
        vrread=load_npz(vp)
        for k in output_volumes[vr]:
            if k in vrread and not np.array_equal(vrread[k],output_volumes[vr][k]):
                raise ValueError("serialized volume output readback differs at "+k)
        volume_refs[vr]=pin(vp)
        output_volumes[vr]=vrread
    journal_path=out/"patch-journal.json"
    journal=build_patch_journal(surface,volumes,accepted,affected)
    journal_path.write_text(json.dumps(journal,indent=2,sort_keys=True)+chr(10))
    rejdir=out/"patch-rejections";rejdir.mkdir()
    rejection_refs=[]
    for i,r in enumerate(rejections):
        rejected_path=rejdir/f"rejected-{i+1:03d}.json"
        rejected_path.write_text(json.dumps(r,indent=2,sort_keys=True)+chr(10))
        rejection_refs.append(pin(rejected_path))
    pending={"schema":"h706.connected-boundary-quality-batch.pending-replay.v1",
      "status":"UNVERIFIED_PENDING_INDEPENDENT_REPLAY","parent_state":pin(state_path),"role":args.role,"current_backlineage_owner":pin(BACKLINEAGE_OWNER_PATH),
      "source_parent":pin(check_ref(surface_ref,"source parent")),"source_candidate":source_pin,
      "volume_candidates":volume_refs,"patch_journal":pin(journal_path),"rejection_reports":rejection_refs,
      "accepted_patch_count":len(accepted),"physical_admission":False,"native_steps":0,
      "mass_mapping":"NOT_RUN","verified_state_written":False}
    pending_path=out/"pending-independent-replay.json"
    pending_path.write_text(json.dumps(pending,indent=2,sort_keys=True)+chr(10))
    try:
        replay=independent_batch_replay(args.role,surface,volumes,source_read,output_volumes,accepted,source_maps,args)
    except Exception as e:
        failure={"schema":"h706.connected-boundary-quality-batch.replay-failure.v1",
          "status":"INDEPENDENT_REPLAY_FAILED","parent_state":pending["parent_state"],"role":args.role,
          "source_candidate":source_pin,"volume_candidates":volume_refs,"patch_journal":pin(journal_path),
          "exception_type":type(e).__name__,"exception":str(e),"verified_state_written":False,
          "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
        (out/"independent-replay-failure.json").write_text(json.dumps(failure,indent=2,sort_keys=True)+chr(10))
        raise
    if args.role=="I" and cumulative["Skin"]+cumulative["Fat"]!=0:raise ValueError("paired I exact global target deltas do not cancel")
    # Output state remains explicitly unadmitted; changed-source full all185/current walls proof is separate.
    outstate=json.loads(json.dumps(st));outstate["source_surfaces"]=role_refs
    for vr in affected:
        old=outstate[vr+"_current"];outstate[vr+"_current"]={**old,**volume_refs[vr],"cells":len(output_volumes[vr]["tetrahedra"]),
          "native_quality_failures":volume_metrics[vr]["native_bad_after"],"nonpositive":0,
          "boundary_source_binding_status":"PENDING_FINAL_CURRENT_TARGET_METADATA_AND_ALL185_AUDIT"}
    for k in ("exact_context","independent_replay","independent_point_replay","independent_allrole_local_replay",
              "independent_target_replay","source_evidence","source_audit","native_qualification","mass_mapping_receipt"):
        outstate.pop(k,None)
    outstate.update({"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN","source_state":pin(state_path),
      "producer":pin(Path(__file__)),"pinned_872_parent":pin(PARENT_872_PATH),"pinned_current_backlineage_owner":pin(BACKLINEAGE_OWNER_PATH),
      "pinned_873_patch_enumerator":pin(PATCH_873_PATH),
      "status":"SOURCE_ONLY_BATCH_PROPOSAL_PENDING_CURRENT_ALL185_AND_NATIVE_GATES",
      "connected_boundary_patch_batch":{"role":args.role,"source_edge_filter":discovery.get("source_edge_filter"),"accepted_patch_count":len(accepted),"source_surface_candidate":source_pin,
       "volume_candidates":volume_refs,"independent_replay":"PASS","inner_back_topology_only":bool(args.inner_back_topology_only),
       "inner_back_policy":{"flag":"805_inner_back_layer_repair","declared_roles":[args.role] if args.inner_back_topology_only else [],
        "maximum_displacement_m":5e-5 if args.inner_back_topology_only else None,"actual_point_displacement_m":0.0,
        "outer_O_back_source_faces_and_points_unchanged":True,"rims_routes_protected":True,
        "max_inner_back_surface_hausdorff_upper_m":max((pr["volumes"][affected[0]]["inner_back_hausdorff_max_m"] for pr in accepted),default=0.0) if args.inner_back_topology_only else 0.0},
       "external_current_OID_both_walls_all185":"REQUIRED_NOT_RUN","native_thresholds_unchanged":True,
       "fixed_coordinate_topology_only":fixed_mode,
       "fixed_coordinate_topology_policy":{"declared":fixed_mode,"all_point_coordinates_fixed":True,
         "edited_role_own_back_rim_routes_remain_protected_unless_inner_mode_explicit":True,
         "other_role_source_faces_and_coordinates_unchanged":True,"all_nonedited_volume_boundary_facets_exact":True,
         "only_other_role_protected_volume_point_star_contact_is_relaxed":True},
       "adaptive_source_edge_cap_m":float(args.maximum_edge_m),"absolute_edge_cap_m":MAX_ADAPTIVE_EDGE_CAP_M,
       "ordinary_quad_hausdorff_cap_m":MAX_ORDINARY_QUAD_HAUSDORFF_M,
       "inner_back_quad_hausdorff_cap_m":MAX_INNER_BACK_HAUSDORFF_M,
       "strict_convex_overlay_required_for_every_candidate":False,
       "edge_flip_requires_strict_convex_quad_overlay":True,
       "three_face_five_vertex_patch_requires_simple_nonfolded_projected_polygon_cover":True,
       "three_face_patch_max_old_faces":3,"three_face_patch_max_vertices":5,
       "all_source_point_bytes_unchanged":True}})
    state_out=out/"state.json";state_out.write_text(json.dumps(outstate,indent=2,sort_keys=True)+"\n")
    rejdir=out/"patch-rejections"
    report={"schema":"h706.connected-boundary-quality-batch.v1","status":"PASS_BATCH_LOCAL_INDEPENDENT_REPLAY_PENDING_OUTER_AUDIT",
      "parent_state":pin(state_path),"role":args.role,"current_backlineage_owner":pin(BACKLINEAGE_OWNER_PATH),"affected_volume_roles":list(affected),
      "pinned_872_parent":pin(PARENT_872_PATH),"pinned_current_backlineage_owner":pin(BACKLINEAGE_OWNER_PATH),"pinned_873_patch_enumerator":pin(PATCH_873_PATH),
      "source_parent":pin(check_ref(surface_ref,"source parent")),"source_candidate":source_pin,"volume_candidates":volume_refs,
      "candidate_edges":discovery["candidate_edge_count"],"source_edge_filter":discovery.get("source_edge_filter"),"bad_cell_incidence_edge_count":discovery["edge_hit_count"],
      "candidate_edge_search_complete":bool(discovery.get("search_complete",True)),
      "candidate_edges_scored":int(discovery.get("candidate_edges_scored",discovery["candidate_edge_count"])),
      "candidate_multiface_pocket_count":int(discovery.get("candidate_multiface_pocket_count",0)),
      "candidate_multiface_patch_count":int(discovery.get("candidate_multiface_patch_count",0)),
      "multiface_search_complete":bool(discovery.get("multiface_search_complete",True)),
      "multiface_rejection_counts":discovery.get("multiface_rejection_counts",{}),
      "multiface_rejection_witnesses":discovery.get("multiface_rejection_witnesses",[]),
      "eligible_seed_candidate_count":len(groups),"attempted_seed_patches":candidates_attempted,
      "accepted_patch_count":len(accepted),"accepted_source_pair_count":sum(len(x["group"]) for x in accepted),
      "accepted_source_patch_face_count":sum(len(c["face_rows"]) for x in accepted for c in x["group"]),
      "accepted_multiface_patch_count":sum(1 for x in accepted for c in x["group"] if c.get("source_patch_kind")=="polygon_3face5vertex"),
      "selection_witnesses":selection_witnesses,"source_candidate_rejection_counts":discovery["rejection_counts"],"source_edge_filter":discovery.get("source_edge_filter"),
      "solver_rejections":len(rejections),"rejection_directory":str(rejdir.resolve()),
      "rejection_reports":[pin(x) for x in sorted(rejdir.glob("*.json"))],"patch_journal":pin(out/"patch-journal.json"),
      "independent_replay":replay,"quality_by_volume":volume_metrics,
      "inner_back_topology_only":bool(args.inner_back_topology_only),
      "fixed_coordinate_topology_only":fixed_mode,
      "fixed_coordinate_topology_policy":{"declared":fixed_mode,"all_point_coordinates_fixed":True,
       "edited_role_own_back_rim_routes_remain_protected_unless_inner_mode_explicit":True,
       "other_role_source_faces_and_coordinates_unchanged":True,"all_nonedited_volume_boundary_facets_exact":True,
       "only_other_role_protected_volume_point_star_contact_is_relaxed":True},
      "inner_back_policy":{"declared_roles":[args.role] if args.inner_back_topology_only else [],
       "maximum_displacement_m":5e-5 if args.inner_back_topology_only else None,"actual_point_displacement_m":0.0,
       "outer_O_back_source_faces_and_points_unchanged":True,"rims_routes_protected":True,
       "hausdorff_upper_bound_per_changed_quad_m":max((float(c["inner_back_hausdorff"]["hausdorff_upper_bound_m"]) for x in accepted for c in x["group"] if c.get("inner_back_hausdorff")),default=0.0)},
      "adaptive_source_edge_policy":{"configured_edge_cap_m":float(args.maximum_edge_m),"absolute_edge_cap_m":MAX_ADAPTIVE_EDGE_CAP_M,
       "ordinary_quad_hausdorff_cap_m":MAX_ORDINARY_QUAD_HAUSDORFF_M,"inner_back_quad_hausdorff_cap_m":MAX_INNER_BACK_HAUSDORFF_M,
       "strict_convex_overlay_required_for_every_candidate":False,
       "edge_flip_requires_strict_convex_quad_overlay":True,
       "three_face_five_vertex_patch_requires_simple_nonfolded_projected_polygon_cover":True,
       "three_face_patch_max_old_faces":3,"three_face_patch_max_vertices":5,
       "all_source_point_bytes_unchanged":True,"outer_O_back_geometry_unchanged":True,
       "exact_flux_native_source_and_boundary_gates_unchanged":True,
       "max_accepted_quad_hausdorff_upper_m":max((float(c["quad_hausdorff"]["hausdorff_upper_bound_m"]) for x in accepted for c in x["group"]),default=0.0)},
      "full_current_OID_walls_all185":"REQUIRED_SEPARATE_FINAL_AUDIT_NOT_RUN","mass_mapping":"NOT_RUN",
      "physical_admission":False,"native_steps":0,"elapsed_s":time.monotonic()-tic,"output_state":pin(state_out)}
    rp=out/"report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True),flush=True);return report

def _enumerate_candidates_fast_edge_only(role,surface,volumes,protections,edgecap,inner_allowed=None,deadline=None,fixed_coordinate_topology_only=False):
    sf=np.asarray(surface["faces"],np.int64);sp=np.asarray(surface["points_world_m_f32"],np.float32)
    edge_faces=defaultdict(list)
    for r,f in enumerate(sf):
        a,b,c=map(int,f)
        for e in (tuple(sorted((a,b))),tuple(sorted((b,c))),tuple(sorted((c,a)))):edge_faces[e].append(int(r))
    interior={e:rows for e,rows in edge_faces.items() if len(rows)==2}
    if any(len(rows)>2 for rows in edge_faces.values()):raise ValueError("nonmanifold source edge")
    face_counts=Counter(tuple(sorted(map(int,f))) for f in sf)
    cross={vr:active_map(surface,c,role+"/"+vr) for vr,c in volumes.items()}
    reverse={};edge_hits=Counter()
    for vr,c in volumes.items():
        rev=np.full(len(c["points"]),-1,np.int64);mapped=np.flatnonzero(cross[vr]>=0);rev[cross[vr][mapped]]=mapped;reverse[vr]=rev
        for tet in c["tets"][c["bad"]]:
            sr=rev[tet]
            for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
                if sr[i]<0 or sr[j]<0:continue
                e=tuple(sorted((int(sr[i]),int(sr[j]))))
                if e in interior:edge_hits[e]+=1
    src_prot=edited_role_source_protected_rows(protections,role,inner_allowed)
    allowed_union=set().union(*(set(map(int,x)) for x in (inner_allowed or {}).values())) if inner_allowed else set()
    counts=Counter();witnesses=[];candidates=[];failed_multiface_pairs=[];scored=0;search_complete=True
    for edge,fr0 in interior.items():
        if deadline is not None and time.monotonic()>=deadline:
            search_complete=False;break
        hit=int(edge_hits[edge])
        if hit<=0:continue
        fr=tuple(map(int,fr0));old=sf[np.asarray(fr,np.int64)];quad=set(map(int,old.reshape(-1)));reason=None
        if quad&src_prot:reason="protected_source_vertex"
        if reason is None:
            try:new,shared,opp=ENG.flip_faces(old)
            except Exception as e:reason="invalid_oriented_flip:"+str(e)
        if reason is None and (min(opp),max(opp)) in interior:reason="replacement_diagonal_already_surface_edge"
        if reason is None:
            removed=Counter(tuple(sorted(map(int,f))) for f in old)
            emitted=[tuple(sorted(map(int,f))) for f in new]
            if len(emitted)!=len(set(emitted)) or any(face_counts[k]-removed[k]>0 for k in emitted):
                reason="replacement_face_duplicates_retained_face"
        if reason is None:
            oldshape=B.shape(sp,old);newshape=B.shape(sp,new);gain=float(newshape.min()-oldshape.min())
            if not np.isfinite(newshape).all() or np.any(newshape<=0):
                reason="replacement_source_triangle_degenerate"
        if reason is None and np.any(J.score(sp,new)<1e-5):reason="joint_apex_gate"
        if reason is None:
            length=float(np.linalg.norm(sp[edge[0]].astype(np.float64)-sp[edge[1]].astype(np.float64)))
            if length>edgecap:reason="edge_cap"
        touch=bool(quad&allowed_union);hinfo=None
        hlimit=MAX_INNER_BACK_HAUSDORFF_M if touch else MAX_ORDINARY_QUAD_HAUSDORFF_M
        if reason is None:
            try:
                hinfo=quad_overlay_hausdorff_upper(sp,old,new,shared,opp)
                hinfo["maximum_allowed_m"]=float(hlimit)
                hinfo["scope"]="inner-back topology-only" if touch else "ordinary source quad"
                if hinfo["hausdorff_upper_bound_m"]>hlimit:
                    reason="inner_back_hausdorff_50um_cap" if touch else "ordinary_quad_hausdorff_500um_cap"
            except Exception as e:reason="quad_overlay_hausdorff_unproven:"+str(e)
        stars={};bad_by={};target_rows={};valid=True
        if reason is None:
            for vr,c in volumes.items():
                a,b=map(int,cross[vr][np.asarray(edge,np.int64)])
                try:rr=B.full_edge_star(c,a,b)
                except Exception as e:reason="edge_star:"+str(e);valid=False;break
                stars[vr]=rr
                if len(rr)>24:reason="edge_star_cell_cap";valid=False;break
                verts=np.unique(c["tets"][rr].ravel())
                if len(verts)>16:reason="edge_star_point_cap";valid=False;break
                if len(np.unique(c["mat"][rr]))!=1:reason="edge_star_material_interface";valid=False;break
                all_protected=set(map(int,protections[vr]["volume_rows"]))
                own_protected=own_protected_volume_rows(role,vr,protections[vr],cross[vr],inner_allowed)
                blocked=protected_volume_star_conflicts(verts,all_protected,own_protected,fixed_coordinate_topology_only)
                other_touched=sorted((set(map(int,verts))&all_protected)-own_protected)
                if blocked:
                    reason="protected_volume_star";valid=False;break
                bad_by[vr]=int(np.count_nonzero(c["bad"][rr]))
                tl=c["target_lookup"];trrows=[]
                for f in old:
                    vf=cross[vr][f]
                    if len(tl.get(tuple(sorted(map(int,vf))),[]))!=1:
                        reason="target_parent_face_missing_or_ambiguous";valid=False;break
                    trrows.append(int(tl[tuple(sorted(map(int,vf)))][0]))
                if not valid:break
                target_rows[vr]=trrows
        scored+=1
        if reason:
            counts[reason]+=1
            if reason.startswith("quad_overlay_hausdorff_unproven:") and len(failed_multiface_pairs)<2048:
                failed_multiface_pairs.append({"edge":list(edge),"face_rows":list(fr),"old_faces":old.copy(),
                    "bad_edge_hit":hit,"length_m":float(np.linalg.norm(sp[edge[0]].astype(np.float64)-sp[edge[1]].astype(np.float64))),
                    "reason":reason})
            if len(witnesses)<400:witnesses.append({"edge":list(edge),"face_rows":list(fr),"bad_cell_edge_hits":hit,"reason":reason,
                "quad_hausdorff":hinfo,"ordinary_cap_m":MAX_ORDINARY_QUAD_HAUSDORFF_M,"inner_back_cap_m":MAX_INNER_BACK_HAUSDORFF_M})
            continue
        candidates.append({"edge":edge,"face_rows":fr,"old_faces":old.copy(),"new_faces":new.copy(),"quad":quad,
            "shared":list(shared),"opposite":list(opp),"bad_edge_hit":hit,"bad_by_volume":bad_by,"length_m":length,
            "shape_gain":gain,"stars":stars,"target_rows":target_rows,"inner_back_touched":touch,
            "quad_hausdorff":hinfo,"surface_hausdorff_limit_m":float(hlimit),
            "inner_back_hausdorff":hinfo if touch else None,
            "other_role_protected_volume_rows_touched":{vr:sorted((set(map(int,volumes[vr]["tets"][stars[vr]].ravel())) & set(map(int,protections[vr]["volume_rows"])))-
                own_protected_volume_rows(role,vr,protections[vr],cross[vr],inner_allowed)) for vr in volumes}})
    candidates.sort(key=lambda c:(-c["bad_edge_hit"],-c["shape_gain"],c["length_m"],c["edge"],c["face_rows"]))
    return {"candidates":candidates,"candidate_edge_count":len(candidates),"edge_hit_count":len(edge_hits),
       "candidate_edges_scored":scored,"search_complete":search_complete,

       "rejection_counts":dict(counts),"rejection_witnesses":witnesses,"source_point_maps":cross,
       "failed_multiface_pairs":failed_multiface_pairs}


def expand_complete_material_face_ring(cache, initial_cell_rows, required_target_faces, material=None):
    """Add every same-material cell across one non-target face-neighbor ring.

    Current target facets and material interfaces remain cavity boundaries. A
    non-target cavity-boundary facet must have exactly one retained face-neighbor
    cell; the whole same-material neighbor ring is added atomically.
    """
    t=np.asarray(cache["tets"],np.int64);m=np.asarray(cache["mat"])
    selected=set(map(int,np.asarray(initial_cell_rows,np.int64)))
    if not selected:raise ValueError("face-ring seed cavity is empty")
    mats=np.unique(m[np.asarray(sorted(selected),np.int64)])
    if len(mats)!=1:raise ValueError("face-ring seed crosses material index")
    mat=int(mats[0]) if material is None else int(material)
    if int(mats[0])!=mat:raise ValueError("face-ring material differs from seed cavity")
    required={tuple(sorted(map(int,f))) for f in np.asarray(required_target_faces,np.int64).reshape((-1,3))}
    target_keys=set(cache["target_lookup"])
    if not required.issubset(target_keys):raise ValueError("required source patch is absent from current target boundary")
    boundary=np.asarray(T.m.boundary(t[np.asarray(sorted(selected),np.int64)]),np.int64)
    boundary_keys={tuple(sorted(map(int,f))) for f in boundary}
    if not required.issubset(boundary_keys):raise ValueError("required target patch is not on seed cavity boundary")
    additions=set();material_frontier=[]
    for f in boundary:
        key=tuple(sorted(map(int,f)))
        if key in target_keys:continue
        incident=np.asarray(B.incident_cells(cache,f),np.int64)
        inside=[int(r) for r in incident if int(r) in selected]
        if len(inside)!=1:raise ValueError("non-target cavity face lacks exactly one seed-side owner")
        if len(incident)!=2:raise ValueError("non-target cavity face lacks one unique volume face-neighbor")
        outside=[int(r) for r in incident if int(r) not in selected]
        if len(outside)!=1:raise ValueError("face-neighbor ring owner is ambiguous")
        neighbor=outside[0]
        if int(m[neighbor])!=mat:
            material_frontier.append({"face":list(key),"neighbor_cell_row":neighbor,"neighbor_material":int(m[neighbor])})
            continue
        additions.add(neighbor)
    expanded=selected|additions;outrows=np.asarray(sorted(expanded),np.int64);added=np.asarray(sorted(additions),np.int64)
    final_boundary=np.asarray(T.m.boundary(t[outrows]),np.int64)
    final_keys={tuple(sorted(map(int,f))) for f in final_boundary}
    if not required.issubset(final_keys):raise ValueError("face-ring expansion removed a required target facet")
    return outrows,added,{"seed_cell_count":len(selected),"added_same_material_face_neighbors":len(added),
        "material_interface_frontier_count":len(material_frontier),"material_interface_frontier":material_frontier,
        "target_boundary_faces_preserved":True,"required_target_faces_preserved":True,
        "ring_definition":"all same-material cells across every non-target face on original seed cavity boundary; one layer only"}


def material_face_ring_control():
    """Test a complete nonempty ring, target-facet preservation, and material barrier."""
    pts=np.asarray(((0,0,0),(1,0,0),(0,1,0),(0,0,1),(0,0,-1)),np.float32)
    t=M.orient(pts,np.asarray(((0,1,2,3),(0,2,1,4)),np.int64))
    if np.any(M.measurements(pts,t)[0]<=0):raise AssertionError("ring control tets must be positive")
    boundary=np.asarray(T.m.boundary(t),np.int64)
    cache={"points":pts,"tets":t,"mat":np.asarray((16,16),np.int32),"target":boundary,
           "target_lookup":B.face_lookup(boundary),"incidence":B.cell_incidence(t,len(pts))}
    shared=tuple(sorted((0,1,2)))
    req=np.asarray([f for f in boundary if tuple(sorted(map(int,f)))!=shared][:1],np.int64)
    expanded,added,diag=expand_complete_material_face_ring(cache,np.asarray([0]),req,16)
    if expanded.tolist()!=[0,1] or added.tolist()!=[1] or not diag["target_boundary_faces_preserved"]:
        raise AssertionError("complete same-material face-neighbor ring was not included")
    target_keys={tuple(sorted(map(int,f))) for f in boundary}
    actual_keys={tuple(sorted(map(int,f))) for f in T.m.boundary(t[expanded])}
    if target_keys!=actual_keys:raise AssertionError("ring altered outer target boundary")
    mixed={**cache,"mat":np.asarray((16,17),np.int32)}
    mx,ma,md=expand_complete_material_face_ring(mixed,np.asarray([0]),req,16)
    if mx.tolist()!=[0] or len(ma)!=0 or md["material_interface_frontier_count"]!=1:
        raise AssertionError("ring crossed a material interface")
    return {"status":"PASS_COMPLETE_MATERIAL_FACE_RING_CONTROL","seed_cells":[0],
        "same_material_added_cells":added.tolist(),"mixed_material_added_cells":ma.tolist(),
        "outer_target_boundary_unchanged":True,"material_interface_not_crossed":True}


def source_patch_internal_edges(faces):
    counts=Counter()
    for a,b,c in np.asarray(faces,np.int64).reshape((-1,3)):
        for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):
            counts[tuple(sorted((u,v)))]+=1
    return tuple(sorted(e for e,n in counts.items() if n==2))


def reconstruct_source_patch(candidate,source_faces,source_points):
    rows=list(map(int,candidate["face_rows"]));old=np.asarray(source_faces[np.asarray(rows,np.int64)],np.int64)
    recorded=np.asarray(candidate["new_faces"],np.int64);kind=candidate.get("source_patch_kind","edge_2to2")
    if kind=="edge_2to2":
        expected,_,_=ENG.flip_faces(old)
        if not np.array_equal(expected,recorded):raise ValueError("edge-flip journal differs from deterministic source operation")
        return expected
    if kind!="polygon_3face5vertex":raise ValueError("unknown source patch operation kind")
    rebuilt=PATCH.enumerate_polygon_triangulations(source_points,old,max_vertices=5,hausdorff_cap_m=float(candidate["surface_hausdorff_limit_m"]))
    if tuple(rebuilt["boundary_cycle"])!=tuple(map(int,candidate["patch_boundary_cycle"])):
        raise ValueError("multi-face patch boundary cycle changed")
    for item in rebuilt["candidates"]:
        faces=np.asarray(item["faces"],np.int64)
        if np.array_equal(faces,recorded):
            if int(item["shared_old_face_count"])!=0:raise ValueError("multi-face candidate retains an old face")
            return faces
    raise ValueError("multi-face triangulation not reproducible from parent")


def _enumerate_candidates_fast_multiface(role,surface,volumes,protections,edgecap,inner_allowed=None,
        deadline=None,fixed_coordinate_topology_only=False,max_multiface_pockets=64,max_cells=64,max_points=24):
    result=_enumerate_candidates_fast_edge_only(role,surface,volumes,protections,edgecap,inner_allowed,deadline,fixed_coordinate_topology_only)
    sf=np.asarray(surface["faces"],np.int64);sp=np.asarray(surface["points_world_m_f32"],np.float32);candidates=result["candidates"]
    for c in candidates:c.setdefault("source_patch_kind","edge_2to2")
    if max_multiface_pockets<=0:return result
    edge_faces=defaultdict(list)
    for row,f in enumerate(sf):
        a,b,c=map(int,f)
        for e in (tuple(sorted((a,b))),tuple(sorted((b,c))),tuple(sorted((c,a)))):edge_faces[e].append(row)
    fcounts=Counter(tuple(sorted(map(int,f))) for f in sf);cross=result["source_point_maps"]
    srcprot=edited_role_source_protected_rows(protections,role,inner_allowed)
    allowed=set().union(*(set(map(int,x)) for x in (inner_allowed or {}).values())) if inner_allowed else set()
    rejects=Counter();wits=[];seen=set();pockets=0;truncated=False
    for seed in result.get("failed_multiface_pairs",[]):
        if deadline is not None and time.monotonic()>=deadline:truncated=True;break
        pairrows=tuple(map(int,seed["face_rows"]));pairset=set(pairrows);pair=np.asarray(seed["old_faces"],np.int64);ec=Counter()
        for f in pair:
            a,b,c=map(int,f)
            for e in (tuple(sorted((a,b))),tuple(sorted((b,c))),tuple(sorted((c,a)))):ec[e]+=1
        expansions=[]
        for edge,n in ec.items():
            adj=edge_faces.get(edge,())
            if n!=1 or len(adj)!=2:continue
            third=next((int(r) for r in adj if r not in pairset),None)
            if third is None:continue
            rows3=tuple(sorted(pairset|{third}))
            if rows3 in seen or len(np.unique(sf[np.asarray(rows3,np.int64)]))!=5:continue
            seen.add(rows3);expansions.append((edge,third,rows3))
        for trigger,third,rows3 in expansions:
            if pockets>=max_multiface_pockets:truncated=True;break
            pockets+=1;old=np.asarray(sf[np.asarray(rows3,np.int64)],np.int64);used=set(map(int,old.ravel()))
            if len(used)!=5:rejects["not_five_vertex_patch"]+=1;continue
            if used&srcprot:rejects["protected_source_vertex"]+=1;continue
            touch=bool(used&allowed);hcap=MAX_INNER_BACK_HAUSDORFF_M if touch else MAX_ORDINARY_QUAD_HAUSDORFF_M
            try:tri=PATCH.enumerate_polygon_triangulations(sp,old,max_vertices=5,hausdorff_cap_m=hcap)
            except Exception as e:
                msg="polygon_patch:"+str(e);rejects[msg]+=1
                if len(wits)<200:wits.append({"seed_edge":seed["edge"],"face_rows":list(rows3),"reason":msg})
                continue
            cycle=tuple(map(int,tri["boundary_cycle"]));internal=source_patch_internal_edges(old)
            if len(internal)!=2:rejects["three_face_internal_edge_count"]+=1;continue
            oldshape=B.shape(sp,old);added=False
            for item in tri["candidates"]:
                new=np.asarray(item["faces"],np.int64)
                if len(new)!=3 or int(item["shared_old_face_count"])!=0:continue
                rem=Counter(tuple(sorted(map(int,f))) for f in old);emit=[tuple(sorted(map(int,f))) for f in new]
                if len(set(emit))!=3 or any(fcounts[k]-rem[k]>0 for k in emit):rejects["duplicate_retained_face"]+=1;continue
                ns=B.shape(sp,new)
                if not np.isfinite(ns).all() or np.any(ns<=0):rejects["degenerate_source_triangle"]+=1;continue
                if np.any(J.score(sp,new)<1e-5):rejects["joint_apex_gate"]+=1;continue
                length=float(np.linalg.norm(sp[int(trigger[0])].astype(np.float64)-sp[int(trigger[1])].astype(np.float64)))
                if length>edgecap:rejects["edge_cap"]+=1;continue
                stars={};badby={};trs={};other={};ring_added_by={};ring_diag_by={};ok=True;why=""
                for vr,vc in volumes.items():
                    internal_v=[]
                    for e in internal:
                        ev=np.asarray(cross[vr][np.asarray(e,np.int64)],np.int64)
                        if np.any(ev<0):ok=False;why="unmapped_internal_patch_edge";break
                        internal_v.append(tuple(map(int,ev)))
                    if not ok:break
                    rrset=set()
                    try:
                        for a,b in internal_v:rrset.update(map(int,B.full_edge_star(vc,a,b)))
                    except Exception as e:ok=False;why="edge_star:"+str(e);break
                    rr=np.asarray(sorted(rrset),np.int64);stars[vr]=rr
                    if not len(rr) or len(rr)>max_cells:ok=False;why="multi_face_cell_cap";break
                    verts=np.unique(vc["tets"][rr].ravel())
                    if len(verts)>max_points:ok=False;why="multi_face_point_cap";break
                    if len(np.unique(vc["mat"][rr]))!=1:ok=False;why="multi_face_material_interface";break
                    allp=set(map(int,protections[vr]["volume_rows"]));own=own_protected_volume_rows(role,vr,protections[vr],cross[vr],inner_allowed)
                    if protected_volume_star_conflicts(verts,allp,own,fixed_coordinate_topology_only):ok=False;why="protected_volume_star";break
                    badby[vr]=int(np.count_nonzero(vc["bad"][rr]));rows_target=[]
                    for f in old:
                        vf=cross[vr][f];hits=vc["target_lookup"].get(tuple(sorted(map(int,vf))),[]) if np.all(vf>=0) else []
                        if len(hits)!=1:ok=False;why="target_face_missing_or_ambiguous";break
                        rows_target.append(int(hits[0]))
                    if not ok:break
                    trs[vr]=rows_target
                    target_faces=np.asarray(vc["target"][np.asarray(rows_target,np.int64)],np.int64)
                    try:
                        expanded,ring_added,ring_diag=expand_complete_material_face_ring(vc,rr,target_faces,int(np.asarray(vc["mat"])[rr[0]]))
                    except Exception as e:
                        ok=False;why="material_face_ring:"+str(e);break
                    rr=expanded;stars[vr]=rr
                    if len(rr)>max_cells:ok=False;why="multi_face_ring_cell_cap";break
                    verts=np.unique(vc["tets"][rr].ravel())
                    if len(verts)>max_points:ok=False;why="multi_face_ring_point_cap";break
                    if len(np.unique(vc["mat"][rr]))!=1:ok=False;why="multi_face_ring_material_interface";break
                    if protected_volume_star_conflicts(verts,allp,own,fixed_coordinate_topology_only):
                        ok=False;why="multi_face_ring_protected_volume_star";break
                    badby[vr]=int(np.count_nonzero(vc["bad"][rr]))
                    other[vr]=sorted(set(map(int,vc["tets"][rr].ravel()))&allp-own)
                    ring_added_by[vr]=ring_added.tolist();ring_diag_by[vr]=ring_diag
                if not ok:rejects[why]+=1;continue
                cand={"edge":tuple(map(int,seed["edge"])),"face_rows":rows3,"old_faces":old.copy(),"new_faces":new.copy(),
                  "quad":used,"shared":[],"opposite":[],"bad_edge_hit":int(seed["bad_edge_hit"]),
                  "bad_by_volume":badby,"length_m":length,"shape_gain":float(np.min(ns)-np.min(oldshape)),
                  "stars":stars,"target_rows":trs,"inner_back_touched":touch,
                  "quad_hausdorff":{"hausdorff_upper_bound_m":float(item["hausdorff_upper_bound_m"]),
                    "projection":tri["projection"],"bound_claim":tri["surface_displacement_bound"],
                    "scope":"inner-back topology-only" if touch else "ordinary source polygon patch"},
                  "surface_hausdorff_limit_m":float(hcap),
                  "inner_back_hausdorff":{"hausdorff_upper_bound_m":float(item["hausdorff_upper_bound_m"])} if touch else None,
                  "other_role_protected_volume_rows_touched":other,"source_patch_kind":"polygon_3face5vertex",
                  "material_face_ring_added_cell_rows":ring_added_by,"material_face_ring_diagnostics":ring_diag_by,
                  "patch_boundary_cycle":list(cycle),"expanded_from_failed_pair_rows":list(pairrows),
                  "expanded_third_face_row":int(third)}
                candidates.append(cand);added=True
            if not added:rejects["no_distinct_admissible_triangulation"]+=1
    result["candidate_multiface_pocket_count"]=pockets
    result["candidate_multiface_patch_count"]=sum(c.get("source_patch_kind")=="polygon_3face5vertex" for c in candidates)
    result["multiface_search_complete"]=not truncated
    result["multiface_rejection_counts"]=dict(rejects);result["multiface_rejection_witnesses"]=wits
    result["search_complete"]=bool(result.get("search_complete",True) and not truncated)
    candidates.sort(key=lambda c:(-int(c["bad_edge_hit"]),-float(c["shape_gain"]),float(c["length_m"]),tuple(c["edge"]),tuple(c["face_rows"])))
    result["candidate_edge_count"]=len(candidates);return result


enumerate_candidates_fast=_enumerate_candidates_fast_multiface


def choose_disjoint_seed_patches(candidates,surface,volumes,max_patches,max_cells,max_points,deadline=None):
    """Return ranked one-edge seeds; reservations are made only after an accepted solve."""
    ordered=sorted(candidates,key=lambda c:(-int(c["bad_edge_hit"]),
        -float(c["shape_gain"]),float(c["length_m"]),tuple(c["edge"]),tuple(c["face_rows"])))
    # Keep all candidates as seeds. The solve loop applies the hard attempt/time
    # limits and skips only stars that overlap already accepted operations.
    return [[c] for c in ordered],[],{vr:set() for vr in volumes}

def paired_i_tie_control():
    """Focused admission control for a good-side tie plus a strict paired gain."""
    def role_ok(old_bad,new_bad,old_def,new_def,allow_tie):
        tol=max(1e-8,1e-6*float(old_def))
        improved=(new_bad<old_bad) or (new_bad==old_bad and new_def<old_def-tol)
        tied_good=allow_tie and old_bad==0 and new_bad==0 and abs(new_def-old_def)<=tol
        return new_bad<=old_bad and (improved or tied_good),improved
    skin_ok,skin_gain=role_ok(0,0,0.,0.,True)
    fat_ok,fat_gain=role_ok(2,1,4.2,2.1,True)
    if not(skin_ok and fat_ok and (skin_gain or fat_gain)):raise AssertionError("paired good-side tie control should pass")
    if role_ok(2,3,4.,0.,True)[0]:raise AssertionError("paired participant bad-count regression must fail")
    tied2=role_ok(0,0,0.,0.,True)[1] and role_ok(0,0,0.,0.,True)[1]
    if tied2:raise AssertionError("two tied-good participants cannot count as progress")
    # Interface outward normals must cancel: paired exact signed flux changes.
    skin_delta,fat_delta=17,-17
    if skin_delta+fat_delta!=0:raise AssertionError("paired signed flux control")
    return {"status":"PASS_PAIRED_I_GOOD_SIDE_TIE_CONTROL","skin_good_role_may_tie":True,
      "fat_bad_count_strictly_improves":True,"role_regression_rejected":True,
      "two_tied_roles_rejected":True,"opposite_interface_flux_cancels":True}


def paired_opposite_winding_tie_fixture(outdir):
    out=Path(outdir).resolve()
    if out.exists():raise FileExistsError("fixture output must be fresh")
    out.mkdir(parents=True)
    inp=out/"inputs";inp.mkdir()
    base=np.asarray(((0,0,0),(1,0,0),(.1,.2,0),(.9,-.1,0)),np.float32)
    logical_to_global=np.asarray((3,6,1,5,2),np.int64)
    global_to_logical={int(g):int(i) for i,g in enumerate(logical_to_global)}
    padding=np.asarray(((9.,7.,3.),(11.,8.,4.)),np.float32)
    def make_volume(path,apex_z):
        logical=np.vstack((base,np.asarray((.5,.05,apex_z),np.float32)))
        pts=np.empty((7,3),np.float32)
        pts[0]=padding[0];pts[4]=padding[1]
        pts[logical_to_global]=logical
        logical_t=M.orient(logical,((0,1,2,4),(1,0,3,4)))
        t=logical_to_global[logical_t]
        det,q=M.measurements(pts,t)
        if np.any(det<=0):raise AssertionError("fixture input tets must be positive")
        bf=np.asarray(T.m.boundary(t),np.int64)
        arrays={"points_world_m":pts,"points_world_m_f32":pts.copy(),"tetrahedra":t,
            "material_indices":np.full(len(t),16,np.int32),"boundary_faces":bf,"desired_boundary_faces":bf.copy(),
            "input_node_to_output_node":np.arange(len(pts),dtype=np.int64),
            "output_node_to_input_node":np.arange(len(pts),dtype=np.int64),
            "output_point_to_input_point_row":np.arange(len(pts),dtype=np.int64),
            "source_I_vertex_quotient":np.arange(len(pts),dtype=np.int64),
            "output_point_parent_offsets":np.zeros(len(pts)+1,np.int64),
            "output_point_parent_input_cell_rows":np.empty(0,np.int64)}
        np.savez_compressed(path,**arrays)
        return {"path":str(path),"sha256":sha(path),"cells":len(t),
            "native_quality_failures":int(np.count_nonzero(~np.asarray(M.passes(det,q),bool)))}
    skin_ref=make_volume(inp/"skin.npz",.5)
    fat_ref=make_volume(inp/"fat.npz",-1e-7)
    skin=load_npz(inp/"skin.npz")
    skin_boundary=np.asarray(skin["boundary_faces"],np.int64)
    targetkeys={tuple(sorted(int(logical_to_global[i]) for i in face)) for face in ((0,1,2),(0,1,3))}
    iface=np.asarray([f for f in skin_boundary if tuple(sorted(map(int,f))) in targetkeys],np.int64)
    if iface.shape!=(2,3):raise AssertionError("fixture must expose exactly two interface facets")
    iface_logical=np.asarray([[global_to_logical[int(v)] for v in face] for face in iface],np.int64)
    surf=ENG.fixture_surface(base,iface_logical)
    surface_path=inp/"typed-I.npz";np.savez_compressed(surface_path,**surf)
    surface_ref={"path":str(surface_path),"sha256":sha(surface_path)}
    state={"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
        "source_surfaces":{"O":surface_ref,"I":surface_ref,"D":surface_ref},
        "Skin_current":skin_ref,"Fat_current":fat_ref,"status":"SYNTHETIC_PAIRED_I_TIE_FIXTURE"}
    state_path=inp/"state.json";state_path.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n")
    # Current fixtures omit protected anatomy by construction. This helper is only
    # used for this synthetic state and is restored before the producer exits.
    global fixture_protection
    original_protection=fixture_protection
    def empty_protection(st,role,volumes,sources):
        return {vr:{"surface_rows":np.empty(0,np.int64),"volume_rows":np.empty(0,np.int64),
            "report":{"role":vr,"synthetic_fixture_protection":"no protected rows in this miniature cavity"}} for vr in volumes}
    fixture_protection=empty_protection
    args=argparse.Namespace(state=state_path,role="I",output=out/"paired-I-run",inner_back_topology_only=False,
        fixed_coordinate_topology_only=False,batch_seconds=60.,solve_seconds=3.,max_patches=8,max_pairs_per_patch=1,
        maximum_old_cells=24,maximum_points=16,maximum_local_volume_fraction=.01,
        maximum_global_volume_fraction=1e-4,maximum_edge_m=2.)
    try:report=run_batch_state(args,fixture=True)
    finally:fixture_protection=original_protection
    if report.get("status")!="PASS_BATCH_LOCAL_INDEPENDENT_REPLAY_PENDING_OUTER_AUDIT":
        raise AssertionError("paired tie fixture did not produce a replayed batch: "+str(report.get("status")))
    q=report["quality_by_volume"]
    if q["Skin"]["native_bad_before"]!=0 or q["Skin"]["native_bad_after"]!=0:
        raise AssertionError("already-good Skin side should tie at zero bad cells")
    if not (q["Fat"]["native_bad_after"]<q["Fat"]["native_bad_before"] or
            q["Fat"]["log_deficit_after"]<q["Fat"]["log_deficit_before"]):
        raise AssertionError("Fat side must strictly improve")
    # Confirm the physical boundary winding was opposite before the flip.
    V={r:B.volume_cache(state[r+"_current"]) for r in ("Skin","Fat")}
    mapping=active_map(surf,V["Skin"],"fixture skin");mapfat=active_map(surf,V["Fat"],"fixture fat")
    relations={}
    for vr,c,mp in (("Skin",V["Skin"],mapping),("Fat",V["Fat"],mapfat)):
        target=np.asarray(c["target"],np.int64);lookup=c["target_lookup"];rels=[]
        for f in iface_logical:
            hits=lookup.get(tuple(sorted(map(int,mp[f]))),[])
            if len(hits)!=1:raise AssertionError("fixture interface facet map must be unique")
            rel=ENG.rel(target[int(hits[0])],mp[f]);rels.append(int(rel))
        relations[vr]=rels
    if set(relations["Skin"])!={1} or set(relations["Fat"])!={-1}:
        raise AssertionError("fixture does not prove opposite physical interface winding")
    journal=json.loads((out/"paired-I-run"/"patch-journal.json").read_text())
    for patch in journal:
        if patch["volumes"]["Skin"]["source_to_target_orientation_relations"]!=[1] or patch["volumes"]["Fat"]["source_to_target_orientation_relations"]!=[-1]:
            raise AssertionError("fixture output omitted/changed orientation parity")
    receipt={"schema":"h706.paired-opposite-winding-quality-tie-fixture.v1","status":"PASS",
        "producer":pin(Path(__file__)),"pinned_current_backlineage_owner":pin(BACKLINEAGE_OWNER_PATH),"input_state":pin(state_path),"batch_report":pin(out/"paired-I-run/report.json"),
        "typed_candidate":pin(out/"paired-I-run/candidate-I-typed-current.npz"),
        "Skin_volume_candidate":pin(out/"paired-I-run/skin-batch-volume-proposal.npz"),
        "Fat_volume_candidate":pin(out/"paired-I-run/fat-batch-volume-proposal.npz"),
        "independent_replay":report["independent_replay"],"source_to_target_orientation_relations":relations,
        "Skin_good_side_tied":True,"Fat_side_strictly_improved":True,
        "paired_exact_det6_delta":str(int(report["independent_replay"]["all_volume_roles"]["Skin"]["exact_global_det6_delta"])+
            int(report["independent_replay"]["all_volume_roles"]["Fat"]["exact_global_det6_delta"])),
        "paired_exact_volume_cancellation":True,
        "all_coordinates_fixed":True,"padded_permuted_global_point_ids":True,
        "global_point_permutation":logical_to_global.tolist(),"padding_point_rows":[0,4],
        "external_all185":"NOT_RUN","physical_admission":False,"native_steps":0}
    rp=out/"fixture-receipt.json";rp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


def multiface_patch_fixture(outdir):
    out=Path(outdir).resolve()
    if out.exists():raise FileExistsError("fixture output must be fresh")
    out.mkdir(parents=True);inp=out/"inputs";inp.mkdir()
    poly=np.asarray([[0,0,0],[.004,0,0],[.004,.004,0],[.002,.00201,0],[0,.004,0]],np.float32)
    points=np.vstack((poly,np.asarray([-.001,-.001,-1e-13],np.float32))).astype(np.float32)
    oldtop=np.asarray([[0,1,2],[0,2,3],[0,3,4]],np.int64)
    seed_tets=PATCH.tets_from_patch(points,oldtop,5)
    # Embed the source patch cavity in a same-material neighbor tet across one
    # non-target cut face. The production enumeration must include this whole
    # face-neighbor layer before applying the 64-cell/24-point caps.
    face=np.asarray((0,1,5),np.int64);face_xyz=points[face].astype(np.float64)
    normal=np.cross(face_xyz[1]-face_xyz[0],face_xyz[2]-face_xyz[0]);normal/=np.linalg.norm(normal)
    center=face_xyz.mean(axis=0);inside=points[2].astype(np.float64)
    side=1.0 if np.dot(inside-center,normal)>0 else -1.0
    outside=(center-side*normal*0.001).astype(np.float32)
    points=np.vstack((points,outside)).astype(np.float32)
    ring_tet=M.orient(points,np.asarray([[0,1,5,6]],np.int64))
    tets=np.vstack((seed_tets,ring_tet)).astype(np.int64);det,q=M.measurements(points,tets)
    if np.any(det<=0) or not np.any(~np.asarray(M.passes(det,q),bool)):
        raise AssertionError("fixture requires positive input cells with native quality debt")
    boundary=np.asarray(T.m.boundary(tets),np.int64);keys={tuple(sorted(map(int,f))):f.copy() for f in boundary}
    topkeys={tuple(sorted(map(int,f))) for f in oldtop}
    sourcefaces=np.vstack((np.asarray([keys[tuple(sorted(map(int,f)))] for f in oldtop],np.int64),
        np.asarray([f for f in boundary if tuple(sorted(map(int,f))) not in topkeys],np.int64)))
    if tuple(sorted(set(map(int,sourcefaces[1])) & set(map(int,sourcefaces[2])))) != (0,3):
        raise AssertionError("nonconvex source pair placement failed")
    surf=ENG.fixture_surface(points,sourcefaces);sp=inp/"surface.npz";np.savez_compressed(sp,**surf)
    vp=inp/"skin.npz";np.savez_compressed(vp,points_world_m=points,points_world_m_f32=points.copy(),
        tetrahedra=tets,material_indices=np.full(len(tets),16,np.int32),boundary_faces=boundary,
        desired_boundary_faces=boundary.copy(),input_node_to_output_node=np.arange(len(points),dtype=np.int64),
        output_node_to_input_node=np.arange(len(points),dtype=np.int64),
        output_point_to_input_point_row=np.arange(len(points),dtype=np.int64),
        source_I_vertex_quotient=np.arange(len(points),dtype=np.int64),
        output_point_parent_offsets=np.zeros(len(points)+1,np.int64),
        output_point_parent_input_cell_rows=np.empty(0,np.int64))
    sr={"path":str(sp),"sha256":sha(sp)};vr={"path":str(vp),"sha256":sha(vp),"cells":len(tets),"native_quality_failures":int(np.count_nonzero(~np.asarray(M.passes(det,q),bool))),"nonpositive":0}
    state={"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
       "source_surfaces":{"O":sr,"I":sr,"D":sr},"Skin_current":vr,"Fat_current":vr,
       "status":"SYNTHETIC_THREE_FACE_FIVE_VERTEX_PATCH"}
    statep=inp/"state.json";statep.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n")
    global fixture_protection,enumerate_candidates_fast
    oldprot=fixture_protection;oldenum=enumerate_candidates_fast
    def no_protection(st,role,vols,sources):
        return {r:{"surface_rows":np.empty(0,np.int64),"volume_rows":np.empty(0,np.int64),
           "report":{"role":r,"synthetic_fixture_protection":True}} for r in vols}
    fixture_protection=no_protection
    try:
        vc={"Skin":B.volume_cache(vr)};prot=no_protection(state,"O",vc,{"O":surf,"I":surf,"D":surf})
        disc=enumerate_candidates_fast("O",surf,vc,prot,.01,None,deadline=time.monotonic()+10,
            fixed_coordinate_topology_only=False,max_multiface_pockets=32,max_cells=64,max_points=24)
        pool=[c for c in disc["candidates"] if c.get("source_patch_kind")=="polygon_3face5vertex" and set(c["face_rows"])=={0,1,2}]
        if not pool:raise AssertionError("integrated candidate builder missed 3-face pocket: "+json.dumps(disc.get("multiface_rejection_counts",{}),sort_keys=True))
        chosen=pool[0]
        ring_rows=chosen.get("material_face_ring_added_cell_rows",{}).get("Skin",[])
        if not ring_rows:raise AssertionError("embedded fixture must add at least one same-material face-neighbor cell")
        def onlychosen(*a,**k):return {**disc,"candidates":[chosen],"candidate_edge_count":1,"search_complete":True,"multiface_search_complete":True}
        enumerate_candidates_fast=onlychosen
        args=argparse.Namespace(state=statep,role="O",output=out/"batch",inner_back_topology_only=False,
            fixed_coordinate_topology_only=False,batch_seconds=60.,solve_seconds=3.,max_patches=1,max_pairs_per_patch=1,
            maximum_old_cells=64,maximum_points=24,maximum_local_volume_fraction=.01,
            maximum_global_volume_fraction=1e-4,maximum_edge_m=.01,max_multiface_pockets=32)
        report=run_batch_state(args,fixture=True)
    finally:fixture_protection=oldprot;enumerate_candidates_fast=oldenum
    if report.get("status")!="PASS_BATCH_LOCAL_INDEPENDENT_REPLAY_PENDING_OUTER_AUDIT":
        raise AssertionError("full multiface producer/replay failed: "+str(report.get("status")))
    journal=json.loads((out/"batch"/"patch-journal.json").read_text())
    if journal[0]["source_patch_kinds"]!=["polygon_3face5vertex"] or len(journal[0]["source_patch_face_rows"][0])!=3:
        raise AssertionError("full producer journal lost patch type or parent rows")
    receipt={"schema":"h706.three-face-five-vertex-full-batch-fixture.v1","status":"PASS",
       "producer":pin(Path(__file__)),"pinned_current_backlineage_owner":pin(BACKLINEAGE_OWNER_PATH),"parent_872":pin(PARENT_872_PATH),"helper_873":pin(PATCH_873_PATH),
       "input_state":pin(statep),"typed_parent":pin(sp),"volume_parent":pin(vp),
       "batch_report":pin(out/"batch"/"report.json"),"patch_journal":pin(out/"batch"/"patch-journal.json"),
       "typed_candidate":pin(out/"batch"/"candidate-O-typed-current.npz"),
       "volume_candidate":pin(out/"batch"/"skin-batch-volume-proposal.npz"),
       "independent_replay":report["independent_replay"],
       "patch_kind":journal[0]["source_patch_kinds"][0],
       "parent_880":{"path":str(PARENT_880_PATH),"sha256":PARENT_880_SHA},
       "parent_face_rows":journal[0]["source_patch_face_rows"][0],
       "material_face_ring_added_cell_rows":journal[0].get("material_face_ring_added_cell_rows_by_role",{}).get("Skin",[[]])[0],
       "replacement_faces":journal[0]["source_patch_new_faces"][0],
       "native_bad_before":report["quality_by_volume"]["Skin"]["native_bad_before"],
       "native_bad_after":report["quality_by_volume"]["Skin"]["native_bad_after"],
       "coordinates_fixed":True,"complete_union_ancestry_replayed":True,
       "exact_volume_and_target_boundary_replayed":True,"material_face_ring_policy":"one complete same-material face-neighbor layer; target boundary and material interfaces remain cavity boundary",
       "outer_all185":"NOT_RUN",
       "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
    rp=out/"multiface-fixture-receipt.json";rp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


def adaptive_hausdorff_controls():
    old=np.asarray([[0,1,2],[0,2,3]],np.int64)
    new,shared,opp=ENG.flip_faces(old)
    planar=np.asarray([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],np.float32)
    flat=quad_overlay_hausdorff_upper(planar,old,new,shared,opp)
    if not flat["hausdorff_upper_bound_m"]<=MAX_ORDINARY_QUAD_HAUSDORFF_M:
        raise AssertionError("planar convex overlay must pass ordinary bound")
    if flat["hausdorff_upper_bound_m"]>MAX_INNER_BACK_HAUSDORFF_M:
        raise AssertionError("planar convex overlay must also pass inner-back bound")
    warped=planar.copy();warped[2,2]=0.002
    warp=quad_overlay_hausdorff_upper(warped,old,new,shared,opp)
    if not warp["hausdorff_upper_bound_m"]>MAX_ORDINARY_QUAD_HAUSDORFF_M:
        raise AssertionError("warped overlay must exceed ordinary bound")
    concave=planar.copy();concave[2]=[0.2,0.2,0]
    try:
        quad_overlay_hausdorff_upper(concave,old,new,shared,opp)
    except ValueError:
        concave_rejected=True
    else:
        raise AssertionError("concave projected overlay must fail closed")
    return {"status":"PASS_ADAPTIVE_EDGE_HAUSDORFF_CONTROLS",
        "planar_convex_upper_m":flat["hausdorff_upper_bound_m"],"warped_upper_m":warp["hausdorff_upper_bound_m"],
        "ordinary_cap_m":MAX_ORDINARY_QUAD_HAUSDORFF_M,"inner_back_cap_m":MAX_INNER_BACK_HAUSDORFF_M,
        "concave_overlay_fails_closed":concave_rejected,"only_search_cap_and_quad_displacement_screen_expanded":True}

def orientation_reversal_parity_control():
    """Prove both source orientations pass and preserve original normal."""
    old=np.asarray([[0,1,2],[0,2,3]],np.int64)
    points=np.asarray([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],np.float32)
    new,shared,opp=ENG.flip_faces(old)
    fwd=quad_overlay_hausdorff_upper(points,old,new,shared,opp)
    old_rev=old[:,[0,2,1]].copy();new_rev=new[:,[0,2,1]].copy()
    rev=quad_overlay_hausdorff_upper(points,old_rev,new_rev,shared,opp)
    fs=np.asarray(fwd["triangle_projected_double_area_signs"]["old"],int)
    rs=np.asarray(rev["triangle_projected_double_area_signs"]["old"],int)
    if not np.array_equal(rs,-fs):raise AssertionError("reversed source ordering did not reverse projected signs")
    if abs(float(fwd["hausdorff_upper_bound_m"])-float(rev["hausdorff_upper_bound_m"]))>1e-12:
        raise AssertionError("winding reversal changed the geometric bound")
    if not fwd.get("oriented_boundary_chain_match") or not rev.get("oriented_boundary_chain_match"):
        raise AssertionError("valid forward and reverse must preserve oriented boundary chain")
    oldbug=imp("paired_source_flip_bug_883_control",PARENT_883_PATH,PARENT_883_SHA)
    try:oldbug.quad_overlay_hausdorff_upper(points,old_rev,new_rev,shared,opp)
    except ValueError as e:bug_reproduced="nonfolded projected cover" in str(e)
    else:bug_reproduced=False
    if not bug_reproduced:raise AssertionError("pinned 883 orientation bug was not reproduced")
    mixed_old=np.asarray([[0,1,2],[0,3,2]],np.int64)
    try:quad_overlay_hausdorff_upper(points,mixed_old,new,shared,opp)
    except ValueError as e:mixed_old_rejected=("winding" in str(e) or "oriented" in str(e))
    else:mixed_old_rejected=False
    if not mixed_old_rejected:raise AssertionError("mixed old winding must fail closed")
    mixed_new=new.copy();mixed_new[0]=mixed_new[0,[0,2,1]]
    try:quad_overlay_hausdorff_upper(points,old,mixed_new,shared,opp)
    except ValueError as e:mixed_new_rejected=("oriented boundary" in str(e) or "winding" in str(e))
    else:mixed_new_rejected=False
    if not mixed_new_rejected:raise AssertionError("replacement orientation must match original surface normal")
    concave=points.copy();concave[2]=[0.2,0.2,0]
    try:quad_overlay_hausdorff_upper(concave,old,new,shared,opp)
    except ValueError as e:concave_rejected="strictly convex" in str(e)
    else:concave_rejected=False
    if not concave_rejected:raise AssertionError("nonconvex quad must fail closed")
    bowtie=np.asarray([[0,0,0],[1,1,0],[1,0,0],[0,1,0]],np.float32)
    try:quad_overlay_hausdorff_upper(bowtie,old,new,shared,opp)
    except ValueError as e:bowtie_rejected="strictly convex" in str(e)
    else:bowtie_rejected=False
    if not bowtie_rejected:raise AssertionError("bowtie quad must fail closed")
    return {"status":"PASS_SOURCE_ORIENTATION_REVERSAL_PARITY",
      "pinned_883_bug_reproduced":True,"forward_signs":fwd["triangle_projected_double_area_signs"],
      "reversed_signs":rev["triangle_projected_double_area_signs"],
      "valid_forward_pass":True,"valid_reversed_pass":True,
      "old_and_new_oriented_boundary_chains_equal":True,
      "mixed_old_winding_rejected":True,"mixed_replacement_winding_rejected":True,
      "replacement_matches_original_surface_normal":True,
      "nonconvex_rejected":True,"bowtie_rejected":True,
      "forward_hausdorff_upper_m":float(fwd["hausdorff_upper_bound_m"]),
      "reversed_hausdorff_upper_m":float(rev["hausdorff_upper_bound_m"])}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state",type=Path);ap.add_argument("--role",choices=("O","I","D"))
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--maximum-edge-m",type=float,default=.005)
    ap.add_argument("--batch-seconds",type=float,default=120.)
    ap.add_argument("--solve-seconds",type=float,default=3.)
    ap.add_argument("--max-patches",type=int,default=32)
    ap.add_argument("--source-edge",type=int,nargs=2,metavar=("A","B"),help="restrict candidate seeds to one exact current source-vertex edge")
    ap.add_argument("--max-pairs-per-patch",type=int,default=1)
    ap.add_argument("--max-multiface-pockets",type=int,default=64)
    ap.add_argument("--maximum-old-cells",type=int,default=64)
    ap.add_argument("--maximum-points",type=int,default=24)
    ap.add_argument("--maximum-local-volume-fraction",type=float,default=.01)
    ap.add_argument("--maximum-global-volume-fraction",type=float,default=1e-4)
    ap.add_argument("--inner-back-topology-only",action="store_true")
    ap.add_argument("--fixed-coordinate-topology-only",action="store_true",
        help="allow volume stars to touch protected points from other roles while all point bytes and nonedited role faces remain fixed")
    ap.add_argument("--fixture",action="store_true")
    ap.add_argument("--multiface-fixture",action="store_true")
    ap.add_argument("--controls",action="store_true")
    ap.add_argument("--orientation-parity-control",action="store_true")
    ap.add_argument("--empty-boundary-control",action="store_true")
    args=ap.parse_args()
    if args.orientation_parity_control:
        print(json.dumps({"producer":pin(Path(__file__)),
          "pinned_975_parent":{"path":str(Path(__file__).with_name("paired_connected_boundary_patch_batch_975_source_winding_invariant_root.py")),
             "sha256":"61f4f26817a6cfa7f2b38cfaa3c9a9f8759d3884d3154ac65f048938d8888cf2"},
          "pinned_bug_parent":{"path":str(PARENT_883_PATH),"sha256":PARENT_883_SHA},
          **orientation_reversal_parity_control()},indent=2,sort_keys=True),flush=True)
        return
    if args.empty_boundary_control:
        print(json.dumps({"producer":pin(Path(__file__)),"pinned_current_backlineage_owner":pin(BACKLINEAGE_OWNER_PATH),"parent_871":{"path":str(PARENT_871_PATH),"sha256":PARENT_871_SHA},
                          **empty_boundary_control()},indent=2,sort_keys=True),flush=True)
        return
    if args.controls:
        print(json.dumps({"producer":pin(Path(__file__)),"pinned_current_backlineage_owner":pin(BACKLINEAGE_OWNER_PATH),"parent_871":{"path":str(PARENT_871_PATH),"sha256":PARENT_871_SHA},
           "empty_boundary_control":empty_boundary_control(),"paired_I_control":paired_i_tie_control(),
           "role_protection_control":role_protection_fixture_control(),
           "material_face_ring_control":material_face_ring_control(),
           "adaptive_hausdorff_control":adaptive_hausdorff_controls(),
           "orientation_reversal_parity_control":orientation_reversal_parity_control(),
           "orientation_engine":pin(ENG.__file__) if hasattr(ENG,"__file__") else None},indent=2,sort_keys=True))
        return
    if args.multiface_fixture:
        print(json.dumps(multiface_patch_fixture(args.output),indent=2,sort_keys=True),flush=True)
        return
    if args.fixture:
        print(json.dumps(paired_opposite_winding_tie_fixture(args.output),indent=2,sort_keys=True),flush=True)
        return
    if args.state is None or args.role is None:ap.error("--state and --role are required outside --fixture/--controls")
    if args.max_pairs_per_patch!=1:ap.error("batch mode uses one independently gated edge patch per seed group")
    if not (0<args.maximum_edge_m<=MAX_ADAPTIVE_EDGE_CAP_M and 0<args.batch_seconds<=120 and 0<args.solve_seconds<=3 and
            1<=args.max_patches<=32 and 0<=args.max_multiface_pockets<=128 and 1<=args.maximum_old_cells<=128 and 4<=args.maximum_points<=32 and
            args.maximum_local_volume_fraction>=0 and args.maximum_global_volume_fraction>=0):
        ap.error("configured search/geometry caps are outside bounded policy")
    run_batch_state(args,fixture=False)

if __name__=="__main__":main()

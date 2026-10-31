#!/usr/bin/env python3
"""Import-safe fixed-coordinate local 2-3/3-2/4-4 volume repair.

Uses the caller's exact owner (certify, measurements, passes, orient, chain).
No project imports, filesystem reads, scans, or work start at import time.
Every accepted replacement keeps the full oriented cavity boundary and exact
Float32 volume, construction objective uses unchanged native quality predicates, and
each new cell inherits the complete union of the retired cells' parent rows.
Full retained-mesh SAT and anatomy/source-context checks remain caller gates.
"""
from __future__ import annotations
from collections import Counter
from itertools import combinations
from typing import Any, Mapping, Sequence
import numpy as np

FACE_PATTERN = np.asarray(((1,2,3),(0,3,2),(0,1,3),(0,2,1)),np.int64)


def build_node_cell_index(tetrahedra: np.ndarray, point_count: int) -> dict[str,np.ndarray]:
    t=np.asarray(tetrahedra,np.int64)
    if t.ndim!=2 or t.shape[1]!=4: raise ValueError("tetrahedra must be (n,4)")
    if t.size and (t.min()<0 or t.max()>=point_count): raise ValueError("point row outside domain")
    if any(len(np.unique(x))!=4 for x in t): raise ValueError("degenerate input tetrahedron")
    nodes=t.ravel(); cells=np.repeat(np.arange(len(t),dtype=np.int64),4)
    order=np.lexsort((cells,nodes)); offsets=np.r_[0,np.cumsum(np.bincount(nodes,minlength=point_count))]
    return {"node_cell_offsets":offsets.astype(np.int64),"node_cell_rows":cells[order]}


def _incident(vertices, index, point_count, cell_count):
    off=np.asarray(index["node_cell_offsets"],np.int64); rows=np.asarray(index["node_cell_rows"],np.int64)
    if off.shape!=(point_count+1,) or off[0]!=0 or off[-1]!=len(rows) or np.any(off[1:]<off[:-1]) or np.any(rows<0) or np.any(rows>=cell_count):
        raise ValueError("malformed node-cell CSR")
    vv=tuple(map(int,vertices))
    if not vv or len(set(vv))!=len(vv) or min(vv)<0 or max(vv)>=point_count: raise ValueError("bad incidence query")
    hits=[rows[off[v]:off[v+1]] for v in vv]; out=hits[0]
    for x in hits[1:]:
        out=np.intersect1d(out,x,assume_unique=True)
        if not len(out): break
    return np.asarray(out,np.int64)


def _ring(old, edge):
    d,e=edge; links=[tuple(sorted(set(map(int,t))-set((d,e)))) for t in old]
    if any(len(x)!=2 for x in links) or len(set(links))!=len(links): return None
    adj={}
    for a,b in links: adj.setdefault(a,[]).append(b); adj.setdefault(b,[]).append(a)
    if len(adj)!=len(old) or any(len(x)!=2 for x in adj.values()): return None
    start=min(adj); result=[start]; prev=start; cur=min(adj[start])
    while cur!=start:
        if cur in result: return None
        result.append(cur); nxt=adj[cur][0] if adj[cur][0]!=prev else adj[cur][1]; prev,cur=cur,nxt
    return result if len(result)==len(old) else None


def _outside(hits, cavity):
    return any(int(x) not in cavity for x in hits)


def _validate_owner(owner):
    for name in ("certify","measurements","passes","orient","chain"):
        if not callable(getattr(owner,name,None)): raise TypeError("exact owner missing "+name)


def parent_csr_from_candidate(candidate: Mapping[str,np.ndarray]):
    """Expand retained/replacement ancestry to a complete CSR, one row per cell."""
    t=np.asarray(candidate["tetrahedra"],np.int64); n=len(t)
    keys=("retained_input_cell_rows","replacement_parent_offsets","replacement_parent_cell_rows","replacement_tetrahedra")
    if not all(k in candidate for k in keys): return np.arange(n+1,dtype=np.int64),np.arange(n,dtype=np.int64)
    kept=np.asarray(candidate["retained_input_cell_rows"],np.int64)
    repl=np.asarray(candidate["replacement_tetrahedra"],np.int64).reshape((-1,4))
    po=np.asarray(candidate["replacement_parent_offsets"],np.int64); pr=np.asarray(candidate["replacement_parent_cell_rows"],np.int64)
    nr=len(repl)
    if len(kept)+nr!=n or po.shape!=(nr+1,) or po[0]!=0 or po[-1]!=len(pr) or np.any(po[1:]<po[:-1]) or np.any(po[1:]==po[:-1]) or np.any(pr<0):
        raise ValueError("canonical parent-union CSR malformed")
    if nr and not np.array_equal(t[len(kept):],repl): raise ValueError("replacement tail differs from canonical lineage")
    if "output_cell_to_input_cell" in candidate:
        expected=np.r_[kept,np.full(nr,-1,np.int64)]
        if not np.array_equal(candidate["output_cell_to_input_cell"],expected): raise ValueError("output-cell map disagrees with replacement order")
    off=np.zeros(n+1,np.int64); flat=[]
    for r in range(n):
        if r<len(kept): flat.append(int(kept[r]))
        else:
            j=r-len(kept); flat.extend(map(int,pr[po[j]:po[j+1]]))
        off[r+1]=len(flat)
    return off,np.asarray(flat,np.int64)


def _parent_csr(offsets, rows, n):
    if offsets is None and rows is None: return np.arange(n+1,dtype=np.int64),np.arange(n,dtype=np.int64)
    if offsets is None or rows is None: raise ValueError("parent offsets and rows must be supplied together")
    off=np.asarray(offsets,np.int64); rr=np.asarray(rows,np.int64)
    if off.shape!=(n+1,) or off[0]!=0 or off[-1]!=len(rr) or np.any(off[1:]<off[:-1]) or np.any(rr<0) or np.any(off[1:]==off[:-1]):
        raise ValueError("parent CSR must provide a nonempty union for every cell")
    return off,rr


def propose(owner, points, tets, old_rows, kind, node_cell_index=None):
    """Return topology-legal oriented candidates for a complete local star."""
    _validate_owner(owner); p=np.asarray(points,np.float32); t=np.asarray(tets,np.int64)
    rows=np.asarray(old_rows,np.int64).reshape(-1)
    if p.ndim!=2 or p.shape[1]!=3 or not np.all(np.isfinite(p)): raise ValueError("bad Float32 points")
    if t.ndim!=2 or t.shape[1]!=4 or (t.size and (t.min()<0 or t.max()>=len(p))): raise ValueError("bad tetrahedra")
    if not len(rows) or len(np.unique(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=len(t)): raise ValueError("bad old-cell rows")
    idx=node_cell_index or build_node_cell_index(t,len(p)); old=t[rows]; cavity=set(map(int,rows))
    if kind=="2-3":
        if len(rows)!=2: raise ValueError("2-3 requires two cells")
        shared=set(map(int,old[0]))&set(map(int,old[1]))
        if len(shared)!=3 or set(map(int,_incident(shared,idx,len(p),len(t))))!=cavity: raise ValueError("incomplete shared-face star")
        a,b,c=sorted(shared); x=list(set(map(int,old[0]))-shared); y=list(set(map(int,old[1]))-shared)
        if len(x)!=1 or len(y)!=1 or x[0]==y[0]: raise ValueError("invalid opposite vertices")
        d,e=int(x[0]),int(y[0])
        if len(_incident((d,e),idx,len(p),len(t))): raise ValueError("new 2-3 edge already exists")
        nt=owner.orient(p,np.asarray(((d,e,a,b),(d,e,b,c),(d,e,c,a)),np.int64))
        return [{"kind":kind,"old_rows":rows.copy(),"new_tets":nt,"new_edge":(d,e)}]
    if kind=="polygon":
        common=set(map(int,old[0]))
        for tet in old[1:]:common.intersection_update(map(int,tet))
        if len(common)!=2:raise ValueError("polygon requires full closed interior edge star")
        d,e=sorted(common)
        if set(map(int,_incident((d,e),idx,len(p),len(t))))!=cavity:raise ValueError("incomplete polygon edge star")
        ring=_ring(old,(d,e))
        if ring is None or len(ring)<5 or len(ring)>32:raise ValueError("invalid bounded polygon ring")
        from pathlib import Path
        import importlib.util
        spec=importlib.util.spec_from_file_location("polygon202for656",Path(__file__).resolve().parent/"fixed_boundary_volume_polygon_flip_202_root.py")
        poly=importlib.util.module_from_spec(spec);spec.loader.exec_module(poly)
        proposal=poly.best_triangulation(p,ring,d,e)
        if proposal is None:return []
        nt,score,tris=proposal
        return [{"kind":kind,"old_rows":rows.copy(),"new_tets":nt,"removed_edge":(d,e),"ring":tuple(ring),"polygon_triangles":tris}]
    if kind not in ("3-2","4-4"): raise ValueError("kind must be 2-3, 3-2, or 4-4")
    nold=3 if kind=="3-2" else 4
    if len(rows)!=nold: raise ValueError(kind+" has wrong old-cell count")
    common=set(map(int,old[0]))
    for tet in old[1:]: common.intersection_update(map(int,tet))
    if len(common)!=2: raise ValueError(kind+" requires one common edge")
    d,e=sorted(common)
    if set(map(int,_incident((d,e),idx,len(p),len(t))))!=cavity: raise ValueError("not a full edge star")
    ring=_ring(old,(d,e))
    if ring is None or len(ring)!=nold: raise ValueError("edge link is not a simple ring")
    if kind=="3-2":
        a,b,c=ring; nt=owner.orient(p,np.asarray(((a,b,c,d),(a,c,b,e)),np.int64))
        if _outside(_incident((a,b,c),idx,len(p),len(t)),cavity): raise ValueError("new separator face exists outside cavity")
        return [{"kind":kind,"old_rows":rows.copy(),"new_tets":nt,"removed_edge":(d,e),"ring":tuple(ring)}]
    options=(
        ((ring[0],ring[1],ring[2]),(ring[0],ring[2],ring[3]),(ring[0],ring[2])),
        ((ring[1],ring[2],ring[3]),(ring[1],ring[3],ring[0]),(ring[1],ring[3])),
    )
    out=[]
    for variant,(f0,f1,diagonal) in enumerate(options):
        faces=(f0,f1)
        if any(_outside(_incident(f,idx,len(p),len(t)),cavity) for f in faces): continue
        if _outside(_incident(diagonal,idx,len(p),len(t)),cavity): continue
        raw=[(*f,apex) for f in faces for apex in (d,e)]
        out.append({"kind":kind,"old_rows":rows.copy(),"new_tets":owner.orient(p,np.asarray(raw,np.int64)),"removed_edge":(d,e),"new_diagonal":tuple(diagonal),"ring":tuple(ring),"variant":variant})
    if not out: raise ValueError("both 4-4 diagonals conflict with existing incidence")
    return out


def evaluate(owner, points, tets, materials, old_rows, proposal, parent_offsets=None, parent_rows=None, retained_cell_keys=None):
    """Apply exact owner certification, fixed-boundary native gates and full parent union."""
    _validate_owner(owner); p=np.asarray(points,np.float32); t=np.asarray(tets,np.int64); mat=np.asarray(materials)
    n=len(t); rows=np.asarray(old_rows,np.int64).reshape(-1); po,pr=_parent_csr(parent_offsets,parent_rows,n)
    if mat.shape!=(n,) or not len(rows) or len(np.unique(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=n): raise ValueError("bad materials or cavity rows")
    old=t[rows]; new=np.asarray(proposal["new_tets"],np.int64).reshape((-1,4))
    verts=set(map(int,old.ravel()))
    if not len(new) or np.any(new<0) or np.any(new>=len(p)) or any(len(set(map(int,x)))!=4 for x in new): return {"accepted":False,"reason":"invalid_replacement_tetrahedra"}
    if not set(map(int,new.ravel()))<=verts: return {"accepted":False,"reason":"replacement_uses_non_cavity_point"}
    material_rows=np.unique(mat[rows])
    if len(material_rows)!=1: return {"accepted":False,"reason":"mixed_material_cavity"}
    material=int(material_rows[0]); keys=[tuple(sorted(map(int,x))) for x in new]
    if len(set(keys))!=len(keys): return {"accepted":False,"reason":"duplicate_replacement_cells"}
    keep=np.ones(n,bool);keep[rows]=False
    if retained_cell_keys is None: retained_cell_keys={(int(m),tuple(sorted(map(int,x)))) for m,x in zip(mat[keep],t[keep])}
    if any((material,k) in retained_cell_keys for k in keys): return {"accepted":False,"reason":"duplicate_retained_cell"}
    od,oq=owner.measurements(p,old); nd,nq=owner.measurements(p,new)
    op=np.asarray(owner.passes(od,oq),bool); npass=np.asarray(owner.passes(nd,nq),bool)
    if np.any(od<=0): return {"accepted":False,"reason":"old_cavity_nonpositive"}
    if np.any(nd<=0): return {"accepted":False,"reason":"replacement_F32_nonpositive"}
    # Construction search may retain failing cells while strictly improving the
    # lexicographic (native failure count, log-deficit) objective.
    newbad=int(np.count_nonzero(~npass))
    def deficit(det,q):
        ratio=np.minimum(np.minimum(np.asarray(q)/1e-5,np.asarray(det)/6e-18),1.)
        return float(-np.log(np.maximum(ratio,1e-300)).sum())
    oldscore=deficit(od,oq);newscore=deficit(nd,nq)
    oldbad=int(np.count_nonzero(~op))
    if oldbad==0 or newbad>oldbad or (newbad==oldbad and newscore>=oldscore-max(1e-8,1e-6*oldscore)): return {"accepted":False,"reason":"no_strict_lexicographic_quality_progress"}
    ok,why=owner.certify(p,old,new)
    if not ok: return {"accepted":False,"reason":"exact_m_certify:"+str(why)}
    if owner.chain(old)!=owner.chain(new): return {"accepted":False,"reason":"oriented_cavity_chain_mismatch"}
    fc=Counter();fs=Counter()
    for tet in new:
        for face in tet[FACE_PATTERN]:
            f=tuple(map(int,face)); key=tuple(sorted(f)); inv=int(f[0]>f[1])+int(f[0]>f[2])+int(f[1]>f[2])
            fc[key]+=1;fs[key]+=1 if inv%2==0 else -1
    if any(v>2 for v in fc.values()) or any(v==2 and fs[k]!=0 for k,v in fc.items()): return {"accepted":False,"reason":"replacement_complex_not_manifold"}
    union=set()
    for row in rows: union.update(map(int,pr[po[row]:po[row+1]]))
    if not union: return {"accepted":False,"reason":"empty_parent_union"}
    parent_union=np.asarray(sorted(union),np.int64); parent_out_offsets=np.arange(len(new)+1,dtype=np.int64)*len(parent_union)
    parent_out_rows=np.tile(parent_union,len(new)); retained=np.flatnonzero(keep).astype(np.int64)
    out_to_in=np.r_[retained,np.full(len(new),-1,np.int64)]; in_to_out=np.full(n,-1,np.int64);in_to_out[retained]=np.arange(len(retained))
    return {
        "accepted":True,"status":"PASS_LOCAL_FIXED_BOUNDARY_FLIP","kind":proposal["kind"],"old_cell_rows":rows.copy(),
        "new_tets":new.copy(),"new_material_indices":np.full(len(new),material,dtype=mat.dtype),
        "tetrahedra":np.vstack((t[keep],new)),"material_indices":np.r_[mat[keep],np.full(len(new),material,dtype=mat.dtype)],
        "retained_input_cell_rows":retained,"replacement_parent_offsets":parent_out_offsets,"replacement_parent_cell_rows":parent_out_rows,
        "replacement_tet_offsets":np.asarray((0,len(new)),np.int64),"replacement_tetrahedra":new.copy(),
        "output_cell_to_input_cell":out_to_in,"input_cell_to_output_cell":in_to_out,
        "native_bad_before_local":oldbad,"native_bad_after_local":int(np.count_nonzero(~npass)),"native_bad_gain_local":oldbad-newbad,"native_log_deficit_before":oldscore,"native_log_deficit_after":newscore,"native_log_deficit_gain":oldscore-newscore,
        "minimum_normalized_det_before":float(np.min(oq)),"minimum_normalized_det_after":float(np.min(nq)),
        "minimum_signed_volume_before_m3":float(np.min(od/6)),"minimum_signed_volume_after_m3":float(np.min(nd/6)),
        "exact_m_certify":"PASS","exact_local_oriented_boundary":"PASS","exact_local_signed_volume":"PASS",
        "all_replacement_native_quality_pass":bool(np.all(npass)),"construction_only":True,"native_thresholds_unchanged":True,"all_coordinates_fixed":True,"material_preserved":True,
        "complete_parent_union_rows":parent_union,"complete_parent_union_count":len(parent_union),
        **{k:v for k,v in proposal.items() if k not in ("new_tets","old_rows","kind")}
    }


def repair_local_cavity(owner, points, tets, materials, old_rows, kind, *, parent_offsets=None, parent_rows=None, node_cell_index=None, retained_cell_keys=None):
    """Try legal candidates for one complete local flip and return the best certified result."""
    _validate_owner(owner); p=np.asarray(points,np.float32);t=np.asarray(tets,np.int64);mat=np.asarray(materials)
    po,pr=_parent_csr(parent_offsets,parent_rows,len(t)); idx=node_cell_index or build_node_cell_index(t,len(p))
    candidates=propose(owner,p,t,old_rows,kind,idx); tested=[]; accepted=[]
    for c in candidates:
        r=evaluate(owner,p,t,mat,old_rows,c,po,pr,retained_cell_keys)
        tested.append({"variant":c.get("variant"),"accepted":bool(r["accepted"]),"reason":r.get("reason","PASS")})
        if r["accepted"]: accepted.append(r)
    if not accepted: return {"accepted":False,"status":"REJECT_LOCAL_FIXED_BOUNDARY_FLIP","kind":kind,"tested_variants":tested}
    accepted.sort(key=lambda r:(-int(r["native_bad_gain_local"]),-float(r["native_log_deficit_gain"]),tuple(map(int,r["new_tets"].ravel()))))
    accepted[0]["tested_variants"]=tested
    return accepted[0]


def run_controls():
    import hashlib, importlib.util, json
    from pathlib import Path
    h=Path(__file__).resolve().parents[1]
    paths={k:h/f"fixed_boundary_volume_{n}_root.py" for k,n in (("198","flip_198"),("199","edge_flip_199"),("202","polygon_flip_202"))}
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    owner=load("exact_flip_owner_198_625",paths["198"])
    load("flip_owner_199_625",paths["199"]);load("flip_owner_202_625",paths["202"])
    controls=[]
    p23=np.asarray(((0,0,0),(1,0,0),(0,1,0),(.2,.2,1e-7),(.2,.2,-1)),np.float32)
    old23=owner.orient(p23,np.asarray(((0,1,2,3),(0,2,1,4)),np.int64))
    r23=repair_local_cavity(owner,p23,old23,np.full(2,16,np.int32),(0,1),"2-3")
    assert r23["accepted"] and r23["native_bad_gain_local"]>=1 and owner.certify(p23,old23,r23["new_tets"])==(True,"PASS")
    controls.append({"kind":"2-3","bad_before":r23["native_bad_before_local"],"bad_after":r23["native_bad_after_local"]})
    p32=np.asarray(((0,0,0),(1,0,0),(0,1,0),(.5,1e-7,1),(.5,1e-7,-1)),np.float32)
    old32=owner.orient(p32,np.asarray(((3,4,0,1),(3,4,1,2),(3,4,2,0)),np.int64))
    r32=repair_local_cavity(owner,p32,old32,np.full(3,16,np.int32),(0,1,2),"3-2")
    assert r32["accepted"] and r32["native_bad_gain_local"]>=1 and owner.certify(p32,old32,r32["new_tets"])==(True,"PASS")
    controls.append({"kind":"3-2","bad_before":r32["native_bad_before_local"],"bad_after":r32["native_bad_after_local"]})
    p44=np.asarray(((-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0),(0,-.9999999,1),(0,-.9999999,-1)),np.float32)
    old44=owner.orient(p44,np.asarray(tuple((4,5,i,(i+1)%4) for i in range(4)),np.int64))
    po=np.asarray((0,2,3,5,6),np.int64);pr=np.asarray((10,11,12,11,14,15),np.int64)
    r44=repair_local_cavity(owner,p44,old44,np.full(4,16,np.int32),(0,1,2,3),"4-4",parent_offsets=po,parent_rows=pr)
    assert r44["accepted"] and r44["native_bad_gain_local"]>=1 and owner.certify(p44,old44,r44["new_tets"])==(True,"PASS")
    union=np.asarray((10,11,12,14,15),np.int64)
    assert np.array_equal(r44["complete_parent_union_rows"],union)
    assert np.array_equal(r44["replacement_parent_cell_rows"],np.tile(union,len(r44["new_tets"])))
    assert np.array_equal(r44["replacement_parent_offsets"],np.arange(len(r44["new_tets"])+1)*len(union))
    controls.append({"kind":"4-4","bad_before":r44["native_bad_before_local"],"bad_after":r44["native_bad_after_local"],"complete_parent_union_count":len(union)})
    try:
        repair_local_cavity(owner,p32,old32,np.full(3,16,np.int32),(0,1),"3-2")
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete edge star should fail closed")
    try: repair_local_cavity(owner,p23,old23,np.full(2,16,np.int32),(0,1),"2-3",parent_offsets=np.asarray((0,0,1)),parent_rows=np.asarray((0,),np.int64))
    except ValueError: pass
    else: raise AssertionError("empty parent CSR should fail closed")
    controls.append({"kind":"negative","status":"PASS_INCOMPLETE_STAR_AND_PARENT_CSR_REJECTED"})
    pins={k:{"path":str(v),"sha256":hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in paths.items()}
    return {"status":"PASS_LOCAL_FIXED_BOUNDARY_23_32_44_REPAIR_CONTROLS","controls":controls,"owner_pins":pins,"native_quality_thresholds_changed":False,"coordinates_changed":False,"complete_parent_union_propagated":True,"exact_gate":"existing owner.certify","quality_gate":"existing owner.measurements + owner.passes","physical_admission":False,"native_steps":0,"limitation":"full retained-volume SAT and complete source-context checks remain external"}


def main():
    import argparse,json
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--fixture",action="store_true"); a=ap.parse_args()
    if not a.fixture: ap.error("import-safe function module; run --fixture for controls")
    print(json.dumps(run_controls(),indent=2))


if __name__=="__main__": main()


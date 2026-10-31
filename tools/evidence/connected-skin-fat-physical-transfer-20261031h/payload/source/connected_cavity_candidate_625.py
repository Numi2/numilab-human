#!/usr/bin/env python3
"""Bounded connected-cavity candidate adapters for fixed-boundary quality repair.

This module does not launch geometry work at import. It validates candidates
from 220/230 MILP or 303 TetGen against the caller's existing exact owner.
Steiner points may be appended only inside the exact cavity. Existing global
surface points, oriented cavity chain, material and exact Float32 volume stay
fixed. Callers must supply exact replacement-self and retained-volume SAT
owners; source context and final global target replay remain external gates.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from itertools import combinations
from math import comb
from typing import Callable, Mapping, Sequence
import numpy as np

FACE_PATTERN = np.asarray(((1,2,3),(0,3,2),(0,1,3),(0,2,1)),np.int64)


def _require_owner(owner):
    for name in ("certify","measurements","passes","orient","chain"):
        if not callable(getattr(owner,name,None)):
            raise TypeError("exact owner missing "+name)


def _as_bool(result, *, kind):
    if isinstance(result,(bool,np.bool_)): return bool(result)
    if isinstance(result,Mapping):
        if "pass" in result: return bool(result["pass"])
        if "count" in result: return int(result["count"])==0
        if "location" in result: return result["location"]=="inside"
        if "status" in result: return str(result["status"]).startswith("PASS")
    raise TypeError(kind+" callback must return bool or a recognized gate mapping")


def oriented_boundary_faces(tetrahedra: np.ndarray) -> np.ndarray:
    """Return the unique outward-oriented cavity faces; reject bad face incidence."""
    t=np.asarray(tetrahedra,np.int64)
    if t.ndim!=2 or t.shape[1]!=4 or not len(t): raise ValueError("empty or malformed cavity")
    entries=defaultdict(list)
    for tet in t:
        if len(set(map(int,tet)))!=4: raise ValueError("degenerate cavity tet")
        for face in tet[FACE_PATTERN]:
            f=tuple(map(int,face)); key=tuple(sorted(f))
            inv=int(f[0]>f[1])+int(f[0]>f[2])+int(f[1]>f[2])
            entries[key].append((f,1 if inv%2==0 else -1))
    out=[]
    for key,inc in entries.items():
        if len(inc)==1: out.append(inc[0][0])
        elif len(inc)==2:
            if inc[0][1]+inc[1][1]!=0: raise ValueError("same-direction internal face")
        else: raise ValueError("nonmanifold face incidence")
    return np.asarray(out,np.int64).reshape((-1,3))


def _check_one_closed_component(boundary: np.ndarray) -> dict:
    b=np.asarray(boundary,np.int64)
    if b.ndim!=2 or b.shape[1]!=3 or len(b)==0: raise ValueError("empty cavity boundary")
    edge_inc=defaultdict(list); links=defaultdict(list); face_adj=[set() for _ in range(len(b))]
    for fi,f in enumerate(b):
        a,c,d=map(int,f)
        links[a].append((c,d)); links[c].append((d,a)); links[d].append((a,c))
        for u,v in ((a,c),(c,d),(d,a)):
            edge_inc[tuple(sorted((u,v)))].append((fi,1 if u<v else -1))
    for edge,inc in edge_inc.items():
        if len(inc)!=2 or inc[0][1]+inc[1][1]!=0: raise ValueError("cavity boundary edge is not oppositely paired")
        x,y=inc[0][0],inc[1][0]; face_adj[x].add(y); face_adj[y].add(x)
    bad_links=[]
    for v,pairs in links.items():
        adj=defaultdict(set)
        for a,b0 in pairs: adj[a].add(b0); adj[b0].add(a)
        if not adj or any(len(ns)!=2 for ns in adj.values()):
            bad_links.append(v); continue
        seen=set(); todo=[next(iter(adj))]
        while todo:
            x=todo.pop()
            if x in seen: continue
            seen.add(x); todo.extend(adj[x]-seen)
        if len(seen)!=len(adj): bad_links.append(v)
    if bad_links: raise ValueError("cavity boundary has invalid vertex link")
    seen=set(); components=0
    for seed in range(len(b)):
        if seed in seen: continue
        components+=1; todo=[seed]
        while todo:
            x=todo.pop()
            if x in seen: continue
            seen.add(x); todo.extend(face_adj[x]-seen)
    if components!=1: raise ValueError("cavity boundary is disconnected")
    return {"faces":len(b),"edges":len(edge_inc),"vertices":len(links),"components":components}


def _validate_gate_callbacks(self_check, retained_check):
    if not callable(self_check) or not callable(retained_check):
        raise ValueError("exact replacement self-SAT and retained-volume SAT callbacks are mandatory")


def evaluate_connected_cavity(
    owner,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    materials: np.ndarray,
    old_rows: Sequence[int],
    replacement_tetrahedra: np.ndarray,
    *,
    output_points: np.ndarray | None = None,
    new_point_rows: Sequence[int] | None = None,
    point_inside: Callable[[np.ndarray], object] | None = None,
    exact_replacement_self_check: Callable[[np.ndarray,np.ndarray], object] | None = None,
    exact_retained_intersection_check: Callable[[np.ndarray,np.ndarray,np.ndarray,np.ndarray,Sequence[np.ndarray]], object] | None = None,
    prior_replacement_tetrahedra: Sequence[np.ndarray] = (),
    parent_offsets: np.ndarray | None = None,
    parent_rows: np.ndarray | None = None,
    retained_cell_keys: set | None = None,
    max_new_points: int = 16,
    max_replacement_cells: int = 50000,
    max_old_cells: int = 50000,
    require_all_native: bool = True,
):
    """Certify one connected fixed-boundary retetrahedralization.

    Args:
      owner: existing exact module with certify/chain/measurements/passes/orient.
      points: parent Float32 global point array.
      tetrahedra/materials: full parent volume and material labels.
      old_rows: complete current input-cell rows for this cavity.
      replacement_tetrahedra: replacement cells in output_points row space.
      output_points: optional full point array. Its points[:len(points)] must be bit-identical.
      new_point_rows: explicit appended point rows used by this replacement. For one isolated
        cavity defaults to every appended row. Required when output_points contains points
        appended by other already-composed cavities as well.
      point_inside: exact shell point-location callback, called on each listed new point.
      exact_replacement_self_check: exact self-intersection check for new tetrahedral complex.
      exact_retained_intersection_check: callback(points, full_parent_tets, old_rows, new_tets, prior_replacements).
      parent_offsets/parent_rows: complete parent-union CSR for the current candidate; omitted
        means each current parent cell row is its own source identity.
      retained_cell_keys: optional {(material, sorted_corner_tuple)} for duplicate screening.
      require_all_native: require every replacement tet to pass the unchanged native gate.
        If false, the cavity must still have a strict native-defect count decrease.

    Returns a proposal mapping with output tets/materials, exact parent-union CSR and gates.
    It never declares source-context, full-body admission, or mass qualification.
    """
    _require_owner(owner); _validate_gate_callbacks(exact_replacement_self_check,exact_retained_intersection_check)
    p=np.asarray(points); t=np.asarray(tetrahedra,np.int64); mat=np.asarray(materials)
    if p.dtype!=np.float32 or p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all(): raise ValueError("parent points must be finite Float32 (n,3)")
    if t.ndim!=2 or t.shape[1]!=4 or (t.size and (t.min()<0 or t.max()>=len(p))): raise ValueError("parent tetrahedra outside point domain")
    if mat.shape!=(len(t),): raise ValueError("one material label required per parent tet")
    rows=np.asarray(old_rows,np.int64).reshape(-1)
    if not len(rows) or len(rows)>max_old_cells or max_old_cells>50000 or len(np.unique(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=len(t)): raise ValueError("bad or over-budget cavity cell rows")
    op=np.asarray(p if output_points is None else output_points)
    if op.dtype!=np.float32 or op.ndim!=2 or op.shape[1]!=3 or len(op)<len(p) or not np.isfinite(op).all(): raise ValueError("output points must be finite Float32 and retain the full parent point domain")
    if not np.array_equal(op[:len(p)].view(np.uint32),p.view(np.uint32)): raise ValueError("parent point prefix changed")
    appended=np.arange(len(p),len(op),dtype=np.int64)
    used_rows=np.asarray(appended if new_point_rows is None else new_point_rows,np.int64).reshape(-1)
    if len(np.unique(used_rows))!=len(used_rows) or (len(used_rows) and (used_rows.min()<len(p) or used_rows.max()>=len(op))): raise ValueError("new point rows must be distinct appended rows")
    new=np.asarray(replacement_tetrahedra,np.int64).reshape((-1,4))
    if not len(new) or len(new)>max_replacement_cells: raise ValueError("replacement cell count outside bound")
    if new.min()<0 or new.max()>=len(op) or any(len(set(map(int,x)))!=4 for x in new): raise ValueError("invalid replacement tetrahedron")
    if max_new_points<0 or max_new_points>128 or max_replacement_cells<1 or max_replacement_cells>50000: raise ValueError("hard local candidate limits exceeded")
    if len(used_rows)>max_new_points: raise ValueError("Steiner point budget exceeded")
    new_used=np.intersect1d(np.unique(new),appended,assume_unique=False)
    if not np.array_equal(np.sort(new_used),np.sort(used_rows)): raise ValueError("explicit appended-point set differs from replacement incidence")
    if len(used_rows):
        if not callable(point_inside): raise ValueError("exact point-location callback required for Steiner points")
        keys=op.view(np.uint32).reshape((-1,3))
        if len(np.unique(keys,axis=0))!=len(keys): raise ValueError("duplicate Float32 coordinate rows")
        for r in used_rows:
            if not _as_bool(point_inside(op[int(r)].copy()),kind="point-inside"): raise ValueError("Steiner point is not strictly inside exact cavity shell")
    old=t[rows]
    old_vertices=set(map(int,old.ravel()))
    permitted=old_vertices|set(map(int,used_rows))
    if not set(map(int,new.ravel()))<=permitted: raise ValueError("replacement references a point outside cavity or appended interior-point set")
    if len(np.unique(mat[rows]))!=1: raise ValueError("mixed-material cavity")
    material=mat[rows[0]]
    keys=[tuple(sorted(map(int,x))) for x in new]
    if len(set(keys))!=len(keys): raise ValueError("duplicate replacement cells")
    keep=np.ones(len(t),bool); keep[rows]=False
    if retained_cell_keys is None:
        retained_cell_keys={(int(mm),tuple(sorted(map(int,tt)))) for mm,tt in zip(mat[keep],t[keep])}
    if any((int(material),k) in retained_cell_keys for k in keys): raise ValueError("replacement duplicates retained cell")
    old_det,old_q=owner.measurements(p,old)
    new_det,new_q=owner.measurements(op,new)
    old_pass=np.asarray(owner.passes(old_det,old_q),bool)
    new_pass=np.asarray(owner.passes(new_det,new_q),bool)
    if np.any(old_det<=0): raise ValueError("parent cavity must be exact-positive before quality repair")
    if np.any(new_det<=0): raise ValueError("replacement contains Float32 nonpositive cell")
    if require_all_native and not np.all(new_pass): return {"accepted":False,"status":"REJECT_REPLACEMENT_NATIVE_DEFECT","bad_before":int(np.count_nonzero(~old_pass)),"bad_after":int(np.count_nonzero(~new_pass))}
    if np.count_nonzero(~new_pass)>=np.count_nonzero(~old_pass): return {"accepted":False,"status":"REJECT_NO_STRICT_NATIVE_GAIN","bad_before":int(np.count_nonzero(~old_pass)),"bad_after":int(np.count_nonzero(~new_pass))}
    old_boundary=oriented_boundary_faces(old); _check_one_closed_component(old_boundary)
    new_boundary=oriented_boundary_faces(new); _check_one_closed_component(new_boundary)
    if owner.chain(old)!=owner.chain(new): raise ValueError("oriented cavity boundary changed")
    exact_ok,exact_why=owner.certify(op,old,new)
    if not exact_ok: raise ValueError("existing exact owner rejected cavity: "+str(exact_why))
    self_result=exact_replacement_self_check(op,new)
    if not _as_bool(self_result,kind="replacement-self"): raise ValueError("exact replacement self-intersection")
    prior=[np.asarray(x,np.int64).reshape((-1,4)) for x in prior_replacement_tetrahedra]
    retained_result=exact_retained_intersection_check(op,t,rows,new,prior)
    if not _as_bool(retained_result,kind="retained-volume"): raise ValueError("exact replacement/retained intersection")
    po,pr=_parent_csr(parent_offsets,parent_rows,len(t))
    union=set()
    for r in rows: union.update(map(int,pr[po[r]:po[r+1]]))
    if not union: raise ValueError("empty complete parent union")
    union_rows=np.asarray(sorted(union),np.int64)
    retained=np.flatnonzero(keep).astype(np.int64)
    output_tets=np.vstack((t[keep],new))
    output_materials=np.r_[mat[keep],np.full(len(new),material,dtype=mat.dtype)]
    out_parent=[];out_offsets=[0]
    for r in retained:
        out_parent.extend(map(int,pr[po[r]:po[r+1]]));out_offsets.append(len(out_parent))
    replacement_offsets=np.arange(len(new)+1,dtype=np.int64)*len(union_rows)
    replacement_rows=np.tile(union_rows,len(new))
    for _ in new:
        out_parent.extend(map(int,union_rows));out_offsets.append(len(out_parent))
    output_to_input=np.r_[retained,np.full(len(new),-1,np.int64)]
    input_to_output=np.full(len(t),-1,np.int64);input_to_output[retained]=np.arange(len(retained))
    return {
        "accepted":True,"status":"PASS_BOUNDED_CONNECTED_CAVITY_CANDIDATE",
        "points_world_m":op,"tetrahedra":output_tets,"material_indices":output_materials,
        "retained_input_cell_rows":retained,"replacement_tetrahedra":new.copy(),
        "replacement_parent_offsets":replacement_offsets,"replacement_parent_cell_rows":replacement_rows,
        "output_cell_to_input_cell":output_to_input,"input_cell_to_output_cell":input_to_output,
        "output_parent_offsets":np.asarray(out_offsets,np.int64),"output_parent_rows":np.asarray(out_parent,np.int64),
        "old_cell_rows":rows.copy(),"complete_parent_union_rows":union_rows,
        "new_point_rows":used_rows.copy(),"native_bad_before_local":int(np.count_nonzero(~old_pass)),
        "native_bad_after_local":int(np.count_nonzero(~new_pass)),
        "native_bad_gain_local":int(np.count_nonzero(~old_pass)-np.count_nonzero(~new_pass)),
        "minimum_normalized_det_before":float(np.min(old_q)),"minimum_normalized_det_after":float(np.min(new_q)),
        "exact_oriented_cavity_boundary":"PASS","exact_F32_volume_and_sign":"PASS",
        "replacement_self_intersection":self_result,"retained_volume_intersection":retained_result,"prior_replacement_count":len(prior),
        "all_replacement_native_quality_pass":bool(np.all(new_pass)),
        "all_parent_point_bits_preserved":True,"material_preserved":True,
        "physical_admission":False,"native_steps":0,
    }


def solve_bounded_fixed_vertex_milp(
    owner, points, tetrahedra, materials, old_rows, *,
    coverage_fn: Callable, solve_fn: Callable,
    seconds: float = 2.0, max_points: int = 32, max_candidate_tets: int = 60000,
    max_cavity_cells: int = 50000,
    exact_replacement_self_check=None, exact_retained_intersection_check=None,
    parent_offsets=None, parent_rows=None,
):
    """Bounded 230/220 adapter: face coverage first, then MILP, then full local certification."""
    _require_owner(owner)
    p=np.asarray(points); t=np.asarray(tetrahedra,np.int64); mat=np.asarray(materials)
    rows=np.asarray(old_rows,np.int64).reshape(-1)
    if p.dtype!=np.float32: raise ValueError("MILP cavity requires Float32 parent points")
    if max_points<4 or max_points>36 or max_candidate_tets<1 or max_candidate_tets>60000 or seconds<=0 or seconds>20 or max_cavity_cells<1 or max_cavity_cells>50000: raise ValueError("hard MILP limits exceeded")
    if t.ndim!=2 or t.shape[1]!=4 or mat.shape!=(len(t),): raise ValueError("malformed parent volume/material arrays")
    if not len(rows) or len(rows)>max_cavity_cells or len(np.unique(rows))!=len(rows) or np.any(rows<0) or np.any(rows>=len(t)): raise ValueError("bad or over-budget old rows")
    if len(np.unique(mat[rows]))!=1: return {"accepted":False,"status":"REJECT_MIXED_MATERIAL_CAVITY"}
    old=t[rows]; vertices=np.unique(old); n=len(vertices)
    if n>max_points: return {"accepted":False,"status":"REJECT_POINT_BUDGET","point_count":n}
    candidates=comb(n,4) if n>=4 else 0
    if candidates>max_candidate_tets: return {"accepted":False,"status":"REJECT_CANDIDATE_BUDGET","candidate_tetrahedra":candidates}
    boundary=oriented_boundary_faces(old); _check_one_closed_component(boundary)
    remap=np.full(len(p),-1,np.int64); remap[vertices]=np.arange(n,dtype=np.int64)
    local_points=p[vertices]; local_boundary=remap[boundary]
    covered,coverage_report=coverage_fn(local_points,local_boundary)
    if not covered: return {"accepted":False,"status":"REJECT_NATIVE_BOUNDARY_COVERAGE","coverage":coverage_report,"candidate_tetrahedra_upper_bound":candidates}
    solve_report,local_new=solve_fn(local_points,local_boundary,seconds=seconds,max_points=max_points,material=int(mat[rows[0]]))
    if local_new is None: return {"accepted":False,"status":"REJECT_MILP_NO_PROPOSAL","coverage":coverage_report,"solver":solve_report}
    global_new=vertices[np.asarray(local_new,np.int64)]
    result=evaluate_connected_cavity(
        owner,p,t,mat,rows,global_new,
        exact_replacement_self_check=exact_replacement_self_check,
        exact_retained_intersection_check=(lambda pp,tt,rr,nn,prior: exact_retained_intersection_check(pp,tt,rr,nn)) if callable(exact_retained_intersection_check) else None,
        parent_offsets=parent_offsets,parent_rows=parent_rows,
    )
    result["coverage"]=coverage_report;result["solver"]=solve_report
    return result


def _parent_csr(offsets,rows,n):
    if offsets is None and rows is None: return np.arange(n+1,dtype=np.int64),np.arange(n,dtype=np.int64)
    if offsets is None or rows is None: raise ValueError("parent offsets and rows supplied together")
    off=np.asarray(offsets,np.int64);rr=np.asarray(rows,np.int64)
    if off.shape!=(n+1,) or off[0]!=0 or off[-1]!=len(rr) or np.any(off[1:]<off[:-1]) or np.any(off[1:]==off[:-1]) or np.any(rr<0): raise ValueError("malformed parent-union CSR")
    return off,rr


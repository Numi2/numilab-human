#!/usr/bin/env python3
"""Adaptive same-material interior-Steiner kernel-cone cavities with exact immediate lineage.

Cavities retain their oriented boundary. A bounded LP finds strict visibility-kernel
interior and a three-coordinate quality-margin SLSQP proposes one appended F32 apex.
Every cone tet must pass the unchanged native owner and exact volume/chain gates.
Construction proposals only; source surfaces stay fixed; no mass mapping/admission.
"""
from pathlib import Path
from collections import Counter, defaultdict, deque
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import inspect
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
TOP_PATH = BASE / "remesh_complete_surface_cavities_218_root.py"
SOLVER_PATH = HERE / "solve_partial_oriented_cavity_milp_684.py"

def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

T = imp("partial_quality_top218", TOP_PATH)
S = imp("partial_quality_solver684", SOLVER_PATH)
M = S.M
KERNEL_PATH = BASE / "quality-kernel-cone-746-root" / "interior_kernel_cone_746.py"
EXPECTED_KERNEL_SHA256 = "d390a196d8b38e480e875414e9bc40e0fb683ecc0115db2b13660909b03520e5"
if hashlib.sha256(KERNEL_PATH.read_bytes()).hexdigest() != EXPECTED_KERNEL_SHA256:
    raise RuntimeError("pinned 734 kernel-cone helper SHA mismatch")
KERNEL = imp("quality_kernel_cone_746", KERNEL_PATH)
EXPECTED_QUALITY_OWNER_PATH = Path(T.m.m.__file__).resolve()
EXPECTED_QUALITY_OWNER_SHA256 = "650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0"
if hashlib.sha256(EXPECTED_QUALITY_OWNER_PATH.read_bytes()).hexdigest() != EXPECTED_QUALITY_OWNER_SHA256:
    raise RuntimeError("pinned M.certify owner SHA mismatch")
if Path(inspect.getsourcefile(M.certify)).resolve() != EXPECTED_QUALITY_OWNER_PATH or \
        Path(inspect.getsourcefile(KERNEL.M.certify)).resolve() != EXPECTED_QUALITY_OWNER_PATH:
    raise RuntimeError("kernel and 747 must use the same pinned M.certify owner")
PAT = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), np.int64)
QUALITY_GATE = 1.0e-5
VOLUME_GATE = 1.0e-18

# These are immediate-parent maps for this no-point-motion operation. The
# parent artifact remains pinned, so historical quotient lineage stays there;
# this step's canonical maps are identity and its point-parent CSR is empty.
CANONICAL_POINT_IDENTITY_KEYS = (
    "input_node_to_output_node",
    "output_node_to_input_node",
    "output_point_to_input_point_row",
    "source_I_vertex_quotient",
    "source_input_vertex_quotient",
    "source_input_node_to_output_node",
    "input_point_to_output_point_row",
    "quality_milp_input_node_to_output_node",
    "quality_milp_output_node_to_input_node",
)
CANONICAL_POINT_PARENT_OFFSETS_KEYS = ("output_point_parent_offsets", "point_parent_offsets")
CANONICAL_POINT_PARENT_ROWS_KEYS = (
    "output_point_parent_input_cell_rows", "output_point_parent_cell_rows", "point_parent_rows"
)
READBACK_MUTABLE_KEYS=frozenset({
    "points_world_m","points_world_m_f32","tetrahedra","material_indices","boundary_faces","desired_boundary_faces",
    "output_cell_to_input_cell","input_cell_to_output_cell","retained_input_cell_rows","replacement_tetrahedra",
    "replacement_parent_offsets","replacement_parent_cell_rows","replacement_tet_offsets",
    "quality_fixed_boundary_operation_parent_offsets","quality_fixed_boundary_operation_parent_cell_rows",
    "quality_fixed_boundary_operation_count","quality_fixed_boundary_operation_methods",
    "quality_fixed_boundary_operation_seed_rows","quality_fixed_boundary_input_candidate_path",
    "quality_fixed_boundary_input_candidate_sha256","quality_milp_operation_parent_offsets",
    "quality_milp_operation_parent_cell_rows","quality_milp_partial_bad_replacement",
    "quality_milp_operation_count","quality_milp_operation_seed_rows",
    "quality_milp_input_candidate_path","quality_milp_input_candidate_sha256",
    "quality_milp_input_node_to_output_node","quality_milp_output_node_to_input_node",
    "quality_steiner_point_output_rows","quality_steiner_operation_count",
    "quality_steiner_kernel_helper_path","quality_steiner_kernel_helper_sha256",
    "quality_steiner_cavity_selection_methods","current_geometry_parent_path",
    "current_geometry_parent_sha256","canonical_parent_union_operation_order",
    "input_node_to_output_node","output_node_to_input_node","output_point_to_input_point_row",
    "input_point_to_output_point_row","source_I_vertex_quotient","source_input_vertex_quotient",
    "source_input_node_to_output_node","quality_milp_input_node_to_output_node",
    "quality_milp_output_node_to_input_node","output_point_parent_offsets",
    "output_point_parent_input_cell_rows","point_parent_offsets","point_parent_rows",
    "output_point_parent_cell_rows","metadata_finalizer_point_map_status",
    "metadata_finalizer_point_parent_status","source_interior_vertex_quotient",
    "steiner_parent_point_csr_input_path","steiner_parent_point_csr_input_sha256",
    "steiner_parent_point_csr_input_offsets","steiner_parent_point_csr_input_rows",
    "steiner_parent_point_csr_input_output_point_parent_offsets",
    "steiner_parent_point_csr_input_output_point_parent_input_cell_rows",
    "steiner_parent_point_csr_input_point_parent_offsets",
    "steiner_parent_point_csr_input_point_parent_rows",
    "steiner_parent_point_csr_input_output_point_parent_cell_rows",
    "steiner_point_csr_policy",
})

CANONICAL_POINT_MUTABLE_KEYS = frozenset(
    CANONICAL_POINT_IDENTITY_KEYS + CANONICAL_POINT_PARENT_OFFSETS_KEYS +
    CANONICAL_POINT_PARENT_ROWS_KEYS
)

def unchanged_point_lineage_updates(parent_arrays, npoints):
    """Return immediate-parent point maps for an unchanged point namespace."""
    identity = np.arange(int(npoints), dtype=np.int64)
    updates = {}
    for key in CANONICAL_POINT_IDENTITY_KEYS:
        if key in parent_arrays or key in {
            "input_node_to_output_node", "output_node_to_input_node",
            "output_point_to_input_point_row", "source_I_vertex_quotient",
        }:
            updates[key] = identity.copy()
    for key in CANONICAL_POINT_PARENT_OFFSETS_KEYS:
        if key in parent_arrays or key == "output_point_parent_offsets":
            updates[key] = np.zeros(int(npoints) + 1, dtype=np.int64)
    for key in CANONICAL_POINT_PARENT_ROWS_KEYS:
        if key in parent_arrays or key == "output_point_parent_input_cell_rows":
            updates[key] = np.empty(0, dtype=np.int64)
    return updates

def assert_unchanged_point_lineage(arrays, npoints):
    identity = np.arange(int(npoints), dtype=np.int64)
    required = {
        "input_node_to_output_node", "output_node_to_input_node",
        "output_point_to_input_point_row", "source_I_vertex_quotient",
        "output_point_parent_offsets", "output_point_parent_input_cell_rows",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError("canonical point lineage fields missing: " + ",".join(sorted(missing)))
    for key in CANONICAL_POINT_IDENTITY_KEYS:
        if key in arrays and not np.array_equal(np.asarray(arrays[key]), identity):
            raise ValueError(f"unchanged point map is not identity: {key}")
    for key in CANONICAL_POINT_PARENT_OFFSETS_KEYS:
        if key in arrays and not np.array_equal(np.asarray(arrays[key]), np.zeros(int(npoints)+1, np.int64)):
            raise ValueError(f"unchanged-point parent CSR is not empty: {key}")
    for key in CANONICAL_POINT_PARENT_ROWS_KEYS:
        if key in arrays and np.asarray(arrays[key]).size != 0:
            raise ValueError(f"unchanged-point parent rows are not empty: {key}")

def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()

def pin(path):
    return {"path": str(path), "sha256": sha(path)}

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def boundary_target(arrays):
    for key in ("desired_boundary_faces", "proposed_current_desired_boundary_faces",
                "proposed_current_desired_boundary_faces_rows", "boundary_faces"):
        if key in arrays:
            return np.asarray(arrays[key], np.int64), key
    raise KeyError("candidate has no recognized desired/current boundary faces")

def quality_score(points, tets):
    det, q = M.measurements(points, tets)
    ok = np.asarray(M.passes(det, q), bool)
    ratio = np.minimum(np.minimum(q / QUALITY_GATE, det / (6.0 * VOLUME_GATE)), 1.0)
    deficit = float(-np.log(np.maximum(ratio, 1.0e-300)).sum())
    return int((~ok).sum()), deficit, det, q

def exact_positive(points, tets, exact_point_cache=None):
    """Check exact positivity from only the referenced F32 nodes, caching lattice rows."""
    p=np.asarray(points,np.float32);cells=np.asarray(tets,np.int64)
    if cells.ndim!=2 or cells.shape[1]!=4 or not len(cells):return False
    cache={} if exact_point_cache is None else exact_point_cache
    used=np.unique(cells.reshape(-1))
    missing=[int(v) for v in used if int(v) not in cache]
    if missing:
        records=M.integer_points(p[np.asarray(missing,np.int64)])
        cache.update((v,row) for v,row in zip(missing,records))
    lookup={int(v):i for i,v in enumerate(used)}
    coords=[cache[int(v)] for v in used]
    values=[M.exact_det(coords,[lookup[int(v)] for v in tet]) for tet in cells]
    return min(values)>0

POINT_KEY_DTYPE=np.dtype([("x",np.uint32),("y",np.uint32),("z",np.uint32)])

def _point_key(point):
    bits=np.asarray(point,np.float32).reshape(3).view(np.uint32).copy()
    bits[bits==0x80000000]=0
    return tuple(map(int,bits))

def _point_key_index(points):
    bits=np.ascontiguousarray(points,dtype=np.float32).view(np.uint32).reshape((-1,3)).copy()
    bits[bits==0x80000000]=0
    keys=np.empty(len(bits),dtype=POINT_KEY_DTYPE)
    keys["x"]=bits[:,0];keys["y"]=bits[:,1];keys["z"]=bits[:,2]
    return np.sort(keys,kind="quicksort")

def _point_key_present(index,key):
    row=np.asarray([key],dtype=POINT_KEY_DTYPE)[0]
    at=int(np.searchsorted(index,row))
    return at<len(index) and bool(index[at]==row)

class CompactFaceAdjacency:
    def __init__(self, rows): self.rows=rows
    def get(self, row, default=()):
        values=self.rows[int(row)]
        return values[values>=0]

def face_bad_adjacency(tets, materials, bad):
    # Include passing neighbors: fixed-boundary repair often needs an adjacent
    # good tetrahedron to remove an isolated sliver. Compact four-face CSR.
    ncell=len(tets)
    faces=np.sort(np.asarray(tets,np.int64)[:,PAT].reshape(-1,3),axis=1)
    order=np.lexsort((faces[:,2],faces[:,1],faces[:,0]));f=faces[order]
    starts=np.r_[0,1+np.flatnonzero(np.any(f[1:]!=f[:-1],axis=1))]
    counts=np.diff(np.r_[starts,len(f)]);starts=starts[counts==2]
    one=order[starts];two=order[starts+1];a=one//4;b=two//4
    same=np.asarray(materials)[a]==np.asarray(materials)[b]
    one=one[same];two=two[same];a=a[same];b=b[same]
    rows=np.full((ncell,4),-1,np.int32)
    rows[a,one%4]=b;rows[b,two%4]=a
    return CompactFaceAdjacency(rows)

def grow_bad_cluster(seed, adjacency, bad, materials, vertices_by_cell,
                     max_cells, max_vertices, reserved_cells, reserved_vertices,
                     deficit_by_cell):
    if seed in reserved_cells or materials[seed] is None:
        return None
    mat = materials[seed]
    chosen = [int(seed)]
    usedv = set(map(int, vertices_by_cell[seed]))
    if usedv & reserved_vertices or len(usedv) > max_vertices:
        return None
    queued = set()
    frontier = [int(seed)]
    badset = set(map(int, bad))
    while frontier and len(chosen) < max_cells:
        current = frontier.pop(0)
        for nb in adjacency.get(current, ()):
            if nb in queued or nb in chosen:
                continue
            queued.add(nb)
            if nb in reserved_cells or materials[nb] != mat:
                continue
            union = usedv | set(map(int, vertices_by_cell[nb]))
            if len(union) > max_vertices or union & reserved_vertices:
                continue
            chosen.append(nb)
            usedv = union
            frontier.append(nb)
            if len(chosen) >= max_cells:
                break
        frontier.sort(key=lambda r: (-float(deficit_by_cell[r]), r))
    return np.asarray(sorted(chosen), np.int64)

def complete_same_material_vertex_star(seed,vertex,materials,vertices_by_cell,
                                       vertex_incidence,max_cells,max_vertices,
                                       reserved_cells,reserved_vertices):
    incident=vertex_incidence.get(int(vertex))
    if incident is None or not len(incident):return None
    mat=materials[int(seed)]
    rows=np.asarray(incident[materials[incident]==mat],np.int64)
    if not len(rows) or int(seed) not in set(map(int,rows)):return None
    rows=np.unique(rows)
    if len(rows)>int(max_cells) or set(map(int,rows))&reserved_cells:return None
    verts=np.unique(vertices_by_cell[rows])
    if len(verts)>int(max_vertices) or set(map(int,verts))&reserved_vertices:return None
    return np.sort(rows)

def adaptive_cavity_candidates(seed,adjacency,bad,materials,vertices_by_cell,
                               vertex_incidence,face_cell_cap,star_cell_cap,
                               max_vertices,reserved_cells,reserved_vertices,
                               deficit_by_cell):
    candidates=[];seen=set();star_rows=[]
    for vertex in np.unique(vertices_by_cell[int(seed)]):
        rows=complete_same_material_vertex_star(seed,int(vertex),materials,vertices_by_cell,
            vertex_incidence,star_cell_cap,max_vertices,reserved_cells,reserved_vertices)
        if rows is None:continue
        key=tuple(map(int,rows))
        if key in seen:continue
        seen.add(key);star_rows.append((f"complete_vertex_star_node_{int(vertex)}",rows))
    # Prefer a bounded full star with fewer cells; never truncate a vertex star.
    star_rows.sort(key=lambda item:(len(item[1]),item[0]))
    candidates.extend(star_rows)
    for size in (2,4,8,12,24):
        if size>int(face_cell_cap):continue
        rows=grow_bad_cluster(seed,adjacency,bad,materials,vertices_by_cell,
            size,max_vertices,reserved_cells,reserved_vertices,deficit_by_cell)
        if rows is None or not len(rows):continue
        key=tuple(map(int,rows))
        if key in seen:continue
        seen.add(key);candidates.append((f"face_ball_{size}",rows))
    return candidates

HISTORICAL_POINT_CSR_KEYS=(
    "output_point_parent_offsets","output_point_parent_input_cell_rows",
    "point_parent_offsets","point_parent_rows","output_point_parent_cell_rows",
)

def _point_parent_csr(parent,ops,nold):
    # Immediate parent for the old point prefix is identity, hence empty CSR rows.
    offsets=[0]*(int(nold)+1);rows=[]
    for op in ops:
        parent_rows=sorted(set(map(int,op["old_cell_rows"])))
        rows.extend(parent_rows);offsets.append(len(rows))
    return np.asarray(offsets,np.int64),np.asarray(rows,np.int64)

def _historical_point_csr_updates(parent,nold,source_path,source_sha):
    offsets=np.asarray(parent.get("output_point_parent_offsets",np.zeros(int(nold)+1,np.int64)),np.int64)
    rows=np.asarray(parent.get("output_point_parent_input_cell_rows",np.empty(0,np.int64)),np.int64)
    if offsets.shape!=(int(nold)+1,) or offsets[0]!=0 or offsets[-1]!=len(rows) or np.any(np.diff(offsets)<0):
        raise ValueError("input point-parent CSR is malformed")
    for key in ("point_parent_offsets",):
        if key in parent and not np.array_equal(np.asarray(parent[key],np.int64),offsets):
            raise ValueError("input point-parent offset alias differs from canonical CSR: "+key)
    for key in ("point_parent_rows","output_point_parent_cell_rows"):
        if key in parent and not np.array_equal(np.asarray(parent[key],np.int64),rows):
            raise ValueError("input point-parent row alias differs from canonical CSR: "+key)
    out={
        "steiner_parent_point_csr_input_path":np.asarray([str(source_path)]),
        "steiner_parent_point_csr_input_sha256":np.asarray([str(source_sha)]),
        "steiner_parent_point_csr_input_offsets":offsets.copy(),
        "steiner_parent_point_csr_input_rows":rows.copy(),
        "steiner_parent_point_csr_input_output_point_parent_offsets":offsets.copy(),
        "steiner_parent_point_csr_input_output_point_parent_input_cell_rows":rows.copy(),
        "steiner_point_csr_policy":np.asarray("RESET_IMMEDIATE_OLD_POINT_PREFIX_AND_ARCHIVE_PRIOR_ABSOLUTE_PARENT"),
    }
    for key in HISTORICAL_POINT_CSR_KEYS[2:]:
        if key in parent:
            out["steiner_parent_point_csr_input_"+key]=np.asarray(parent[key]).copy()
    return out

def _point_map_updates(parent,nold,nnew,point_offsets,point_rows):
    in_to_out=np.arange(nold,dtype=np.int64)
    out_to_in=np.r_[in_to_out,np.full(nnew,-1,np.int64)]
    updates={
      "input_node_to_output_node":in_to_out.copy(),
      "output_node_to_input_node":out_to_in.copy(),
      "output_point_to_input_point_row":out_to_in.copy(),
      "input_point_to_output_point_row":in_to_out.copy(),
      "source_I_vertex_quotient":out_to_in.copy(),
      "source_input_vertex_quotient":out_to_in.copy(),
      "source_input_node_to_output_node":in_to_out.copy(),
      "quality_milp_input_node_to_output_node":in_to_out.copy(),
      "quality_milp_output_node_to_input_node":out_to_in.copy(),
      "output_point_parent_offsets":point_offsets,
      "output_point_parent_input_cell_rows":point_rows,
    }
    if "point_parent_offsets" in parent:updates["point_parent_offsets"]=point_offsets.copy()
    if "point_parent_rows" in parent:updates["point_parent_rows"]=point_rows.copy()
    if "output_point_parent_cell_rows" in parent:updates["output_point_parent_cell_rows"]=point_rows.copy()
    if "source_interior_vertex_quotient" in parent:
        updates["source_interior_vertex_quotient"]=np.r_[
            np.asarray(parent["source_interior_vertex_quotient"],np.int64),
            np.full(nnew,-1,np.int64)]
    return updates

def independent_readback_replay(parent,readback,ops,retained,target,bad_before,expected_bad_after,source_path,source_sha):
    """Readback-only exact replay; does not call the 746 construction solver."""
    # Every source, typed-surface, back/rim, anchor, and unrelated parent field
    # must survive readback unchanged unless named in the explicit mutation set.
    for key,value in parent.items():
        if key in READBACK_MUTABLE_KEYS:continue
        if key not in readback or not np.array_equal(np.asarray(readback[key]),np.asarray(value)):
            raise ValueError("readback changed or dropped immutable parent binding: "+str(key))
    oldp=np.asarray(parent["points_world_m"],np.float32);newp=np.asarray(readback["points_world_m"],np.float32)
    oldt=np.asarray(parent["tetrahedra"],np.int64);oldtmat=np.asarray(parent["material_indices"])
    outt=np.asarray(readback["tetrahedra"],np.int64);outmat=np.asarray(readback["material_indices"])
    nold=len(oldp);nnew=len(ops);retained=np.asarray(retained,np.int64)
    if not np.array_equal(newp[:nold].view(np.uint32),oldp.view(np.uint32)):raise ValueError("readback changed original F32 point prefix")
    if len(newp)!=nold+nnew:raise ValueError("readback Steiner point count mismatch")
    if "points_world_m_f32" in parent and "points_world_m_f32" not in readback:
        raise ValueError("readback dropped the parent F32 point binding")
    if "points_world_m_f32" in readback and not np.array_equal(np.asarray(readback["points_world_m_f32"]).view(np.uint32),newp.view(np.uint32)):
        raise ValueError("readback F32 point copies disagree")
    new_blocks=[np.asarray(op["new_tetrahedra"],np.int64) for op in ops]
    replacement=np.vstack(new_blocks)
    expected_t=np.vstack((oldt[retained],replacement))
    expected_mat=np.r_[oldtmat[retained],np.concatenate([np.full(len(block),int(op["material_index"]),oldtmat.dtype) for op,block in zip(ops,new_blocks)])]
    if not np.array_equal(outt,expected_t) or not np.array_equal(outmat,expected_mat):
        raise ValueError("readback canonical cell arrays differ from parent-union journal")
    if len(np.unique(np.sort(outt,axis=1),axis=0))!=len(outt):raise ValueError("readback duplicate tetrahedra")
    det,q=M.measurements(newp,outt);good=np.asarray(M.passes(det,q),bool)
    if np.any(det<=0) or int(np.count_nonzero(~good))!=int(expected_bad_after):raise ValueError("readback native quality/positive-cell replay failed")
    olddet,oldq=M.measurements(oldp,oldt);oldgood=np.asarray(M.passes(olddet,oldq),bool)
    output_to_input=np.asarray(readback["output_cell_to_input_cell"],np.int64)
    if output_to_input.shape!=(len(outt),):raise ValueError("output-to-input cell map shape mismatch")
    for outrow,inrow in enumerate(output_to_input):
        if inrow>=0 and oldgood[int(inrow)] and not good[outrow]:raise ValueError("readback regressed an old good cell")
    if not int(np.count_nonzero(~good))<int(bad_before):raise ValueError("readback did not strictly reduce native-bad count")
    actual=T.m.boundary(outt)
    if "boundary_faces" not in readback or not np.array_equal(
            T.m.boundary_identity(np.asarray(readback["boundary_faces"],np.int64)),
            T.m.boundary_identity(actual)):
        raise ValueError("stored readback boundary_faces differs from reconstructed volume boundary")
    if "desired_boundary_faces" not in readback or not np.array_equal(
            T.m.boundary_identity(np.asarray(readback["desired_boundary_faces"],np.int64)),
            T.m.boundary_identity(np.asarray(target,np.int64))):
        raise ValueError("stored desired boundary differs from parent target")
    if not np.array_equal(T.m.boundary_identity(actual),T.m.boundary_identity(np.asarray(target,np.int64))):
        raise ValueError("readback changed oriented target boundary")
    nreplacement=len(replacement);po=np.asarray(readback["replacement_parent_offsets"],np.int64);pr=np.asarray(readback["replacement_parent_cell_rows"],np.int64)
    if po.shape!=(nreplacement+1,) or po[0]!=0 or po[-1]!=len(pr) or np.any(np.diff(po)<0):
        raise ValueError("replacement parent CSR shape mismatch")
    if "replacement_tetrahedra" not in readback or not np.array_equal(
            np.asarray(readback["replacement_tetrahedra"],np.int64),replacement):
        raise ValueError("stored replacement_tetrahedra differs from output replacement suffix")
    expected_retained=np.asarray(retained,np.int64)
    if "retained_input_cell_rows" not in readback or not np.array_equal(
            np.asarray(readback["retained_input_cell_rows"],np.int64),expected_retained):
        raise ValueError("retained input cell rows are not canonical")
    expected_tet_offsets=np.r_[0,np.cumsum([len(block) for block in new_blocks])].astype(np.int64)
    if not np.array_equal(np.asarray(readback["replacement_tet_offsets"],np.int64),expected_tet_offsets):
        raise ValueError("replacement_tet_offsets do not match operation blocks")
    expected_parent_offsets=[0];expected_parent_rows=[];expected_op_offsets=[0];expected_op_rows=[]
    for op in ops:
        oldrows=sorted(set(map(int,op["old_cell_rows"])))
        for _ in op["new_tetrahedra"]:
            expected_parent_rows.extend(oldrows);expected_parent_offsets.append(len(expected_parent_rows))
        expected_op_rows.extend(oldrows);expected_op_offsets.append(len(expected_op_rows))
    if not np.array_equal(po,np.asarray(expected_parent_offsets,np.int64)) or not np.array_equal(pr,np.asarray(expected_parent_rows,np.int64)):
        raise ValueError("replacement tet parent CSR does not reconstruct complete cavity unions")
    for key,expected in (
        ("quality_fixed_boundary_operation_parent_offsets",np.asarray(expected_op_offsets,np.int64)),
        ("quality_fixed_boundary_operation_parent_cell_rows",np.asarray(expected_op_rows,np.int64)),
        ("quality_fixed_boundary_operation_count",np.asarray([len(ops)],np.int64)),
        ("quality_fixed_boundary_operation_methods",np.asarray([op["proposal_method"] for op in ops])),
        ("quality_fixed_boundary_operation_seed_rows",np.asarray([op["seed_cell_row"] for op in ops],np.int64)),
        ("quality_milp_operation_parent_offsets",np.asarray([0],np.int64)),
        ("quality_milp_operation_parent_cell_rows",np.empty(0,np.int64)),
        ("quality_milp_partial_bad_replacement",np.asarray([False])),
        ("quality_milp_operation_count",np.asarray([0],np.int64)),
        ("quality_milp_operation_seed_rows",np.empty(0,np.int64)),
        ("quality_steiner_operation_count",np.asarray([len(ops)],np.int64)),
        ("quality_steiner_point_output_rows",np.asarray([op["apex_point_output_row"] for op in ops],np.int64)),
        ("quality_steiner_cavity_selection_methods",np.asarray([op["candidate_selection_method"] for op in ops])),
    ):
        if key not in readback or not np.array_equal(np.asarray(readback[key]),expected):
            raise ValueError("readback operation/parent binding mismatch: "+key)
    point_off=np.asarray(readback["output_point_parent_offsets"],np.int64);point_rows=np.asarray(readback["output_point_parent_input_cell_rows"],np.int64)
    if point_off.shape!=(len(newp)+1,) or point_off[0]!=0 or point_off[-1]!=len(point_rows) or np.any(np.diff(point_off)<0):
        raise ValueError("Steiner point parent CSR shape mismatch")
    expected_point_offsets=[0]*(nold+1);expected_point_rows=[]
    for op in ops:
        oldrows=sorted(set(map(int,op["old_cell_rows"])))
        expected_point_rows.extend(oldrows);expected_point_offsets.append(len(expected_point_rows))
    if not np.array_equal(point_off,np.asarray(expected_point_offsets,np.int64)) or not np.array_equal(point_rows,np.asarray(expected_point_rows,np.int64)):
        raise ValueError("immediate point-parent CSR is not empty for identity old-point rows plus current cavity parents")
    for key,expected in (
        ("point_parent_offsets",np.asarray(expected_point_offsets,np.int64)),
        ("point_parent_rows",np.asarray(expected_point_rows,np.int64)),
        ("output_point_parent_cell_rows",np.asarray(expected_point_rows,np.int64)),
    ):
        if key in parent and (key not in readback or not np.array_equal(np.asarray(readback[key]),expected)):
            raise ValueError("point-parent CSR alias differs from immediate parent CSR: "+key)
    primary_history={
        "output_point_parent_offsets":"steiner_parent_point_csr_input_offsets",
        "output_point_parent_input_cell_rows":"steiner_parent_point_csr_input_rows",
    }
    for parent_key in HISTORICAL_POINT_CSR_KEYS:
        if parent_key not in parent:continue
        archive_key=primary_history.get(parent_key,"steiner_parent_point_csr_input_"+parent_key)
        if archive_key not in readback or not np.array_equal(np.asarray(readback[archive_key]),np.asarray(parent[parent_key])):
            raise ValueError("historical point-parent CSR was not archived exactly: "+parent_key)
    if str(np.asarray(readback.get("steiner_parent_point_csr_input_path",[""])).reshape(-1)[0])!=str(source_path):
        raise ValueError("historical point-parent CSR input path binding mismatch")
    if str(np.asarray(readback.get("steiner_parent_point_csr_input_sha256",[""])).reshape(-1)[0])!=str(source_sha):
        raise ValueError("historical point-parent CSR input SHA binding mismatch")
    if str(np.asarray(readback.get("steiner_point_csr_policy",[""])).reshape(-1)[0])!="RESET_IMMEDIATE_OLD_POINT_PREFIX_AND_ARCHIVE_PRIOR_ABSOLUTE_PARENT":
        raise ValueError("immediate-versus-historical point CSR policy binding mismatch")
    replacement_cursor=0;op_receipts=[]
    for oi,op in enumerate(ops):
        oldrows=np.asarray(sorted(set(map(int,op["old_cell_rows"]))),np.int64)
        start=int(op["replacement_tet_range"][0]);end=int(op["replacement_tet_range"][1])
        if start!=replacement_cursor or end-start!=len(op["new_tetrahedra"]):raise ValueError("operation replacement row range mismatch")
        old=oldt[np.asarray(op["old_cell_rows"],np.int64)];new=outt[len(retained)+start:len(retained)+end]
        if not np.array_equal(new,np.asarray(op["new_tetrahedra"],np.int64)):raise ValueError("operation cone connectivity differs on readback")
        if M.chain(old)!=M.chain(new):raise ValueError("operation exact oriented chain mismatch")
        ok,why=M.certify(newp,old,new)
        if not ok:raise ValueError("operation exact determinant/volume certificate failed: "+str(why))
        nd,nq=M.measurements(newp,new)
        if not np.all(M.passes(nd,nq)):raise ValueError("operation new cone cell failed native quality gate")
        if int(np.count_nonzero(~M.passes(*M.measurements(oldp,old))))<=0:raise ValueError("operation did not remove an old bad cell")
        matrows=outmat[len(retained)+start:len(retained)+end]
        if np.any(matrows!=int(op["material_index"])):raise ValueError("operation material changed")
        apexrow=int(op["apex_point_output_row"])
        if apexrow!=nold+oi or not np.array_equal(newp[apexrow].view(np.uint32),np.asarray(op["apex_point_f32"],np.float32).view(np.uint32)):
            raise ValueError("operation Steiner point row/coordinate mismatch")
        if not np.array_equal(point_rows[point_off[apexrow]:point_off[apexrow+1]],oldrows):
            raise ValueError("Steiner point parent CSR lacks full original cavity union")
        for rr in range(start,end):
            if not np.array_equal(pr[po[rr]:po[rr+1]],oldrows):raise ValueError("replacement tet parent CSR lacks full original cavity union")
        op_receipts.append({"operation":oi,"old_cell_count":len(old),"new_cell_count":len(new),
                            "old_bad_count":int(np.count_nonzero(~M.passes(*M.measurements(oldp,old)))),
                            "new_bad_count":int(np.count_nonzero(~M.passes(nd,nq))),
                            "exact_chain":True,"exact_signed_volume_equal":True,"native_quality_all_pass":True,
                            "full_point_and_tet_parent_unions":True})
        replacement_cursor=end
    expected_map=np.r_[retained,np.full(nreplacement,-1,np.int64)]
    if not np.array_equal(output_to_input,expected_map):raise ValueError("output cell map is not canonical immediate-parent map")
    inv=np.full(len(oldt),-1,np.int64);inv[retained]=np.arange(len(retained),dtype=np.int64)
    if not np.array_equal(np.asarray(readback["input_cell_to_output_cell"],np.int64),inv):raise ValueError("input cell inverse map mismatch")
    ident=np.arange(nold,dtype=np.int64);outmap=np.r_[ident,np.full(nnew,-1,np.int64)]
    for key,expected in (("input_node_to_output_node",ident),("output_node_to_input_node",outmap),
                         ("output_point_to_input_point_row",outmap),("input_point_to_output_point_row",ident),
                         ("source_I_vertex_quotient",outmap),("source_input_vertex_quotient",outmap),
                         ("source_input_node_to_output_node",ident),
                         ("quality_milp_input_node_to_output_node",ident),
                         ("quality_milp_output_node_to_input_node",outmap)):
        if key not in readback or not np.array_equal(np.asarray(readback[key],np.int64),expected):
            raise ValueError("readback immediate point map mismatch: "+key)
    if "source_interior_vertex_quotient" in parent:
        expected=np.r_[np.asarray(parent["source_interior_vertex_quotient"],np.int64),np.full(nnew,-1,np.int64)]
        if "source_interior_vertex_quotient" not in readback or not np.array_equal(np.asarray(readback["source_interior_vertex_quotient"],np.int64),expected):
            raise ValueError("source interior quotient prefix or Steiner sentinel changed")
    for path_key,sha_key in (
        ("quality_fixed_boundary_input_candidate_path","quality_fixed_boundary_input_candidate_sha256"),
        ("quality_milp_input_candidate_path","quality_milp_input_candidate_sha256"),
        ("current_geometry_parent_path","current_geometry_parent_sha256"),
    ):
        if path_key not in readback or sha_key not in readback:
            raise ValueError("readback candidate provenance binding missing: "+path_key)
        if str(np.asarray(readback[path_key]).reshape(-1)[0])!=str(source_path) or str(np.asarray(readback[sha_key]).reshape(-1)[0])!=str(source_sha):
            raise ValueError("readback candidate path/SHA binding mismatch: "+path_key)
    if str(np.asarray(readback.get("canonical_parent_union_operation_order",[""])).reshape(-1)[0])!="interior-steiner-kernel-cone-full-original-cavity-union-v1":
        raise ValueError("canonical parent-union operation order alias mismatch")
    if (str(np.asarray(readback.get("quality_steiner_kernel_helper_path",[""])).reshape(-1)[0])!=str(KERNEL_PATH)
            or str(np.asarray(readback.get("quality_steiner_kernel_helper_sha256",[""])).reshape(-1)[0])!=EXPECTED_KERNEL_SHA256):
        raise ValueError("readback Steiner helper pin mismatch")
    return {"status":"PASS_INDEPENDENT_STEINER_CAVITY_READBACK",
            "point_prefix_unchanged":True,"new_steiner_points":nnew,
            "replacement_tets":nreplacement,"native_bad_before":int(bad_before),
            "native_bad_after":int(expected_bad_after),"strict_bad_drop":True,
            "all_output_cells_positive":True,"old_good_cells_preserved":True,
            "oriented_target_boundary_unchanged":True,"exact_cavity_chain_and_volume":True,
            "complete_point_parent_csr":True,"complete_replacement_tet_parent_csr":True,
            "canonical_immediate_point_and_cell_maps":True,"operations":op_receipts,
            "physical_admission":False,"native_steps":0}

def run_role(role, entry, state, outdir, deadline, args):
    role_started = time.monotonic()
    source_path = Path(entry["path"])
    source_sha = sha(source_path)
    if source_sha != entry["sha256"]:
        raise ValueError(f"{role} input SHA mismatch")
    a = load_npz(source_path)
    p = np.asarray(a["points_world_m"])
    t = np.asarray(a["tetrahedra"], np.int64)
    mat = np.asarray(a["material_indices"])
    if p.dtype != np.float32 or p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError(f"{role} requires finite F32 volume points")
    if t.ndim != 2 or t.shape[1] != 4 or mat.shape != (len(t),):
        raise ValueError(f"{role} malformed tetrahedron/material arrays")
    target, target_key = boundary_target(a)
    actual_boundary = T.m.boundary(t)
    if not np.array_equal(T.m.boundary_identity(actual_boundary),
                          T.m.boundary_identity(target)):
        raise ValueError(f"{role} input boundary does not match {target_key}")
    old_bad, old_deficit, old_det, old_q = quality_score(p, t)
    declared_bad = entry.get("native_quality_failures")
    if declared_bad is not None and int(declared_bad) != old_bad:
        raise ValueError(f"{role} pinned native-bad count differs from measured input")
    if np.any(old_det <= 0):
        raise ValueError(f"{role} input contains nonpositive tetrahedra")
    declared_nonpositive = entry.get("nonpositive", a.get("nonpositive", None))
    if declared_nonpositive is not None and int(declared_nonpositive) != 0:
        raise ValueError(f"{role} state declares nonpositive source cells")
    # A construction proposal inherits the source's exact-positive claim.
    # Every replaced old and every generated new cell is checked exactly below.
    if declared_nonpositive is None:
        raise ValueError(f"{role} state must pin nonpositive=0 source evidence")
    bad = np.flatnonzero(~M.passes(old_det, old_q))
    if not len(bad):
        return {"role": role, "status": "NO_NATIVE_BAD_CELLS",
                "input": pin(source_path), "bad_before": 0, "bad_after": 0}
    deficits = -np.log(np.maximum(np.minimum(np.minimum(old_q / QUALITY_GATE,
                                    old_det / (6.0 * VOLUME_GATE)), 1.0), 1.0e-300))
    preparation_started=time.monotonic()
    # Build same-material face adjacency once. Build node-to-cell incidence once
    # for the actual bad-corner roster only, and retain only those compact rows.
    adj = face_bad_adjacency(t, mat, bad)
    bad_corner_vertices=np.unique(t[bad].reshape(-1))
    flat_nodes=t.reshape(-1)
    node_counts=np.bincount(flat_nodes,minlength=len(p))
    node_offsets=np.r_[0,np.cumsum(node_counts)]
    corner_order=np.argsort(flat_nodes,kind="stable")
    corner_cell_rows=corner_order//4
    bad_corner_incidence={
        int(v):corner_cell_rows[node_offsets[int(v)]:node_offsets[int(v)+1]].astype(np.int32,copy=True)
        for v in bad_corner_vertices
    }
    bad_corner_incident_cell_rows=int(sum(len(v) for v in bad_corner_incidence.values()))
    del node_counts,node_offsets,corner_order,corner_cell_rows
    point_key_index=_point_key_index(p)
    accepted_point_keys=set()
    exact_point_cache={}
    index_preparation_s=time.monotonic()-preparation_started
    preparation_s=time.monotonic()-role_started
    proposal_started=time.monotonic()
    vertices_by_cell = t
    seeds = sorted(map(int, bad), key=lambda r: (-float(deficits[r]), r))
    reserved_cells = set()
    reserved_vertices = set()
    attempted = set()
    ops = []
    rejection = Counter()
    trial_rows = []
    # Cache cone proposals by this immutable role input's exact old-cell cluster.
    kernel_cone_cache = {}
    kernel_cone_stats = Counter()
    boundary_before = T.m.boundary_identity(actual_boundary)
    source_float32_bad = old_bad
    candidate_selection_counts=Counter()
    for seed in seeds:
        if seed in reserved_cells:continue
        candidates=adaptive_cavity_candidates(seed,adj,bad,mat,vertices_by_cell,
            bad_corner_incidence,args.maximum_old_cells,args.maximum_star_cells,
            args.maximum_points,reserved_cells,reserved_vertices,deficits)
        for selection_label,rows in candidates:
            now=time.monotonic()
            if (now>=deadline or len(ops)>=args.maximum_operations
                    or len(attempted)>=args.maximum_attempts):
                break
            key=tuple(sorted(map(int,rows)))
            if key in attempted:
                if key in kernel_cone_cache:kernel_cone_stats["cache_hits"]+=1
                continue
            if set(key)&reserved_cells:
                rejection["candidate_conflicts_with_accepted_cells"]+=1
                continue
            candidate_vertices=set(map(int,np.unique(t[rows])))
            if candidate_vertices&reserved_vertices:
                rejection["candidate_conflicts_with_accepted_vertices"]+=1
                continue
            attempted.add(key);candidate_selection_counts[selection_label]+=1
            old = t[rows]
            verts = np.unique(old)
            if len(verts) > args.maximum_points:
                rejection["vertex_limit"] += 1
                continue
            if len(np.unique(mat[rows])) != 1:
                rejection["mixed_material"] += 1
                continue
            local_boundary = T.m.boundary(old)
            closed, open_edges = T.closed(local_boundary)
            topology = T.manifold_topology(local_boundary)
            if (not closed or topology["bad_vertex_links"] or
                    topology["boundary_components"] != 1):
                rejection["nonmanifold_cavity"] += 1
                continue
            if not exact_positive(p, old, exact_point_cache):
                rejection["old_cavity_not_exact_positive"] += 1
                continue
            lp = p[verts]
            lb = np.searchsorted(verts,local_boundary)
            old_local=np.searchsorted(verts,old)
            cone = kernel_cone_cache.get(key)
            if cone is None:
                try:
                    cone = KERNEL.propose_interior_kernel_cone(
                        lp, lb, old_local, mat[rows],
                        time_limit_s=max(0.02, deadline-time.monotonic()),
                        construction_margin=args.construction_margin)
                except Exception as exc:
                    cone = {"status": "INTERIOR_STEINER_EXCEPTION", "error": repr(exc)}
                kernel_cone_cache[key] = cone
                kernel_cone_stats["evaluations"] += 1
            else:
                kernel_cone_stats["cache_hits"] += 1
            cone_summary = {k: cone[k] for k in (
                "status", "candidate_source", "old_bad", "new_bad", "old_log_deficit",
                "new_log_deficit", "visibility_height_normalized",
                "continuous_quality_margin_normalized", "slsqp_success", "slsqp_status",
                "slsqp_iterations", "construction_margin", "normalized_cavity_diameter_m",
                "minimum_native_quality", "minimum_det6", "exact_positive_new_cell_count",
                "exact_det6_delta_integer", "oriented_chain_matches_old",
                "exact_volume_matches_old", "quality_gate_verified_after_F32_rounding",
                "volume_gate_verified_after_F32_rounding", "reason", "error",
                "candidate_rejections") if k in cone}
            if cone.get("status") != "PASS_INTERIOR_STEINER_KERNEL_CONE":
                rejection["interior_kernel_" + str(cone.get("status", "NO_RESULT"))] += 1
                kernel_cone_stats["rejected"] += 1
                trial_rows.append({
                    "seed_cell_row": int(seed), "old_cell_rows": rows.tolist(),
                    "old_vertex_rows": verts.tolist(), "old_cells": int(len(rows)),
                    "old_bad": int(np.count_nonzero(~M.passes(old_det[rows], old_q[rows]))),
                    "old_deficit": float(deficits[rows].sum()), "open_edges": int(open_edges),
                    "proposal_method": "interior_steiner_kernel_cone",
                    "candidate_selection_method": selection_label,
                    "kernel_cone": cone_summary})
                continue
            apex = np.asarray(cone["apex_point_f32"], np.float32)
            point_row = len(p) + len(ops)
            apex_key=_point_key(apex)
            if (not np.isfinite(apex).all() or _point_key_present(point_key_index,apex_key)
                    or apex_key in accepted_point_keys):
                rejection["duplicate_or_nonfinite_steiner_point"] += 1
                kernel_cone_stats["rejected"] += 1
                continue
            local_new = np.asarray(cone["replacement_tetrahedra_local"],np.int64)
            if local_new.ndim!=2 or local_new.shape[1]!=4 or (local_new.size and local_new.max()!=len(verts)):
                rejection["invalid_local_steiner_connectivity"] += 1
                kernel_cone_stats["rejected"] += 1
                continue
            local_to_global=np.r_[verts,np.asarray([point_row],np.int64)]
            new=local_to_global[local_new]
            candidate_local_points=np.vstack((lp,apex.reshape(1,3)))
            if not exact_positive(candidate_local_points,local_new):
                rejection["new_cell_not_exact_positive"] += 1
                kernel_cone_stats["rejected"] += 1
                continue
            ok,why=M.certify(candidate_local_points,old_local,local_new)
            if not ok:
                rejection["exact_cavity_certificate_"+str(why)] += 1
                kernel_cone_stats["rejected"] += 1
                continue
            before_bad=int(np.count_nonzero(~M.passes(old_det[rows],old_q[rows])))
            before_score=float(deficits[rows].sum())
            after_bad,after_score,new_det,new_q=quality_score(candidate_local_points,local_new)
            if before_bad<=0 or after_bad!=0 or after_bad>=before_bad:
                rejection["not_strict_all_new_good_bad_count_drop"] += 1
                kernel_cone_stats["rejected"] += 1
                continue
            if not np.all(M.passes(new_det,new_q)):
                rejection["new_cell_owner_quality_gate"] += 1
                kernel_cone_stats["rejected"] += 1
                continue
            replacement_offset=sum(len(op["new_tetrahedra"]) for op in ops)
            trial={
                "seed_cell_row":int(seed),"old_cell_rows":rows.tolist(),
                "old_vertex_rows":verts.tolist(),"old_cells":int(len(old)),
                "old_bad":int(before_bad),"old_deficit":float(before_score),
                "open_edges":int(open_edges),"proposal_method":"interior_steiner_kernel_cone",
                "candidate_selection_method":selection_label,
                "kernel_cone":cone_summary,"apex_point_output_row":int(point_row),
                "replacement_tet_range":[int(replacement_offset),int(replacement_offset+len(new))]}
            trial_rows.append(trial)
            record={
                "proposal_method":"interior_steiner_kernel_cone",
                "candidate_selection_method":selection_label,
                "seed_cell_row":int(seed),"old_cell_rows":rows.tolist(),
                "old_vertex_rows":verts.tolist(),"apex_point_f32":apex.tolist(),
                "apex_point_output_row":int(point_row),
                "new_tetrahedra":new.tolist(),"replacement_tet_range":trial["replacement_tet_range"],
                "material_index":int(mat[rows[0]]),"old_cell_count":int(len(old)),
                "new_cell_count":int(len(new)),"bad_before":int(before_bad),
                "bad_after":int(after_bad),"log_deficit_before":float(before_score),
                "log_deficit_after":float(after_score),"exact_cavity_certificate":why,
                "exact_positive_new_cells":True,"all_new_cells_native_pass":True,
                "exact_volume_conserved":True,"oriented_chain_matches_input":True,
                "full_cavity_parent_rows":rows.astype(int).tolist(),
                "partial_bad_replacement":False,"solver_report":cone_summary}
            ops.append(record)
            accepted_point_keys.add(apex_key)
            kernel_cone_stats["accepted"] += 1
            reserved_cells.update(map(int,rows))
            reserved_vertices.update(map(int,verts))
        if (time.monotonic()>=deadline or len(ops)>=args.maximum_operations or len(attempted)>=args.maximum_attempts):
            break
    active_proposal_s=time.monotonic()-proposal_started
    if not ops:
        return {"role": role, "status": "NO_VALID_BOUNDED_PARTIAL_FIXED_BOUNDARY_QUALITY_PROGRESS",
                "input": pin(source_path), "input_cells": int(len(t)),
                "native_bad_before": int(old_bad), "native_bad_after": int(old_bad),
                "attempts": trial_rows, "rejections": dict(rejection),
                "kernel_cone_cache": dict(kernel_cone_stats),
                "kernel_cone_cache_entries": len(kernel_cone_cache),
                "candidate_selection_counts":dict(candidate_selection_counts),
                "preparation_s":preparation_s,"index_preparation_s":index_preparation_s,
                "active_proposal_s":active_proposal_s,
                "bad_corner_vertices":int(len(bad_corner_vertices)),
                "bad_corner_incident_cell_rows":bad_corner_incident_cell_rows,
                "elapsed_s": time.monotonic() - role_started}
    removed = np.asarray(sorted({r for op in ops for r in op["old_cell_rows"]}),np.int64)
    keep=np.ones(len(t),bool);keep[removed]=False;retained=np.flatnonzero(keep)
    new_blocks=[np.asarray(op["new_tetrahedra"],np.int64) for op in ops]
    replacement=np.vstack(new_blocks)
    replacement_material=np.concatenate([np.full(len(block),int(op["material_index"]),mat.dtype) for op,block in zip(ops,new_blocks)])
    new_points=np.asarray([op["apex_point_f32"] for op in ops],np.float32).reshape((-1,3))
    out_points=np.vstack((p,new_points))
    out_t=np.vstack((t[keep],replacement));out_mat=np.concatenate((mat[keep],replacement_material))
    if p.tobytes(order="C")!=np.asarray(a["points_world_m"]).tobytes(order="C"):raise AssertionError("point prefix changed before append")
    det2,q2=M.measurements(out_points,out_t)
    if np.any(det2<=0):raise ValueError(f"{role} output has nonpositive measured cells")
    if not exact_positive(out_points,replacement,exact_point_cache):raise ValueError(f"{role} replacement exact-positive gate failed")
    if not np.all(M.passes(*M.measurements(out_points,replacement))):raise ValueError(f"{role} replacement owner quality gate failed")
    if len(np.unique(np.sort(out_t,axis=1),axis=0))!=len(out_t):raise ValueError(f"{role} output has duplicate tetrahedra")
    out_boundary=T.m.boundary(out_t)
    if not np.array_equal(T.m.boundary_identity(out_boundary),boundary_before):raise ValueError(f"{role} global oriented boundary changed")
    if not np.array_equal(T.m.boundary_identity(out_boundary),T.m.boundary_identity(target)):raise ValueError(f"{role} output no longer matches current target")
    new_bad,new_deficit,_,_=quality_score(out_points,out_t)
    if new_bad>=old_bad:raise ValueError(f"{role} did not strictly reduce native-bad count")
    origin=np.r_[retained,np.full(len(replacement),-1,np.int64)]
    inverse=np.full(len(t),-1,np.int64);inverse[retained]=np.arange(len(retained),dtype=np.int64)
    replacement_parent_offsets=[0];replacement_parent_rows=[]
    replacement_tet_offsets=[0];op_parent_offsets=[0];op_parent_rows=[]
    point_parent_offsets,point_parent_rows=_point_parent_csr(a,ops,len(p))
    for op,block in zip(ops,new_blocks):
        rows=sorted(set(map(int,op["old_cell_rows"])))
        for _ in block:
            replacement_parent_rows.extend(rows);replacement_parent_offsets.append(len(replacement_parent_rows))
        op_parent_rows.extend(rows);op_parent_offsets.append(len(op_parent_rows))
        replacement_tet_offsets.append(replacement_tet_offsets[-1]+len(block))
    payload=dict(a)
    payload["points_world_m"]=out_points.astype(np.asarray(a["points_world_m"]).dtype,copy=False)
    if "points_world_m_f32" in a:payload["points_world_m_f32"]=out_points.copy()
    updates={
      "tetrahedra":out_t,"material_indices":out_mat,"boundary_faces":out_boundary,
      "desired_boundary_faces":np.asarray(target,np.int64),
      "output_cell_to_input_cell":origin,"input_cell_to_output_cell":inverse,
      "retained_input_cell_rows":retained,"replacement_tetrahedra":replacement,
      "replacement_parent_offsets":np.asarray(replacement_parent_offsets,np.int64),
      "replacement_parent_cell_rows":np.asarray(replacement_parent_rows,np.int64),
      "replacement_tet_offsets":np.asarray(replacement_tet_offsets,np.int64),
      "quality_fixed_boundary_operation_parent_offsets":np.asarray(op_parent_offsets,np.int64),
      "quality_fixed_boundary_operation_parent_cell_rows":np.asarray(op_parent_rows,np.int64),
      "quality_fixed_boundary_operation_count":np.asarray([len(ops)],np.int64),
      "quality_fixed_boundary_operation_methods":np.asarray([op["proposal_method"] for op in ops]),
      "quality_fixed_boundary_operation_seed_rows":np.asarray([op["seed_cell_row"] for op in ops],np.int64),
      "quality_fixed_boundary_input_candidate_path":np.asarray([str(source_path)]),
      "quality_fixed_boundary_input_candidate_sha256":np.asarray([source_sha]),
      "quality_milp_operation_parent_offsets":np.asarray([0],np.int64),
      "quality_milp_operation_parent_cell_rows":np.empty(0,np.int64),
      "quality_milp_partial_bad_replacement":np.asarray([False]),
      "quality_milp_operation_count":np.asarray([0],np.int64),
      "quality_milp_operation_seed_rows":np.empty(0,np.int64),
      "quality_milp_input_candidate_path":np.asarray([str(source_path)]),
      "quality_milp_input_candidate_sha256":np.asarray([source_sha]),
      "quality_milp_input_node_to_output_node":np.arange(len(p),dtype=np.int64),
      "quality_milp_output_node_to_input_node":np.r_[np.arange(len(p),dtype=np.int64),np.full(len(new_points),-1,np.int64)],
      "quality_steiner_point_output_rows":np.asarray([op["apex_point_output_row"] for op in ops],np.int64),
      "quality_steiner_operation_count":np.asarray([len(ops)],np.int64),
      "quality_steiner_cavity_selection_methods":np.asarray([op["candidate_selection_method"] for op in ops]),
      "quality_steiner_kernel_helper_path":np.asarray([str(KERNEL_PATH)]),
      "quality_steiner_kernel_helper_sha256":np.asarray([EXPECTED_KERNEL_SHA256]),
      "current_geometry_parent_path":np.asarray([str(source_path)]),
      "current_geometry_parent_sha256":np.asarray([source_sha]),
      "canonical_parent_union_operation_order":np.asarray("interior-steiner-kernel-cone-full-original-cavity-union-v1"),
    }
    updates.update(_point_map_updates(a,len(p),len(new_points),point_parent_offsets,point_parent_rows))
    if "source_interior_vertex_quotient" in a:
        updates["source_interior_vertex_quotient"]=np.r_[np.asarray(a["source_interior_vertex_quotient"],np.int64),np.full(len(new_points),-1,np.int64)]
    if "metadata_finalizer_point_map_status" in a:
        updates["metadata_finalizer_point_map_status"]=np.asarray("APPENDED_INTERIOR_STEINER_POINTS_NO_SOURCE_BINDING")
    if "metadata_finalizer_point_parent_status" in a:
        updates["metadata_finalizer_point_parent_status"]=np.asarray("COMPLETE_FULL_CAVITY_PARENT_CELL_CSR")
    updates.update(_historical_point_csr_updates(a,len(p),source_path,source_sha))
    payload.update(updates)
    output_dir=Path(outdir);output_dir.mkdir(parents=True,exist_ok=True)
    output=output_dir/f"{role.lower()}-interior-steiner-quality-proposal.npz"
    np.savez_compressed(output,**payload);out_sha=sha(output);rb=load_npz(output)
    if sha(output)!=out_sha:raise ValueError("candidate changed during readback")
    readback=independent_readback_replay(a,rb,ops,retained,target,old_bad,new_bad,source_path,source_sha)
    readback_path=output_dir/f"{role.lower()}-independent-steiner-readback.json"
    readback_path.write_text(json.dumps(readback,indent=2,sort_keys=True)+"\n")
    newdet,nq=M.measurements(out_points,out_t)
    return {
      "role":role,"status":"PASS_INTERIOR_STEINER_KERNEL_CONE_BATCH",
      "input":pin(source_path),"output":pin(output),"independent_readback":pin(readback_path),
      "state_input_sha256":state.get("_state_sha256"),"input_cells":int(len(t)),
      "output_cells":int(len(out_t)),"input_points":int(len(p)),"output_points":int(len(out_points)),
      "new_steiner_points":int(len(new_points)),"native_bad_before":int(old_bad),
      "native_bad_after":int(new_bad),"log_deficit_before":float(old_deficit),
      "log_deficit_after":float(new_deficit),"parent_declared_nonpositive":int(declared_nonpositive),
      "exact_nonpositive_new_cells":0,"all_output_measured_positive":True,
      "all_replacement_cells_exact_positive":True,"all_new_cells_native_pass":True,
      "strict_native_bad_count_drop":True,"target_boundary_key":target_key,
      "oriented_target_boundary_unchanged":True,"all_new_cells_same_material_as_cavity":True,
      "complete_replacement_parent_union_csr":True,"complete_steiner_point_parent_union_csr":True,
      "immediate_old_point_parent_csr_empty":True,"historical_parent_point_csr_archived":True,
      "exact_volume_conserved_per_cavity":True,"accepted_operations":ops,
      "attempts":trial_rows,"rejections":dict(rejection),"native_thresholds_unchanged":True,
      "kernel_cone_helper":{"path":str(KERNEL_PATH),"sha256":EXPECTED_KERNEL_SHA256},
      "kernel_cone_cache":dict(kernel_cone_stats),"kernel_cone_cache_entries":len(kernel_cone_cache),
      "candidate_selection_counts":dict(candidate_selection_counts),
      "bad_corner_vertices":int(len(bad_corner_vertices)),
      "bad_corner_incident_cell_rows":bad_corner_incident_cell_rows,
      "preparation_s":preparation_s,"index_preparation_s":index_preparation_s,
      "active_proposal_s":active_proposal_s,
      "proposal_method_counts":dict(Counter(op["proposal_method"] for op in ops)),
      "surface_context":"UNCHANGED_SOURCE_SURFACES_AND_FIXED_VOLUME_BOUNDARY",
      "native_admission":False,"physical_admission":False,"native_steps":0,
      "mass_mapping":"NOT_RUN","elapsed_s":time.monotonic()-role_started}


def run_state(state_path, output_path, seconds, args):
    start = time.monotonic()
    state_pin = pin(state_path)
    raw = json.loads(Path(state_path).read_text())
    wrapped = "current_verified_state_20261031" in raw
    state = raw.get("current_verified_state_20261031", raw)
    state["_state_sha256"] = state_pin["sha256"]
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    role_results = []
    updated = dict(state)
    available = []
    for role in ("Skin", "Fat"):
        key = role + "_current"
        if key not in state:
            continue
        declared = state[key].get("native_quality_failures")
        if declared is None or int(declared) < 0:
            raise ValueError(f"{role} state must pin a nonnegative native_quality_failures count for scheduling")
        available.append((role, int(declared)))
    # Larger current debt gets first access to the same bounded solver, with
    # exact input-count validation inside run_role. Equal-count ties are Fat-first.
    available.sort(key=lambda item: (-item[1], 0 if item[0] == "Fat" else 1))
    schedule = []
    for idx, (role, pinned_bad) in enumerate(available):
        now = time.monotonic()
        remaining = max(0.0, start + seconds - now)
        roles_left = len(available) - idx
        slice_seconds = remaining / max(roles_left, 1)
        role_deadline = now + slice_seconds
        schedule.append({"role": role, "pinned_native_bad_before": pinned_bad,
                         "budget_seconds": float(slice_seconds),
                         "budget_policy": "equal_remaining_slice_sorted_by_bad_desc_fat_tiebreak"})
        key = role + "_current"
        if slice_seconds <= 0.01:
            role_results.append({"role": role, "status": "SKIPPED_GLOBAL_TIME_BUDGET"})
            continue
        result = run_role(role, state[key], state, output_path, role_deadline, args)
        role_results.append(result)
        if result.get("status") == "PASS_INTERIOR_STEINER_KERNEL_CONE_BATCH":
            updated[key] = {**state[key], **result["output"],
                            "cells": result["output_cells"],
                            "nodes": result["output_points"],
                            "native_quality_failures": result["native_bad_after"],
                            "nonpositive": 0}
    updated["physical_admission"] = False
    updated["native_steps"] = 0
    updated["mass_mapping"] = "NOT_RUN"
    updated.pop("independent_replay", None)
    updated.pop("exact_context", None)
    if wrapped:
        final_state = dict(raw)
        final_state["current_verified_state_20261031"] = updated
    else:
        final_state = updated
    state_out = output_path / "state.json"
    state_out.write_text(json.dumps(final_state, indent=2) + "\n")
    report = {
        "schema": "h706.interior-steiner-adaptive-kernel-cone-quality-batch.v1",
        "status": "PASS_WITH_PARTIAL_NATIVE_DEFECTS" if any(
            x.get("status") == "PASS_INTERIOR_STEINER_KERNEL_CONE_BATCH" for x in role_results
        ) else "NO_VALID_BOUNDED_PROGRESS",
        "source_state": state_pin, "output_state": pin(state_out),
        "producer": pin(Path(__file__)), "exact_quality_loader": pin(SOLVER_PATH),
        "topology_owner": pin(TOP_PATH), "kernel_cone_helper": pin(KERNEL_PATH),
        "role_results": role_results,
        "role_schedule": schedule,
        "roles_sorted_by_measured_source_bad_descending": True,
        "balanced_per_role_deadlines": True,
        "cavity_selection_methods":["complete_same_material_vertex_stars_per_seed_corner",
                                    "same_material_face_balls_2_4_8_12_24"],
        "maximum_vertex_star_cells":int(args.maximum_star_cells),
        "native_thresholds_unchanged": True,
        "all_cells_positive": True,
        "native_admission": False, "physical_admission": False, "native_steps": 0,
        "mass_mapping": "NOT_RUN",
        "elapsed_s": time.monotonic() - start,
        "time_budget_s": float(seconds),
    }
    report_path = output_path / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report

def fixture(output_path,args):
    out=Path(output_path);out.mkdir(parents=True,exist_ok=False)
    control=KERNEL.sliver_tetra_fixture()
    root3=np.sqrt(3.0);root23=np.sqrt(2.0/3.0)
    outer_points=np.asarray([[0.,0.,0.],[1.,0.,0.],[.5,root3/2,0.],
                             [.5,root3/6,root23]],np.float32)
    outer=M.orient(outer_points,np.asarray([[0,1,2,3]],np.int64))
    faces=KERNEL.K734.oriented_boundary(outer)
    near=np.asarray([.5,root3/6,1.0e-5],np.float32)
    points=np.vstack((outer_points,near))
    old=M.orient(points,np.column_stack((np.full(len(faces),4,np.int64),faces)))
    input_path=out/"synthetic-sliver-input.npz"
    np.savez_compressed(input_path,points_world_m=points,tetrahedra=old,
                        material_indices=np.full(len(old),17,np.int32),
                        boundary_faces=faces,desired_boundary_faces=faces,
                        output_point_parent_offsets=np.asarray([0,1,1,1,1,1],np.int64),
                        output_point_parent_input_cell_rows=np.asarray([2],np.int64),
                        nonpositive=np.asarray([0],np.int64))
    old_bad=quality_score(points,old)[0]
    fixture_materials=np.full(len(old),17,np.int32)
    fixture_bad_det,fixture_bad_q=M.measurements(points,old)
    fixture_bad=np.flatnonzero(~M.passes(fixture_bad_det,fixture_bad_q))
    fixture_adj=face_bad_adjacency(old,fixture_materials,fixture_bad)
    fixture_incidence={int(v):np.flatnonzero(np.any(old==int(v),axis=1)).astype(np.int32)
                       for v in np.unique(old[fixture_bad].reshape(-1))}
    star_cap=complete_same_material_vertex_star(int(fixture_bad[0]),0,fixture_materials,old,
        fixture_incidence,128,192,set(),set())
    star_too_small=complete_same_material_vertex_star(int(fixture_bad[0]),0,fixture_materials,old,
        fixture_incidence,2,192,set(),set())
    fixture_deficits=-np.log(np.maximum(np.minimum(np.minimum(fixture_bad_q/QUALITY_GATE,
        fixture_bad_det/(6.0*VOLUME_GATE)),1.0),1.0e-300))
    selected_candidates=adaptive_cavity_candidates(int(fixture_bad[0]),fixture_adj,fixture_bad,
        fixture_materials,old,fixture_incidence,24,128,192,set(),set(),fixture_deficits)
    selected_labels=[label for label,_ in selected_candidates]
    if star_cap is None or len(star_cap)!=3 or star_too_small is not None:
        raise AssertionError("complete-vertex-star fixture truncated or miscounted its star")
    if not any(label.startswith("complete_vertex_star_node_") for label in selected_labels) or "face_ball_2" not in selected_labels:
        raise AssertionError("adaptive candidate fixture omitted vertex-star or small face-ball alternatives")
    entry={"path":str(input_path),"sha256":sha(input_path),
           "native_quality_failures":old_bad,"nonpositive":0}
    producer=out/"producer"
    result=run_role("Skin",entry,{},producer,time.monotonic()+30.0,args)
    if result.get("status")!="PASS_INTERIOR_STEINER_KERNEL_CONE_BATCH":
        raise AssertionError("synthetic batch fixture did not produce an accepted operation: "+json.dumps(result,sort_keys=True))
    rb=load_npz(result["output"]["path"])
    readback=json.loads(Path(result["independent_readback"]["path"]).read_text())
    if readback.get("status")!="PASS_INDEPENDENT_STEINER_CAVITY_READBACK":
        raise AssertionError("synthetic independent readback did not pass")
    point_off=np.asarray(rb["output_point_parent_offsets"],np.int64)
    if not np.array_equal(point_off[:len(points)+1],np.zeros(len(points)+1,np.int64)):
        raise AssertionError("synthetic old point prefix was not reset to immediate identity rows")
    if not np.array_equal(np.asarray(rb["steiner_parent_point_csr_input_offsets"]),np.asarray([0,1,1,1,1,1],np.int64)) or not np.array_equal(
            np.asarray(rb["steiner_parent_point_csr_input_rows"]),np.asarray([2],np.int64)):
        raise AssertionError("synthetic predecessor point CSR was not archived")
    report={"schema":"h706.interior-steiner-kernel-cone-fixture.v3",
            "status":"PASS_INTERIOR_STEINER_BATCH_AND_INDEPENDENT_READBACK_FIXTURE",
            "helper_fixture":control,
            "adaptive_candidate_control":{"status":"PASS_COMPLETE_STAR_AND_FACE_BALLS",
                "candidate_labels":selected_labels,"complete_vertex_star_cells":len(star_cap),
                "too_small_star_cap_rejected":star_too_small is None},
            "producer_result":result,
            "readback":readback,"readback_npz_sha256":sha(result["output"]["path"]),
            "readback_points":int(len(rb["points_world_m"])),
            "physical_admission":False,"native_steps":0}
    (out/"fixture-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seconds",type=float,default=90.0)
    ap.add_argument("--maximum-old-cells",type=int,default=24)
    ap.add_argument("--maximum-points",type=int,default=192)
    ap.add_argument("--maximum-star-cells",type=int,default=128)
    ap.add_argument("--maximum-attempts",type=int,default=64)
    ap.add_argument("--maximum-operations",type=int,default=8)
    ap.add_argument("--construction-margin",type=float,default=1.0e-3)
    ap.add_argument("--fixture",action="store_true")
    args=ap.parse_args()
    if args.maximum_old_cells<1 or args.maximum_old_cells>24:ap.error("--maximum-old-cells must be within 1..24")
    if args.maximum_points<4 or args.maximum_points>256:ap.error("--maximum-points must be within 4..256")
    if args.maximum_star_cells<1 or args.maximum_star_cells>128:ap.error("--maximum-star-cells must be within 1..128")
    if args.seconds<=0 or args.construction_margin<0 or args.construction_margin>0.05:
        ap.error("seconds must be positive and construction margin in [0,0.05]")
    if args.fixture:
        result=fixture(args.output,args)
    else:
        if args.state is None:ap.error("--state is required unless --fixture is selected")
        result=run_state(args.state,args.output,args.seconds,args)
    role_rows=result.get("role_results",[])
    compact={
        "status":result.get("status"),
        "report":pin(args.output/"report.json") if not args.fixture else pin(args.output/"fixture-report.json"),
        "roles":[{k:r.get(k) for k in ("role","status","native_bad_before","native_bad_after",
                  "new_steiner_points","preparation_s","active_proposal_s","elapsed_s")}
                 for r in role_rows],
        "fixture_status":result.get("status") if args.fixture else None,
        "physical_admission":False,"native_steps":0}
    print(json.dumps(compact,sort_keys=True))
if __name__=="__main__":main()

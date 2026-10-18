"""Fail-closed indexed triangle plane split with ancestry and linear attributes.

Visual geometry preparation only; it does not implement physical fracture.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping
import numpy as np


class PlaneCutError(ValueError):
    """Unsupported or invalid split input."""


@dataclass(frozen=True)
class SurfaceHalf:
    vertices: np.ndarray
    triangles: np.ndarray
    attributes: dict[str, np.ndarray]
    vertex_provenance: tuple[dict[str, object], ...]
    triangle_source_face: np.ndarray
    triangle_kind: np.ndarray
    triangle_cap_loop: np.ndarray
    boundary_edges: np.ndarray
    cap_loop_projection_residuals: np.ndarray | None = None


def _edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _uses(faces: np.ndarray):
    rows = defaultdict(list)
    for fi, face in enumerate(faces):
        a, b, c = map(int, face)
        for x, y in ((a, b), (b, c), (c, a)):
            rows[_edge(x, y)].append((fi, x, y))
    return rows


def _validate(points, faces, declared, tol):
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise PlaneCutError("vertices must be finite N x 3 coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise PlaneCutError("triangles must be a nonempty M x 3 array")
    if np.any(faces < 0) or np.any(faces >= len(points)):
        raise PlaneCutError("triangle index is out of range")
    sf = np.sort(faces, axis=1)
    if np.any(sf[:, 1:] == sf[:, :-1]) or len({tuple(x) for x in sf.tolist()}) != len(faces):
        raise PlaneCutError("source repeats a face or a face vertex")
    scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1.0)
    for face in faces:
        a, b, c = points[face]
        area2 = float(np.linalg.norm(np.cross(b-a, c-a)))
        longest = max(float(np.linalg.norm(b-a)), float(np.linalg.norm(c-b)),
                      float(np.linalg.norm(a-c)))
        if not np.isfinite(area2) or area2 <= tol*max(longest, scale*1e-12):
            raise PlaneCutError("source contains a degenerate triangle")
    rows = _uses(faces)
    boundary = {e for e, u in rows.items() if len(u) == 1}
    if {e for e, u in rows.items() if len(u) not in (1, 2)}:
        raise PlaneCutError("source has an edge with more than two faces")
    if boundary != declared:
        raise PlaneCutError(f"source boundary differs from declaration ({len(boundary)} != {len(declared)})")
    for u in rows.values():
        if len(u) == 2 and (u[0][1], u[0][2]) != (u[1][2], u[1][1]):
            raise PlaneCutError("source winding is inconsistent across an edge")
    incident = defaultdict(list)
    for fi, face in enumerate(faces):
        for x in face:
            incident[int(x)].append(fi)
    boundary_vertices = {x for e in boundary for x in e}
    for vertex, fids in incident.items():
        link = defaultdict(set)
        for fi in fids:
            others = [int(x) for x in faces[fi] if int(x) != vertex]
            if len(others) != 2:
                raise PlaneCutError("source vertex link is malformed")
            a, b = others
            link[a].add(b); link[b].add(a)
        reached, stack = set(), [next(iter(link))]
        while stack:
            x = stack.pop()
            if x in reached: continue
            reached.add(x); stack.extend(link[x]-reached)
        if reached != set(link) or any(len(ns) not in (1, 2) for ns in link.values()):
            raise PlaneCutError("source has a disconnected or branching vertex link")
        degree1 = sum(len(ns) == 1 for ns in link.values())
        if (vertex in boundary_vertices and degree1 != 2) or (vertex not in boundary_vertices and degree1 != 0):
            raise PlaneCutError("source vertex link disagrees with boundary")
    bad = defaultdict(set)
    for a, b in boundary:
        bad[a].add(b); bad[b].add(a)
    if any(len(ns) != 2 for ns in bad.values()):
        raise PlaneCutError("existing boundary must be declared as simple loops")
    return boundary


def _cross2(a, b, c):
    return float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))


def _on_segment(p, a, b, eps):
    return (abs(_cross2(a, b, p)) <= eps and
            min(a[0], b[0])-eps <= p[0] <= max(a[0], b[0])+eps and
            min(a[1], b[1])-eps <= p[1] <= max(a[1], b[1])+eps)


def _intersect(a, b, c, d, eps):
    o1, o2, o3, o4 = _cross2(a,b,c), _cross2(a,b,d), _cross2(c,d,a), _cross2(c,d,b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    return (_on_segment(c,a,b,eps) or _on_segment(d,a,b,eps) or
            _on_segment(a,c,d,eps) or _on_segment(b,c,d,eps))


def _area(poly):
    return 0.5*float(np.sum(poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1)))


def _inside(p, poly, eps):
    value = False
    x, y = map(float, p)
    for a, b in zip(poly, np.roll(poly,-1,axis=0), strict=True):
        if _on_segment(p,a,b,eps): return True
        if (a[1] > y) != (b[1] > y):
            crossx = a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if x < crossx: value = not value
    return value


def _earclip(ids, poly, eps):
    if _area(poly) < 0:
        ids, poly = list(reversed(ids)), poly[::-1].copy()
    remain, result = list(range(len(ids))), []
    guard = len(ids)*len(ids)
    while len(remain) > 3 and guard:
        guard -= 1
        ear = None
        for pos, cur in enumerate(remain):
            prev, nxt = remain[pos-1], remain[(pos+1)%len(remain)]
            if _cross2(poly[prev],poly[cur],poly[nxt]) <= eps: continue
            blocked = False
            for k in remain:
                if k in (prev,cur,nxt): continue
                p = poly[k]
                if (_cross2(poly[prev],poly[cur],p) >= -eps and
                    _cross2(poly[cur],poly[nxt],p) >= -eps and
                    _cross2(poly[nxt],poly[prev],p) >= -eps):
                    blocked = True; break
            if not blocked:
                ear = pos, (ids[prev],ids[cur],ids[nxt]); break
        if ear is None:
            raise PlaneCutError("cap polygon cannot be triangulated without dropping boundary vertices")
        pos, tri = ear
        result.append(tri); remain.pop(pos)
    if len(remain) != 3 or _cross2(poly[remain[0]],poly[remain[1]],poly[remain[2]]) <= eps:
        raise PlaneCutError("cap triangulation ended in a degenerate triangle")
    result.append(tuple(ids[k] for k in remain))
    return result


def _loops(segments):
    adj, seen_edges = defaultdict(set), set()
    for a,b in segments:
        key = _edge(a,b)
        if a == b or key in seen_edges: raise PlaneCutError("cut contour repeats a segment")
        seen_edges.add(key); adj[a].add(b); adj[b].add(a)
    if not adj or any(len(ns) != 2 for ns in adj.values()):
        raise PlaneCutError("new cut contour is not a union of closed loops")
    unseen, result = set(adj), []
    while unseen:
        start = min(unseen); row = [start]; prev, cur = None, start
        while True:
            nxt = sorted(x for x in adj[cur] if x != prev)[0]
            if nxt == start: break
            if nxt not in unseen: raise PlaneCutError("cut loops touch or repeat vertices")
            row.append(nxt); prev,cur=cur,nxt
        unseen.difference_update(row); result.append(row)
    return result


def _output_boundary(faces, provenance, expected):
    rows = _uses(faces)
    if any(len(u) not in (1,2) for u in rows.values()):
        raise PlaneCutError("split output has a nonmanifold edge")
    actual_local = {e for e,u in rows.items() if len(u)==1}
    actual = set()
    for a,b in actual_local:
        pa,pb=provenance[a],provenance[b]
        if pa["kind"] != "source_vertex" or pb["kind"] != "source_vertex":
            raise PlaneCutError("split emitted an undeclared new boundary edge")
        actual.add(_edge(int(pa["source_vertex"]),int(pb["source_vertex"])))
    if actual != expected:
        raise PlaneCutError(f"output boundary differs from retained source openings ({len(actual)} != {len(expected)})")
    for u in rows.values():
        if len(u)==2 and (u[0][1],u[0][2]) != (u[1][2],u[1][1]):
            raise PlaneCutError("split output winding is inconsistent")
    return np.asarray(sorted(actual_local),dtype=np.int64).reshape((-1,2))


def split_closed_surface(vertices, triangles, plane_origin, plane_normal, *,
                         attributes: Mapping[str,np.ndarray] | None = None,
                         allowed_existing_boundary_edges: set[tuple[int,int]] | None = None,
                         tolerance=1e-11):
    """Clip to both plane halfspaces, cap new cut loops, preserve declared old loops.

    The declared boundary must exactly equal all input boundary edges and form
    simple loops wholly on one side. By default input is required to be closed.
    Nested or intersecting cap contours fail closed; no hole fans are invented.
    """
    points=np.asarray(vertices,dtype=np.float64)
    faces=np.asarray(triangles,dtype=np.int64)
    origin=np.asarray(plane_origin,dtype=np.float64)
    normal=np.array(plane_normal,dtype=np.float64,copy=True)
    if origin.shape!=(3,) or normal.shape!=(3,) or not np.isfinite(origin).all() or not np.isfinite(normal).all():
        raise PlaneCutError("plane must have finite 3-vector origin and normal")
    magnitude=float(np.linalg.norm(normal))
    if magnitude<=0: raise PlaneCutError("plane normal is zero")
    normal/=magnitude
    tol=float(tolerance)
    if not np.isfinite(tol) or tol<=0: raise PlaneCutError("tolerance must be positive")
    declared=set() if allowed_existing_boundary_edges is None else {_edge(int(a),int(b)) for a,b in allowed_existing_boundary_edges}
    boundary=_validate(points,faces,declared,tol)
    if any(a==b or a<0 or b>=len(points) for a,b in declared):
        raise PlaneCutError("declared boundary references invalid source vertex")
    attrs={}
    for name,value in (attributes or {}).items():
        arr=np.asarray(value)
        if arr.ndim<1 or arr.shape[0]!=len(points) or not np.isfinite(arr).all():
            raise PlaneCutError(f"attribute {name!r} must have finite rows for all vertices")
        attrs[name]=arr.astype(np.float64,copy=False)
    dist=(points-origin)@normal
    if np.any(np.abs(dist)<=tol): raise PlaneCutError("plane passes through a source vertex")
    side=np.where(dist>0,1,-1)
    if not (np.any(side>0) and np.any(side<0)): raise PlaneCutError("plane does not separate source")
    for a,b in boundary:
        if side[a]!=side[b]: raise PlaneCutError("plane intersects a retained existing opening")
    allp=[p.copy() for p in points]
    alla={n:[r.copy() for r in arr] for n,arr in attrs.items()}
    prov=[{"kind":"source_vertex","source_vertex":int(i)} for i in range(len(points))]
    intersections={}
    def crossing(a,b):
        key=_edge(a,b)
        if key in intersections: return intersections[key]
        u,v=key; t=float(dist[u]/(dist[u]-dist[v]))
        if not 0<t<1: raise PlaneCutError("edge crossing is outside source edge")
        idx=len(allp); allp.append(points[u]+t*(points[v]-points[u]))
        for n,arr in attrs.items(): alla[n].append(arr[u]+t*(arr[v]-arr[u]))
        prov.append({"kind":"source_edge","source_edge":[u,v],"t_from_first":t})
        intersections[key]=idx
        return idx
    def clip(face,s):
        poly=[]; ids=list(map(int,face))
        for i,a in enumerate(ids):
            b=ids[(i+1)%3]; ia,ib=side[a]==s,side[b]==s
            if ia and ib: poly.append(b)
            elif ia and not ib: poly.append(crossing(a,b))
            elif not ia and ib: poly.extend((crossing(a,b),b))
        out=[]
        for x in poly:
            if not out or out[-1]!=x: out.append(x)
        if len(out)>1 and out[0]==out[-1]: out.pop()
        return out
    byside={1:[],-1:[]}; segments=[]
    for fi,face in enumerate(faces):
        signs=side[face]
        if np.any(signs>0) and np.any(signs<0):
            ids=[]
            for i in range(3):
                a,b=int(face[i]),int(face[(i+1)%3])
                if side[a]!=side[b]: ids.append(crossing(a,b))
            ids=list(dict.fromkeys(ids))
            if len(ids)!=2: raise PlaneCutError("crossed triangle does not produce one segment")
            segments.append(tuple(ids))
        for s in (1,-1):
            poly=clip(face,s)
            for k in range(1,len(poly)-1):
                tri=(poly[0],poly[k],poly[k+1])
                a,b,c=(np.asarray(allp[x]) for x in tri)
                if np.linalg.norm(np.cross(b-a,c-a))<=tol*max(np.linalg.norm(b-a),np.linalg.norm(c-a),1e-12):
                    raise PlaneCutError("clipping produced a degenerate triangle")
                byside[s].append((tri,fi))
    loops=_loops(segments)
    axis=np.eye(3)[int(np.argmin(np.abs(normal)))]
    u=np.cross(axis,normal);u/=np.linalg.norm(u)
    v=np.cross(normal,u);v/=np.linalg.norm(v)
    eps=max(tol*max(float(np.linalg.norm(np.ptp(points,axis=0))),1e-12),1e-24)
    polygons=[]
    for loop in loops:
        poly=np.column_stack(((np.asarray(allp)[loop]-origin)@u,(np.asarray(allp)[loop]-origin)@v))
        if len(loop)<3 or abs(_area(poly))<=eps: raise PlaneCutError("cut loop has zero area")
        for i in range(len(loop)):
            for j in range(i+1,len(loop)):
                if j==(i+1)%len(loop) or i==(j+1)%len(loop): continue
                if _intersect(poly[i],poly[(i+1)%len(loop)],poly[j],poly[(j+1)%len(loop)],eps):
                    raise PlaneCutError("cut loop is self-intersecting or touching")
        polygons.append(poly)
    for i in range(len(loops)):
        for j in range(i+1,len(loops)):
            pa,pb=polygons[i],polygons[j]
            if any(_intersect(pa[a],pa[(a+1)%len(pa)],pb[b],pb[(b+1)%len(pb)],eps)
                   for a in range(len(pa)) for b in range(len(pb))):
                raise PlaneCutError("cut loops intersect or touch")
            if _inside(pa[0],pb,eps) or _inside(pb[0],pa,eps):
                raise PlaneCutError("nested cap loops require unsupported hole triangulation")
    expected={1:set(),-1:set()}
    for a,b in boundary: expected[int(side[a])].add((a,b))
    def build(s):
        used=sorted({x for tri,_ in byside[s] for x in tri}|{x for e in expected[s] for x in e})
        if not used: raise PlaneCutError("one half has no triangles")
        remap={x:i for i,x in enumerate(used)}
        out=[]; src=[]; kinds=[]; caps=[]
        for tri,fi in byside[s]:
            out.append(tuple(remap[x] for x in tri));src.append(fi);kinds.append(0);caps.append(-1)
        for li,loop in enumerate(loops):
            ids=list(loop); poly=polygons[li]; want_ccw=s<0
            if (_area(poly)>0)!=want_ccw: ids.reverse();poly=poly[::-1].copy()
            cap_tris=_earclip(ids,poly,eps)  # CCW in the plane basis: +normal
            if s > 0:  # positive-side cap outward normal is -normal
                cap_tris=[(a,c,b) for a,b,c in cap_tris]
            for tri in cap_tris:
                out.append(tuple(remap[x] for x in tri));src.append(-1);kinds.append(1);caps.append(li)
        localprov=tuple(prov[x] for x in used)
        facesout=np.asarray(out,dtype=np.int64).reshape((-1,3))
        be=_output_boundary(facesout,localprov,expected[s])
        localattrs={n:np.asarray([alla[n][x] for x in used],dtype=np.float64) for n in attrs}
        return SurfaceHalf(np.asarray([allp[x] for x in used]),facesout,localattrs,localprov,
            np.asarray(src,dtype=np.int64),np.asarray(kinds,dtype=np.uint8),
            np.asarray(caps,dtype=np.int64),be,np.zeros(0,dtype=np.float64))
    return build(1),build(-1)


def decode_nhskin_abi5(raw: bytes):
    """Decode NHSKIN ABI 5 arrays for an owner-specific wrapper."""
    import struct
    h=struct.Struct("<8s5I32s")
    if len(raw)<h.size: raise PlaneCutError("NHSKIN header is truncated")
    magic,abi,nb,nv,ni,fp,sha=h.unpack_from(raw)
    if magic!=b"NHSKIN1\0" or abi!=5 or min(nb,nv,ni)<=0:
        raise PlaneCutError("input is not supported NHSKIN ABI 5")
    bo=h.size;vo=bo+36*nb;io=vo+56*nv;wo=io+4*ni
    if len(raw)!=wo+4*nv*nb: raise PlaneCutError("NHSKIN extent disagrees with header")
    vr=np.frombuffer(raw,"<f4",count=nv*14,offset=vo).reshape(nv,14).copy()
    ix=np.frombuffer(raw,"<u4",count=ni,offset=io).copy()
    wt=np.frombuffer(raw,"<f4",count=nv*nb,offset=wo).reshape(nv,nb).copy()
    # The four diagnostic binding indices occupy integer lanes in the ABI;
    # their float32 bit views may legitimately be NaN for the 0xFFFFFFFF pad.
    if (not np.isfinite(vr[:,:6]).all() or not np.isfinite(vr[:,10:14]).all()
            or not np.isfinite(wt).all() or np.any(wt<0)
            or np.max(np.abs(wt.sum(axis=1)-1))>3e-5):
        raise PlaneCutError("NHSKIN positions, normals, diagnostic weights, or full weights are invalid")
    return {"abi":abi,"binding_count":nb,"vertex_count":nv,"index_count":ni,
        "registration_fingerprint":fp,"source_sha256":sha.hex(),
        "binding_rows":raw[bo:vo],"vertex_rows":vr,"weights":wt,
        "triangles":ix.reshape((-1,3)).astype(np.int64)}


def split_nhskin_abi5_rest_world(raw: bytes, world_vertices: np.ndarray,
                                plane_origin: np.ndarray, plane_normal: np.ndarray, *,
                                allowed_existing_boundary_edges=None,
                                tolerance=1.0e-11):
    """Split NHSKIN ABI 5 geometry at its bound default-rest world positions.

    world_vertices must come from the existing NHSKIN owner reconstruction
    (resting_scene.source_shell_world); the complete ABI-5 body weight matrix
    and registered-rest normals are interpolated with every new edge vertex.
    No payload is rewritten and this helper does not infer physical fracture.
    """
    decoded = decode_nhskin_abi5(raw)
    points = np.asarray(world_vertices, dtype=np.float64)
    if points.shape != (decoded["vertex_count"], 3) or not np.isfinite(points).all():
        raise PlaneCutError("rest-world NHSKIN vertices do not match payload vertex rows")
    rows = decoded["vertex_rows"]
    return split_closed_surface(
        points, decoded["triangles"], plane_origin, plane_normal,
        attributes={
            "full_binding_weights": decoded["weights"],
            "registered_rest_normals": rows[:, 3:6],
        },
        allowed_existing_boundary_edges=allowed_existing_boundary_edges,
        tolerance=tolerance,
    )

def _rotation_xyzw(q):
    x,y,z,w=map(float,q)
    return np.array([
        [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],
    ],dtype=np.float64)


def emit_nhskin_abi5_half(raw: bytes, half: SurfaceHalf, rigid: dict,
                          skin: dict, allowed_body_indices: set[int]):
    """Emit one same-ABI visual half with binding rows filtered to its body side.

    The input plane geometry must be the registered default-rest world shell.
    Complete body weights are filtered and renormalized to allowed_body_indices.
    New atlas positions are solved through the resulting weighted binding
    transform so their registered rest-world positions stay fixed. Singular
    blends, missing owner support, or excessive float32 rest error fail closed.
    """
    import struct

    decoded=decode_nhskin_abi5(raw)
    weights=np.asarray(half.attributes.get("full_binding_weights"),dtype=np.float64)
    normals=np.asarray(half.attributes.get("registered_rest_normals"),dtype=np.float64)
    target_world=np.asarray(half.vertices,dtype=np.float64)
    if weights.ndim!=2 or weights.shape!=(len(target_world),decoded["binding_count"]):
        raise PlaneCutError("half is missing full NHSKIN binding-weight attributes")
    if normals.shape!=(len(target_world),3) or not np.isfinite(normals).all():
        raise PlaneCutError("half is missing registered rest normals")
    allowed=set(int(x) for x in allowed_body_indices)
    source_bindings=skin.get("bindings")
    rigid_poses=rigid.get("core_poses")
    if not isinstance(source_bindings,list) or len(source_bindings)!=decoded["binding_count"] or not isinstance(rigid_poses,dict):
        raise PlaneCutError("owner loader binding rows do not match NHSKIN ABI 5")
    selected=[i for i,row in enumerate(source_bindings) if int(row[0]) in allowed]
    if not selected:
        raise PlaneCutError("half has no NHSKIN bindings on its declared body side")
    filtered=weights[:,selected].copy()
    totals=filtered.sum(axis=1)
    if not np.isfinite(filtered).all() or np.any(filtered<0) or np.any(totals<=1e-12):
        raise PlaneCutError("some half vertices have no positive allowed-side binding weight")
    filtered/=totals[:,None]
    # Match the existing full-matrix ABI precision before inversion and check.
    filtered32=np.asarray(filtered,dtype="<f4")
    if np.max(np.abs(filtered32.astype(np.float64).sum(axis=1)-1.0))>3e-5:
        raise PlaneCutError("filtered NHSKIN weight rows do not sum to one")
    binding_transforms=[]
    for index in selected:
        body,translation,bind_quat,scale=source_bindings[index]
        if int(body) not in rigid_poses:
            raise PlaneCutError("filtered binding body has no registered rigid rest pose")
        pose=np.asarray(rigid_poses[int(body)],dtype=np.float64)
        if pose.shape!=(7,):
            raise PlaneCutError("registered rigid pose has invalid shape")
        Rbody=_rotation_xyzw(pose[3:7])
        Rbind=_rotation_xyzw(bind_quat)
        A=float(scale)*(Rbody@Rbind)
        b=pose[:3]+Rbody@np.asarray(translation,dtype=np.float64)
        binding_transforms.append((A,b))
    local_positions=np.empty_like(target_world)
    max_error=0.0
    for row,world_point in enumerate(target_world):
        A=np.zeros((3,3),dtype=np.float64)
        b=np.zeros(3,dtype=np.float64)
        for lane,(matrix,offset) in enumerate(binding_transforms):
            w=float(filtered32[row,lane])
            A+=w*matrix
            b+=w*offset
        condition=float(np.linalg.cond(A))
        if not np.isfinite(condition) or condition>1e10:
            raise PlaneCutError(f"filtered vertex binding transform is singular (condition={condition:.6g})")
        try:
            local_positions[row]=np.linalg.solve(A,world_point-b)
        except np.linalg.LinAlgError as exc:
            raise PlaneCutError("filtered vertex binding transform cannot be inverted") from exc
        restored=b+A@local_positions[row].astype("<f4").astype(np.float64)
        error=float(np.linalg.norm(restored-world_point))
        max_error=max(max_error,error)
    if not np.isfinite(local_positions).all() or max_error>1e-4:
        raise PlaneCutError(f"filtered NHSKIN rest reconstruction error exceeds 0.1 mm ({max_error:.6g} m)")
    normal_lengths=np.linalg.norm(normals,axis=1)
    if np.any(normal_lengths<=1e-12):
        raise PlaneCutError("half contains a zero registered rest normal")
    normalized_normals=normals/normal_lengths[:,None]
    active=filtered32.astype(np.float64)
    diag_ids=np.zeros((len(target_world),4),dtype="<u4")  # ABI loader validates even zero-weight diagnostic lanes
    diag_weights=np.zeros((len(target_world),4),dtype="<f4")
    for row in range(len(target_world)):
        order=sorted(range(len(selected)),key=lambda i:(-float(active[row,i]),selected[i]))
        order=[i for i in order if active[row,i]>1e-8][:4]
        if not order:
            raise PlaneCutError("half vertex has no diagnostic active binding")
        total=float(sum(active[row,i] for i in order))
        for lane,i in enumerate(order):
            diag_ids[row,lane]=i
            diag_weights[row,lane]=float(active[row,i])/total
    dtype=np.dtype([("p","<f4",(3,)),("n","<f4",(3,)),("bi","<u4",(4,)),("bw","<f4",(4,))])
    if dtype.itemsize!=56:
        raise PlaneCutError("NHSKIN vertex ABI packing is not 56 bytes")
    vertex_rows=np.empty(len(target_world),dtype=dtype)
    vertex_rows["p"]=np.asarray(local_positions,dtype="<f4")
    vertex_rows["n"]=np.asarray(normalized_normals,dtype="<f4")
    vertex_rows["bi"]=diag_ids
    vertex_rows["bw"]=diag_weights
    if np.any(half.triangles<0) or np.any(half.triangles>0xFFFFFFFF):
        raise PlaneCutError("output NHSKIN indices exceed uint32")
    index_bytes=np.asarray(half.triangles,dtype="<u4").reshape(-1).tobytes()
    binding_bytes=b"".join(
        decoded["binding_rows"][i*36:(i+1)*36] for i in selected
    )
    header=struct.pack("<8s5I32s",b"NHSKIN1\0",5,len(selected),len(target_world),
                       int(half.triangles.size),decoded["registration_fingerprint"],
                       bytes.fromhex(decoded["source_sha256"]))
    payload=b"".join((header,binding_bytes,vertex_rows.tobytes(),index_bytes,
                      filtered32.tobytes()))
    expected=60+36*len(selected)+56*len(target_world)+4*half.triangles.size+4*len(target_world)*len(selected)
    if len(payload)!=expected:
        raise PlaneCutError("emitted NHSKIN ABI byte extent is inconsistent")
    return payload, {
        "binding_indices_from_source":selected,
        "binding_body_indices":[int(source_bindings[i][0]) for i in selected],
        "vertex_count":int(len(target_world)),
        "triangle_count":int(len(half.triangles)),
        "cap_triangle_count":int(np.count_nonzero(half.triangle_kind==1)),
        "retained_boundary_edge_count":int(len(half.boundary_edges)),
        "rest_reconstruction_max_error_m":max_error,
        "full_weight_row_sum_max_error":float(np.max(np.abs(filtered32.astype(np.float64).sum(axis=1)-1))),
        "evidence_boundary":"Same-ABI visual NHSKIN half only; not collision, tissue physics, or demonstrated fracture.",
    }
from dataclasses import dataclass as _dataclass

# Owner-defined nonplanar interface cut. The caller supplies a pinned rig-partition
# scalar at each source vertex; no nearest-body assignment is inferred.
def split_surface_by_signed_field(vertices, triangles, signed_field, *,
                                  attributes: Mapping[str,np.ndarray] | None = None,
                                  allowed_existing_boundary_edges: set[tuple[int,int]] | None = None,
                                  tolerance=1.0e-11,
                                  max_projection_residual=None):
    """Split/cap a surface using linear interpolation of a vertex scalar zero contour.

    Caps are triangulated in a deterministic best-fit 2D projection while keeping
    the exact original 3D contour coordinates. Invalid topology fails closed.
    """
    points=np.asarray(vertices,dtype=np.float64)
    faces=np.asarray(triangles,dtype=np.int64)
    values=np.asarray(signed_field,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise PlaneCutError("vertices must be finite N x 3 coordinates")
    if values.shape!=(len(points),) or not np.isfinite(values).all():
        raise PlaneCutError("signed field must be finite with one value per source vertex")
    tol=float(tolerance)
    if not np.isfinite(tol) or tol<=0: raise PlaneCutError("tolerance must be positive")
    scale=max(float(np.linalg.norm(np.ptp(points,axis=0))),1.0)
    declared=set() if allowed_existing_boundary_edges is None else {_edge(int(a),int(b)) for a,b in allowed_existing_boundary_edges}
    boundary=_validate(points,faces,declared,tol)
    if any(a==b or a<0 or b>=len(points) for a,b in declared):
        raise PlaneCutError("declared boundary references invalid source vertex")
    attrs={}
    for name,value in (attributes or {}).items():
        arr=np.asarray(value)
        if arr.ndim<1 or arr.shape[0]!=len(points) or not np.isfinite(arr).all():
            raise PlaneCutError(f"attribute {name!r} must have finite rows for all vertices")
        attrs[name]=arr.astype(np.float64,copy=False)
    field_tol=tol*max(float(np.max(np.abs(values))),1.0)
    if np.any(np.abs(values)<=field_tol):
        raise PlaneCutError("signed field passes through source vertex; exact vertex cases require explicit topology handling")
    side=np.where(values>0,1,-1)
    if not (np.any(side>0) and np.any(side<0)):
        raise PlaneCutError("signed field does not separate source")
    for a,b in boundary:
        if side[a]!=side[b]: raise PlaneCutError("signed-field cut intersects a retained existing opening")
    allp=[p.copy() for p in points]
    alla={n:[r.copy() for r in arr] for n,arr in attrs.items()}
    prov=[{"kind":"source_vertex","source_vertex":int(i)} for i in range(len(points))]
    intersections={}
    def crossing(a,b):
        key=_edge(a,b)
        if key in intersections: return intersections[key]
        u,v=key; t=float(values[u]/(values[u]-values[v]))
        if not 0<t<1: raise PlaneCutError("signed-field edge crossing is outside source edge")
        allp.append(points[u]+t*(points[v]-points[u]))
        for n,arr in attrs.items(): alla[n].append(arr[u]+t*(arr[v]-arr[u]))
        prov.append({"kind":"source_edge","source_edge":[u,v],"t_from_first":t})
        intersections[key]=len(allp)-1
        return intersections[key]
    def clip(face,s):
        poly=[]; ids=list(map(int,face))
        for i,a in enumerate(ids):
            b=ids[(i+1)%3]; ia,ib=side[a]==s,side[b]==s
            if ia and ib: poly.append(b)
            elif ia and not ib: poly.append(crossing(a,b))
            elif not ia and ib: poly.extend((crossing(a,b),b))
        out=[]
        for x in poly:
            if not out or out[-1]!=x: out.append(x)
        if len(out)>1 and out[0]==out[-1]: out.pop()
        return out
    byside={1:[],-1:[]}; segments=[]
    for fi,face in enumerate(faces):
        signs=side[face]
        if np.any(signs>0) and np.any(signs<0):
            ids=[]
            for i in range(3):
                a,b=int(face[i]),int(face[(i+1)%3])
                if side[a]!=side[b]: ids.append(crossing(a,b))
            ids=list(dict.fromkeys(ids))
            if len(ids)!=2: raise PlaneCutError("signed-field crossed triangle does not produce one segment")
            segments.append(tuple(ids))
        for s in (1,-1):
            poly=clip(face,s)
            for k in range(1,len(poly)-1):
                tri=(poly[0],poly[k],poly[k+1])
                a,b,c=(np.asarray(allp[x]) for x in tri)
                if np.linalg.norm(np.cross(b-a,c-a))<=tol*max(np.linalg.norm(b-a),np.linalg.norm(c-a),1e-12):
                    raise PlaneCutError("signed-field clipping produced a degenerate triangle")
                byside[s].append((tri,fi))
    loops=_loops(segments)
    allpoints=np.asarray(allp)
    projections=[]; residuals=[]; projection_axes=[]; eps=max(tol*scale,1e-24)
    for loop in loops:
        poly3=allpoints[loop]; center=poly3.mean(axis=0)
        _,singular,vt=np.linalg.svd(poly3-center,full_matrices=False)
        if len(singular)<2 or singular[1]<=eps:
            raise PlaneCutError("nonplanar cap contour has insufficient projected area")
        e1,e2=vt[0],vt[1]
        projection_axes.append(np.cross(e1,e2))
        poly=np.column_stack(((poly3-center)@e1,(poly3-center)@e2))
        residual=float(np.max(np.abs((poly3-center)@vt[2])))
        if max_projection_residual is not None and residual>float(max_projection_residual):
            raise PlaneCutError(f"cap contour projection residual exceeds limit ({residual:.6g} m)")
        if len(loop)<3 or abs(_area(poly))<=eps:
            raise PlaneCutError("nonplanar cap contour has zero projected area")
        for i in range(len(loop)):
            for j in range(i+1,len(loop)):
                if j==(i+1)%len(loop) or i==(j+1)%len(loop): continue
                if _intersect(poly[i],poly[(i+1)%len(poly)],poly[j],poly[(j+1)%len(poly)],eps):
                    raise PlaneCutError("nonplanar cap contour self-intersects in projection")
        projections.append(poly);residuals.append(residual)
    for i in range(len(loops)):
        for j in range(i+1,len(loops)):
            pa,pb=projections[i],projections[j]
            if any(_intersect(pa[a],pa[(a+1)%len(pa)],pb[b],pb[(b+1)%len(pb)],eps)
                   for a in range(len(pa)) for b in range(len(pb))):
                raise PlaneCutError("nonplanar cap contours intersect or touch")
            if _inside(pa[0],pb,eps) or _inside(pb[0],pa,eps):
                raise PlaneCutError("nested nonplanar cap contours require unsupported hole triangulation")
    expected={1:set(),-1:set()}
    for a,b in boundary: expected[int(side[a])].add((a,b))
    def build(s):
        used=sorted({x for tri,_ in byside[s] for x in tri}|{x for e in expected[s] for x in e})
        if not used: raise PlaneCutError("one signed-field half has no triangles")
        remap={x:i for i,x in enumerate(used)}
        out=[];src=[];kinds=[];caps=[]
        for tri,fi in byside[s]:
            out.append(tuple(remap[x] for x in tri));src.append(fi);kinds.append(0);caps.append(-1)
        cutuses=_uses(np.asarray(out,dtype=np.int64).reshape((-1,3)))
        localpoints=np.asarray([allp[x] for x in used])
        for li,loop in enumerate(loops):
            ids=[remap[x] for x in loop]; captris=_earclip(ids,projections[li],eps)
            edge=_edge(ids[0],ids[1]); use=cutuses.get(edge)
            if use is None or len(use)!=1:
                raise PlaneCutError("cap contour edge is not a single clipped-surface boundary")
            sa,sb=use[0][1],use[0][2]; capdir=None
            for tri in captris:
                for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
                    if _edge(a,b)==edge: capdir=(a,b); break
                if capdir is not None: break
            if capdir is None: raise PlaneCutError("cap triangulation omitted contour edge")
            if capdir==(sa,sb): captris=[(a,c,b) for a,b,c in captris]
            _validate_projected_cap(captris,ids,projections[li],projection_axes[li],localpoints,eps,tol)
            for tri in captris:
                a,b,c=(localpoints[x] for x in tri)
                longest=max(float(np.linalg.norm(b-a)),float(np.linalg.norm(c-a)),float(np.linalg.norm(a-b)))
                if np.linalg.norm(np.cross(b-a,c-a))<=tol*max(longest,1e-12):
                    raise PlaneCutError("nonplanar cap contains a degenerate 3D triangle")
                out.append(tri);src.append(-1);kinds.append(1);caps.append(li)
        localprov=tuple(prov[x] for x in used)
        facesout=np.asarray(out,dtype=np.int64).reshape((-1,3))
        be=_output_boundary(facesout,localprov,expected[s])
        localattrs={n:np.asarray([alla[n][x] for x in used],dtype=np.float64) for n in attrs}
        return SurfaceHalf(np.asarray([allp[x] for x in used]),facesout,localattrs,localprov,
            np.asarray(src,dtype=np.int64),np.asarray(kinds,dtype=np.uint8),
            np.asarray(caps,dtype=np.int64),be,np.asarray(residuals,dtype=np.float64))
    return build(1),build(-1)


def _proper_cross2(a,b,c,d,eps):
    o1,o2,o3,o4=_cross2(a,b,c),_cross2(a,b,d),_cross2(c,d,a),_cross2(c,d,b)
    return ((o1>eps and o2 < -eps) or (o1 < -eps and o2>eps)) and ((o3>eps and o4 < -eps) or (o3 < -eps and o4>eps))


def _strict_inside_triangle2(p, tri, eps):
    a,b,c=tri
    v=(_cross2(a,b,p),_cross2(b,c,p),_cross2(c,a,p))
    return all(x>eps for x in v) or all(x < -eps for x in v)


def _validate_projected_cap(triangles, loop_ids, projected_loop, axis, points3d, eps, tol):
    coord={int(vertex):np.asarray(projected_loop[i],dtype=np.float64) for i,vertex in enumerate(loop_ids)}
    rows=[tuple(map(int,t)) for t in triangles]
    projected=[tuple(coord[x] for x in tri) for tri in rows]
    signs=[]
    for tri in rows:
        a,b,c=(points3d[x] for x in tri)
        n=np.cross(b-a,c-a); length=float(np.linalg.norm(n))
        if length<=tol*max(float(np.linalg.norm(b-a)),float(np.linalg.norm(c-a)),1e-12):
            raise PlaneCutError("nonplanar cap contains a degenerate 3D triangle")
        dot=float(np.dot(n,axis))
        if abs(dot)<=tol*length:
            raise PlaneCutError("nonplanar cap triangle is edge-on to its validated projection")
        signs.append(1 if dot>0 else -1)
    if len(set(signs))>1:
        raise PlaneCutError("nonplanar cap triangle normals fold across the best-fit projection")
    # Ear-clipping is expected to tile the simple projected polygon. Verify
    # pairwise interiors explicitly; disjoint orthogonal projections also
    # rule out intersections of the retained 3D triangles.
    for i in range(len(rows)):
        ai=set(rows[i]); ta=projected[i]
        for j in range(i+1,len(rows)):
            bj=set(rows[j]); common=ai&bj
            if len(common)>=2: continue
            tb=projected[j]
            for ea in range(3):
                a,b=ta[ea],ta[(ea+1)%3]
                for eb in range(3):
                    c,d=tb[eb],tb[(eb+1)%3]
                    if _proper_cross2(a,b,c,d,eps):
                        raise PlaneCutError("nonplanar cap triangles self-intersect in projection")
            for vertex in ai-common:
                if _strict_inside_triangle2(coord[vertex],tb,eps):
                    raise PlaneCutError("nonplanar cap triangle interior overlaps in projection")
            for vertex in bj-common:
                if _strict_inside_triangle2(coord[vertex],ta,eps):
                    raise PlaneCutError("nonplanar cap triangle interior overlaps in projection")

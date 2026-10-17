"""Exact local closed obstacle clipping for offline resting anatomy repairs.

Clipping caps are inferred authoring boundaries. They carry complete conservative
source-face support and must be placed outside the anatomical repair region.
This module does not change anatomy, physical mass, or simulation state.
"""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction
import numpy as np
from . import cardiac_face_arrangement as arrangement

def _require(ok, message):
    if not ok:
        raise ValueError(message)

def _normal(a, b, c):
    u = tuple(b[k]-a[k] for k in range(3))
    v = tuple(c[k]-a[k] for k in range(3))
    return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])

def _inside(point, polygon):
    """Exact odd-winding point classification; boundary is rejected."""
    x, y = point
    inside = False
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        cross = (b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])
        _require(not (cross == 0 and min(a[0],b[0]) <= x <= max(a[0],b[0])
                      and min(a[1],b[1]) <= y <= max(a[1],b[1])),
                 "cap sample is on a constraint")
        if (a[1] > y) != (b[1] > y):
            at_x = a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if x < at_x:
                inside = not inside
    return inside

def _cap_triangles(loops, points, axis, desired_sign):
    axes = [k for k in range(3) if k != axis]
    projected = [[tuple(points[i][k] for k in axes) for i in loop] for loop in loops]
    nested = any(_inside(poly[0], other) for i, poly in enumerate(projected)
                 for j, other in enumerate(projected) if i != j)
    lookup = {tuple(points[i][k] for k in axes): i for loop in loops for i in loop}
    out = []
    if not nested:
        for polygon in projected:
            if arrangement._area2(polygon) < 0:
                polygon = polygon[::-1]
            out.extend(tuple(lookup[x] for x in tri)
                       for tri in arrangement._triangulate_cell(polygon))
    else:
        # Reuse the existing exact constrained arrangement for nested loops.
        # Select only the odd-winding material cells, retaining every hole.
        coords = [x for polygon in projected for x in polygon]
        lo = tuple(min(p[k] for p in coords) for k in range(2))
        hi = tuple(max(p[k] for p in coords) for k in range(2))
        span = max(hi[0]-lo[0], hi[1]-lo[1])
        _require(span > 0, "zero cap extent")
        base2 = [(lo[0]-span,lo[1]-span),
                 (lo[0]+4*span,lo[1]-span),
                 (lo[0]-span,lo[1]+4*span)]
        plane = points[loops[0][0]][axis]
        def lift(x):
            q = [plane,plane,plane]
            for k, v in zip(axes,x):
                q[k] = v
            return tuple(q)
        segments = [(lift(a),lift(b)) for polygon in projected
                    for a,b in zip(polygon,polygon[1:]+polygon[:1])]
        triangles = arrangement.subdivide_triangle([lift(x) for x in base2],segments)
        for triangle in triangles:
            p2 = [tuple(p[k] for k in axes) for p in triangle]
            centroid = tuple(sum(p[k] for p in p2)/3 for k in range(2))
            if sum(_inside(centroid,polygon) for polygon in projected) % 2:
                _require(all(x in lookup for x in p2),
                         "cap arrangement introduced an unsupported vertex")
                out.append(tuple(lookup[x] for x in p2))
    oriented = []
    for tri in out:
        direction = _normal(*(points[i] for i in tri))[axis]
        _require(direction != 0, "zero cap triangle")
        oriented.append(tri if direction*desired_sign > 0 else tri[::-1])
    return oriented

def clip_closed_mesh_to_aabb(vertices, triangles, lower, upper):
    """Clip a closed oriented mesh to exact metre halfspaces.

    Returns the ordinary exact mesh keys, one complete parent-face support list
    per output face, and an inferred-clipping-cap mask. Input coordinates and
    box bounds are interpreted exactly. Nested cap loops preserve holes.
    """
    raw = np.asarray(vertices)
    faces = np.asarray(triangles)
    _require(raw.ndim == 2 and raw.shape[1] == 3 and raw.dtype.kind in "fiu" and np.isfinite(raw).all(),
             "expected finite XYZ coordinates")
    _require(faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype.kind in "iu"
             and np.all(faces >= 0) and np.all(faces < len(raw)),
             "invalid triangle indices")
    _require(len(lower) == len(upper) == 3, "box must have three axes")
    lower = tuple(Fraction(x) for x in lower)
    upper = tuple(Fraction(x) for x in upper)
    _require(all(a < b for a,b in zip(lower,upper)), "invalid box bounds")
    points = [tuple(Fraction.from_float(float(x)) for x in p) for p in raw]
    registry = {p:i for i,p in reversed(list(enumerate(points)))}
    current = [tuple(map(int,f)) for f in faces]
    # Verify source closure before any clipping plane can fabricate a cap over
    # an input defect, including an open boundary coincident with that plane.
    original_edges = defaultdict(list)
    for face in current:
        quotient = tuple(registry[points[i]] for i in face)
        _require(len(set(quotient)) == 3 and any(_normal(*(points[i] for i in face))),
                 "degenerate input triangle")
        for a,b in zip(quotient,quotient[1:]+quotient[:1]):
            original_edges[tuple(sorted((a,b)))].append((a,b))
    _require(all(len(uses) == 2 and uses[0] == uses[1][::-1]
                 for uses in original_edges.values()),
             "input has an unexplained open boundary or inconsistent orientation")
    support = [[i] for i in range(len(current))]
    caps = [False]*len(current)
    stages = []
    def slot(p):
        value = registry.get(p)
        if value is None:
            value = len(points)
            points.append(p)
            registry[p] = value
        return value
    for axis in range(3):
        for bound, keep_sign in ((lower[axis],1),(upper[axis],-1)):
            out = []
            parents = []
            capflags = []
            distance = [(p[axis]-bound)*keep_sign for p in points]
            for face, ancestry, cap in zip(current,support,caps):
                d = [distance[i] for i in face]
                if max(d) < 0:
                    continue
                if min(d) >= 0:
                    out.append(face);parents.append(ancestry);capflags.append(cap)
                    continue
                polygon = []
                for i,a in enumerate(face):
                    b = face[(i+1)%3]
                    da,db = d[i],d[(i+1)%3]
                    if da >= 0:
                        polygon.append(a)
                    if da*db < 0:
                        t = da/(da-db)
                        p = tuple(points[a][k]+t*(points[b][k]-points[a][k]) for k in range(3))
                        polygon.append(slot(p))
                polygon = list(dict.fromkeys(polygon))
                if len(polygon) < 3:
                    continue
                reference = _normal(*(points[i] for i in face))
                # A clipped source triangle is convex. The established exact
                # planar ear clipper preserves all collinear constraint points.
                drop = max(range(3),key=lambda k:abs(reference[k]))
                axes = [k for k in range(3) if k != drop]
                projected = [tuple(points[i][k] for k in axes) for i in polygon]
                by_point = dict(zip(projected,polygon))
                if arrangement._area2(projected) < 0:
                    projected = projected[::-1]
                for tri2 in arrangement._triangulate_cell(projected):
                    tri = tuple(by_point[x] for x in tri2)
                    n = _normal(*(points[i] for i in tri))
                    if sum(a*b for a,b in zip(n,reference)) < 0:
                        tri = tri[::-1]
                    out.append(tri);parents.append(ancestry);capflags.append(cap)
            _require(out, "clipped material is empty")
            edges = defaultdict(list)
            for fi,face in enumerate(out):
                for a,b in zip(face,face[1:]+face[:1]):
                    # Quotient by exact coordinates before boundary detection.
                    a,b = registry[points[a]],registry[points[b]]
                    _require(a != b, "zero exact edge")
                    edges[tuple(sorted((a,b)))].append((a,b,fi))
            boundary = []
            for uses in edges.values():
                _require(len(uses) in (1,2), "nonmanifold clipped edge")
                if len(uses) == 2:
                    _require(uses[0][:2] == uses[1][:2][::-1],
                             "inconsistent oriented clipped edge")
                else:
                    a,b,fi = uses[0]
                    _require(points[a][axis] == points[b][axis] == bound,
                             "input had an unexplained open boundary")
                    boundary.append((a,b,fi))
            outgoing = {}
            incoming = {}
            for a,b,fi in boundary:
                _require(a not in outgoing and b not in incoming,
                         "touching or branching cap boundary is unsupported")
                outgoing[a] = (b,fi);incoming[b] = a
            _require(set(outgoing) == set(incoming), "open cap chain")
            unused = set(outgoing)
            loops = []
            loop_support = []
            while unused:
                start = min(unused);p = start;loop = [];anc = set()
                while True:
                    _require(p in unused, "cap loop entered another loop")
                    unused.remove(p);loop.append(p)
                    p,fi = outgoing[p];anc.update(parents[fi])
                    if p == start:
                        break
                _require(len(loop) >= 3, "degenerate cap loop")
                loops.append(loop);loop_support.append(anc)
            if loops:
                # Conservatively bind nested cap cells to every source loop
                # delimiting this same plane, without inventing a unique parent.
                ancestry = sorted(set().union(*loop_support))
                for tri in _cap_triangles(loops,points,axis,-keep_sign):
                    out.append(tri);parents.append(ancestry);capflags.append(True)
            current,support,caps = out,parents,capflags
            stages.append({"axis":axis,"bound_m":str(bound),"keep_sign":keep_sign,
                           "face_count":len(current),"cap_loop_count":len(loops)})
    return {"vertices_m":points,"triangles":[list(f) for f in current],
            "output_face_parent_ids":support,"inferred_clipping_cap_mask":caps,
            "clipping_stages":stages}

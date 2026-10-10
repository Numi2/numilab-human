"""Exact bounded planar arrangements for authored cardiac surface triangles.

This offline geometry owner never steps physics, moves source vertices, applies
welding tolerances, or resolves physiological ownership. It inserts rational
intersection points and triangulates simple arrangement cells. Closed interior
loops use noncrossing diagonals between existing vertices. Closed loops meeting at cut vertices gain visible triangulation diagonals.
Dangling open cuts remain explicitly rejected.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from functools import cmp_to_key

from .model import ImportError as HumanImportError

Point = tuple[Fraction, Fraction, Fraction]
Point2 = tuple[Fraction, Fraction]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("cardiac face arrangement: " + message)


def _point(value) -> Point:
    require(isinstance(value, (list, tuple)) and len(value) == 3
            and all(type(x) is int or isinstance(x, Fraction) for x in value),
            "coordinates must be exact integer or Fraction triples")
    return tuple(Fraction(x) for x in value)


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])


def _cross2(a, b):
    return a[0]*b[1] - a[1]*b[0]


def _orient(a, b, c):
    return _cross2(_sub(b, a), _sub(c, a))


def _on_segment(point, a, b):
    return _orient(a, b, point) == 0 and all(min(x, y) <= z <= max(x, y) for z, x, y in zip(point, a, b))


def _inside_triangle(point, triangle):
    signs = [_orient(triangle[i], triangle[(i+1) % 3], point) for i in range(3)]
    return min(signs) >= 0 or max(signs) <= 0


def _segment_intersections(a, b, c, d):
    r, s = _sub(b, a), _sub(d, c)
    denominator = _cross2(r, s)
    if denominator:
        t, u = _cross2(_sub(c, a), s) / denominator, _cross2(_sub(c, a), r) / denominator
        if 0 <= t <= 1 and 0 <= u <= 1:
            return {tuple(a[k] + t*r[k] for k in range(2))}
        return set()
    if _orient(a, b, c):
        return set()
    return {point for point in (a, b, c, d) if _on_segment(point, a, b) and _on_segment(point, c, d)}


def _angular_compare(a, b):
    """Counterclockwise order without angles or approximate predicates."""
    half_a = 0 if a[1] > 0 or (a[1] == 0 and a[0] > 0) else 1
    half_b = 0 if b[1] > 0 or (b[1] == 0 and b[0] > 0) else 1
    if half_a != half_b:
        return -1 if half_a < half_b else 1
    turn = _cross2(a, b)
    if turn:
        return -1 if turn > 0 else 1
    # Arrangement subdivision must eliminate two neighbors on the same ray.
    require(a == b, "unsplit collinear half-edges")
    return 0


def _area2(polygon):
    return sum(_cross2(a, b) for a, b in zip(polygon, polygon[1:] + polygon[:1]))


def _triangulate_cell(polygon: list[Point2]) -> list[tuple[Point2, Point2, Point2]]:
    require(len(polygon) >= 3 and len(set(polygon)) == len(polygon) and _area2(polygon) > 0,
            "cell is not a simple positively oriented polygon")
    remaining = list(polygon)
    result = []
    while len(remaining) > 3:
        for i, current in enumerate(remaining):
            previous, following = remaining[i-1], remaining[(i+1) % len(remaining)]
            if _orient(previous, current, following) <= 0:
                continue
            ear = (previous, current, following)
            # Inclusive rejection preserves every collinear boundary vertex:
            # no triangulation diagonal may pass through an unconsumed vertex.
            if any(_inside_triangle(point, ear) for point in remaining if point not in ear):
                continue
            result.append(ear)
            del remaining[i]
            break
        else:
            require(False, "cell cannot be triangulated without losing constraint vertices")
    require(_orient(*remaining) > 0, "degenerate final cell triangle")
    result.append(tuple(remaining))
    require(sum(_orient(*tri) for tri in result) == _area2(polygon), "cell area was not conserved")
    return result


def _first_articulation(adjacency):
    """Find a cut vertex without recursion; all graph vertices are exact points."""
    root = min(adjacency)
    discovery = {root: 0}
    low = {root: 0}
    parent = {}
    children = Counter()
    cuts = set()
    stack = [(root, iter(sorted(adjacency[root])))]
    while stack:
        vertex, neighbors = stack[-1]
        neighbor = next(neighbors, None)
        if neighbor is None:
            stack.pop()
            if vertex in parent:
                above = parent[vertex]
                low[above] = min(low[above], low[vertex])
                if above in parent and low[vertex] >= discovery[above]:
                    cuts.add(above)
            elif children[vertex] > 1:
                cuts.add(vertex)
        elif neighbor not in discovery:
            parent[neighbor] = vertex
            children[vertex] += 1
            discovery[neighbor] = low[neighbor] = len(discovery)
            stack.append((neighbor, iter(sorted(adjacency[neighbor]))))
        elif parent.get(vertex) != neighbor:
            low[vertex] = min(low[vertex], discovery[neighbor])
    require(len(discovery) == len(adjacency), "cut-vertex scan requires a connected graph")
    return min(cuts) if cuts else None


def _bridge_touching_loops(adjacency, edges):
    """Add exact visible triangulation edges across cut vertices.

    No original point or cut is moved or removed. A convex source triangle
    contains every proposed segment; intersections with existing edges are
    permitted only at its own endpoints. Output incidence/area/T-junction
    checks remain authoritative.
    """
    for _ in range(len(adjacency)):
        cut = _first_articulation(adjacency)
        if cut is None:
            return
        components = {}
        for seed in sorted(set(adjacency) - {cut}):
            if seed in components:
                continue
            component_id = len(set(components.values()))
            pending = [seed]
            while pending:
                vertex = pending.pop()
                if vertex in components:
                    continue
                components[vertex] = component_id
                pending.extend(adjacency[vertex] - {cut} - components.keys())
        best = None
        points = sorted(components)
        for index, a in enumerate(points):
            for b in points[index+1:]:
                if components[a] == components[b]:
                    continue
                distance2 = sum((a[k]-b[k])**2 for k in range(2))
                candidate = (distance2, a, b)
                if best is not None and candidate >= best:
                    continue
                if all(_segment_intersections(a, b, c, d) <= {a, b} for c, d in edges):
                    best = candidate
        require(best is not None, "touching loops have no visible triangulation bridge")
        _, a, b = best
        edges.add(tuple(sorted((a, b))))
        adjacency[a].add(b)
        adjacency[b].add(a)
    require(_first_articulation(adjacency) is None, "touching-loop bridge bound exceeded")


def subdivide_triangle(triangle, segments, boundary_points=()) -> list[tuple[Point, Point, Point]]:
    """Split one exact 3D triangle at a bounded rational segment arrangement.

    ``segments`` are nonzero endpoint pairs contained in the triangle's plane
    and closed area. ``boundary_points`` propagate globally shared source-edge
    vertices, and must lie on its boundary. Returned nondegenerate triangles
    preserve source winding, every constraint edge, and exact oriented area.
    """
    require(isinstance(triangle, (list, tuple)) and len(triangle) == 3, "expected one source triangle")
    triangle3 = tuple(_point(x) for x in triangle)
    normal = _cross(_sub(triangle3[1], triangle3[0]), _sub(triangle3[2], triangle3[0]))
    require(any(normal), "degenerate source triangle")
    dropped = max(range(3), key=lambda k: abs(normal[k]))
    axes = tuple(k for k in range(3) if k != dropped)

    def project(point):
        p = _point(point)
        require(sum(normal[k]*(p[k] - triangle3[0][k]) for k in range(3)) == 0,
                "point is outside the source plane")
        return tuple(p[k] for k in axes)

    def lift(point):
        p = list(triangle3[0])
        for k, x in zip(axes, point):
            p[k] = x
        p[dropped] -= sum(normal[k]*(p[k] - triangle3[0][k]) for k in axes) / normal[dropped]
        return tuple(p)

    source = tuple(project(p) for p in triangle3)
    source_area = _orient(*source)
    require(source_area != 0, "singular projection")
    require(isinstance(segments, (list, tuple)) and len(segments) <= 512, "unsupported segment collection")
    constraints = [tuple(sorted((source[i], source[(i+1) % 3]))) for i in range(3)]
    for pair in segments:
        require(isinstance(pair, (list, tuple)) and len(pair) == 2, "invalid intersection segment")
        a, b = (project(p) for p in pair)
        require(a != b, "zero-length intersection segment")
        require(_inside_triangle(a, source) and _inside_triangle(b, source), "intersection segment leaves source triangle")
        constraints.append(tuple(sorted((a, b))))
    constraints = sorted(set(constraints))
    propagated = set()
    try:
        iterator = iter(boundary_points)
    except TypeError as error:
        raise HumanImportError("cardiac face arrangement: invalid boundary points") from error
    for count, p in enumerate(iterator):
        require(count < 4096, "too many propagated boundary points")
        point = project(p)
        require(any(_on_segment(point, source[i], source[(i+1) % 3]) for i in range(3)),
                "propagated point is not on the source boundary")
        propagated.add(point)
    split_points = [{a, b} | {p for p in propagated if _on_segment(p, a, b)} for a, b in constraints]
    for i, (a, b) in enumerate(constraints):
        for j in range(i+1, len(constraints)):
            points = _segment_intersections(a, b, *constraints[j])
            split_points[i].update(points)
            split_points[j].update(points)
    edges = set()
    for (a, b), points in zip(constraints, split_points):
        coordinate = 0 if a[0] != b[0] else 1
        ordered = sorted(points, key=lambda p: (p[coordinate] - a[coordinate]) / (b[coordinate] - a[coordinate]))
        edges.update(tuple(sorted((x, y))) for x, y in zip(ordered, ordered[1:]))
    adjacency = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b); adjacency[b].add(a)
    require(len(adjacency) <= 8192, "arrangement is too large")
    reached = set(); pending = [source[0]]
    while pending:
        point = pending.pop()
        if point not in reached:
            reached.add(point); pending.extend(adjacency[point] - reached)
    require(all(len(neighbors) >= 2 for neighbors in adjacency.values()), "dangling cut is unsupported")
    # Closed interior cut components can coexist with boundary-connected cuts.
    # Join each with two noncrossing visible diagonals between existing vertices.
    # Two distinct endpoints on each component avoid a dangling bridge/weak
    # polygon. These are triangulation edges only: no coordinate, source area,
    # or original constraint is changed. All ordinary incidence/area/T-junction
    # checks below still apply to the resulting planar graph.
    while reached != set(adjacency):
        component = set(); pending = [min(set(adjacency) - reached)]
        while pending:
            point = pending.pop()
            if point not in component:
                component.add(point); pending.extend(adjacency[point] - component)
        visible = []
        for a in sorted(component):
            for b in sorted(reached):
                if all(_segment_intersections(a, b, c, d) <= {a, b} for c, d in edges):
                    distance2 = sum((a[k] - b[k])**2 for k in range(2))
                    visible.append((distance2, a, b))
        visible.sort()
        bridges = None
        for i, (_, a, b) in enumerate(visible):
            for _, c, d in visible[i+1:]:
                if a != c and b != d and not _segment_intersections(a, b, c, d):
                    bridges = ((a, b), (c, d))
                    break
            if bridges is not None:
                break
        require(bridges is not None, "interior loop has no pair of visible noncrossing bridges")
        for a, b in bridges:
            edges.add(tuple(sorted((a, b))))
            adjacency[a].add(b); adjacency[b].add(a)
        reached.update(component)
    _bridge_touching_loops(adjacency, edges)
    ordered_neighbors = {}
    for point, neighbors in adjacency.items():
        def compare(a, b):
            return _angular_compare(_sub(a, point), _sub(b, point))
        ordered_neighbors[point] = sorted(neighbors, key=cmp_to_key(compare))
    unused = {(a, b) for a, b in edges} | {(b, a) for a, b in edges}
    bounded, exterior = [], []
    while unused:
        start = min(unused); current = start; polygon = []
        while True:
            require(current in unused, "half-edge traversal entered another face")
            unused.remove(current)
            a, b = current
            polygon.append(a)
            neighbors = ordered_neighbors[b]
            # The face to the left follows the immediately clockwise outgoing
            # half-edge from the incoming edge's reverse direction.
            current = (b, neighbors[(neighbors.index(a)-1) % len(neighbors)])
            if current == start:
                break
        require(len(polygon) >= 3 and len(set(polygon)) == len(polygon),
                "dangling bridge or nonsimple arrangement face is unsupported")
        area = _area2(polygon)
        require(area != 0, "zero-area arrangement face")
        (bounded if area > 0 else exterior).append(polygon)
    require(len(exterior) == 1 and -_area2(exterior[0]) == abs(source_area), "unsupported exterior or hole topology")
    require(sum(_area2(polygon) for polygon in bounded) == abs(source_area), "arrangement area was not conserved")
    result2 = [tri for polygon in bounded for tri in _triangulate_cell(polygon)]
    if source_area < 0:
        result2 = [(a, c, b) for a, b, c in result2]
    require(sum(_orient(*tri) for tri in result2) == source_area, "source oriented area was not conserved")
    emitted = Counter(tuple(sorted((a, b))) for tri in result2 for a, b in zip(tri, tri[1:] + tri[:1]))
    directed = Counter((a, b) for tri in result2 for a, b in zip(tri, tri[1:] + tri[:1]))
    require(edges <= set(emitted), "an arrangement constraint was lost")
    boundary = {edge for edge in emitted if any(_on_segment(edge[0], source[i], source[(i+1) % 3])
                and _on_segment(edge[1], source[i], source[(i+1) % 3]) for i in range(3))}
    require(all(count == (1 if edge in boundary else 2) for edge, count in emitted.items()), "invalid output edge incidence")
    require(all(edge in boundary or (directed[edge] == 1 and directed[tuple(reversed(edge))] == 1)
                for edge in emitted), "inconsistent output edge orientation")
    used_vertices = {point for tri in result2 for point in tri}
    require(used_vertices == set(adjacency), "constraint vertex was omitted")
    for a, b in emitted:
        require(not any(p != a and p != b and _on_segment(p, a, b) for p in used_vertices), "output contains a T-junction")
    return [tuple(lift(point) for point in tri) for tri in result2]

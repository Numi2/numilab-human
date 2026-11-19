"""Accelerate retained Float32 audits without changing exact pair semantics."""
from functools import lru_cache

from numilab_human import cardiac_cavity_intersections as owner


def signed_distances(triangle, normal, points):
    x,y,z=triangle[0]
    nx,ny,nz=normal
    return tuple(nx*(a-x)+ny*(b-y)+nz*(c-z) for a,b,c in points)


def one_closed_halfspace(distances, expected_zeros):
    nonzero = tuple(d for d in distances if d)
    return (len(nonzero) == len(distances)-expected_zeros and bool(nonzero)
            and (min(nonzero)>0 or max(nonzero)<0))


def exact_integer_pairs(candidate_pairs, *, same_surface):
    """Use integer signs for easy pairs and the existing exact narrow phase."""
    @lru_cache(maxsize=32768)
    def normal(tri):
        n=owner._cross(owner._sub(tri[1],tri[0]),owner._sub(tri[2],tri[0]))
        owner.require(any(n), "degenerate triangle")
        return n

    visits, allowed, intersects = 0, 0, []
    for a,b in candidate_pairs:
        visits += 1
        tri, _, _, i, ids = a
        other, _, _, j, other_ids = b
        na,nb=normal(tri),normal(other)
        shared=set(ids).intersection(other_ids) if same_surface else set()
        common={tri[ids.index(index)] for index in shared}
        local = len(shared) in (1,2) and all(other[other_ids.index(k)]==tri[ids.index(k)] for k in shared)
        if (not shared or local) and (
            one_closed_halfspace(signed_distances(tri,na,other),len(shared))
            or one_closed_halfspace(signed_distances(other,nb,tri),len(shared))
        ):
            # A triangle meeting the plane at exactly the shared indexed
            # endpoints can intersect only the point or edge they span.
            allowed += bool(shared)
            continue
        points=owner._triangle_intersection_points_integer(tri, other)
        if points:
            if local and all(owner._allowed_shared_point(p,common) for p in points):
                allowed+=1
            else:
                intersects.append([i,j])
    return dict(triangle_pairs=sorted(intersects),count=len(intersects),
                aabb_candidate_pairs=visits,allowed_shared_vertex_or_edge_pairs=allowed)


def checked_integer_pairs(candidate_pairs, *, same_surface):
    # The integer-lattice proof requires exact Python ints. Leave callers
    # using rational records on the original path.
    for a,b in candidate_pairs:
        if any(type(c) is not int for r in (a,b) for point in r[0] for c in point):
            raise TypeError("expected emitted Float32 integer-lattice records")
        yield a,b


def install_for_float32():
    def audit(candidate_pairs, *, same_surface):
        return exact_integer_pairs(checked_integer_pairs(candidate_pairs,same_surface=same_surface),
                                   same_surface=same_surface)
    owner._audit_record_pairs=audit

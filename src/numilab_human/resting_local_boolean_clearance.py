"""Bounded clearance of new Boolean cap vertices during anatomy authoring."""
from __future__ import annotations
from fractions import Fraction
import math
import numpy as np
from .resting_intercostal_mesh_arrangement import _round_fraction_to_f32

def offset_new_boolean_cap_vertices(vertices_m, triangles, output_face_ancestry, *,
                                    source_face_count, obstacle_vertices_m,
                                    obstacle_triangles, inferred_obstacle_cap_mask,
                                    gap_m=1/524288, maximum_offset_m=5e-5):
    """Round once after offsetting cap births along their target-normal cone.

    This constructs an offline candidate. Full emitted triangle, solid, self,
    provenance and motion gates remain mandatory. Artificial clipping caps must
    stay outside the repair region and are rejected if they own output material.
    """
    if (type(source_face_count) is not int or source_face_count < 1
            or not math.isfinite(gap_m) or not 0 < gap_m <= maximum_offset_m
            or not math.isfinite(maximum_offset_m)):
        raise ValueError("invalid bounded cap clearance configuration")
    points = [tuple(Fraction(x) for x in p) for p in vertices_m]
    faces = np.asarray(triangles)
    obs = [tuple(Fraction(x) for x in p) for p in obstacle_vertices_m]
    ofaces = np.asarray(obstacle_triangles)
    flags = np.asarray(inferred_obstacle_cap_mask)
    if (faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
            or np.any(faces < 0) or np.any(faces >= len(points))
            or len(output_face_ancestry) != len(faces)
            or ofaces.ndim != 2 or ofaces.shape[1] != 3
            or ofaces.dtype.kind not in "iu" or np.any(ofaces < 0)
            or np.any(ofaces >= len(obs))
            or flags.dtype != np.dtype("bool") or flags.shape != (len(ofaces),)):
        raise ValueError("malformed Boolean output or obstacle face contract")
    normals = {}
    incident = {}
    for face, ancestry in zip(faces,output_face_ancestry):
        if not isinstance(ancestry,dict) or type(ancestry.get("source_face")) is not int:
            raise ValueError("Boolean output face lacks exact input face ancestry")
        source = ancestry["source_face"]
        if source < 0 or source >= source_face_count+len(ofaces):
            raise ValueError("Boolean output source face is out of range")
        if source < source_face_count:
            continue
        negative = source-source_face_count
        if flags[negative]:
            raise ValueError("artificial clipping cap entered anatomical output")
        if negative not in normals:
            tri = np.asarray([[float(x) for x in obs[int(i)]] for i in ofaces[negative]])
            normal = np.cross(tri[1]-tri[0],tri[2]-tri[0])
            size = float(np.linalg.norm(normal))
            if size == 0 or not math.isfinite(size):
                raise ValueError("degenerate anatomical obstacle face")
            normals[negative] = normal/size
        for vertex in face:
            incident.setdefault(int(vertex),set()).add(negative)
    rounded = np.asarray([[_round_fraction_to_f32(x) for x in p] for p in points],dtype="<f4")
    operations = []
    for vertex,neighbors in sorted(incident.items()):
        normal_array = np.asarray([normals[i] for i in sorted(neighbors)])
        direction = normal_array.sum(axis=0)
        size = float(np.linalg.norm(direction))
        if size == 0:
            raise ValueError("opposed target normals require another clearance construction")
        direction /= size
        projection = normal_array@direction
        minimum = float(projection.min())
        if minimum <= 0:
            raise ValueError("target normal cone has no positive average direction")
        displacement = direction*(gap_m/minimum)
        norm = float(np.linalg.norm(displacement))
        if norm > maximum_offset_m:
            raise ValueError("cap clearance exceeds bounded local displacement")
        point = tuple(points[vertex][k]+Fraction.from_float(float(displacement[k]))
                      for k in range(3))
        rounded[vertex] = [_round_fraction_to_f32(x) for x in point]
        actual = rounded[vertex].astype(float)-np.asarray([float(x) for x in points[vertex]])
        actual_minimum = float((normal_array@actual).min())
        if actual_minimum < gap_m*.5:
            raise ValueError("Float32 cap emission lost the declared clearance margin")
        operations.append({"vertex":vertex,"obstacle_face_ids":sorted(neighbors),
                           "inferred_displacement_m":displacement.tolist(),
                           "maximum_offset_m":maximum_offset_m,
                           "minimum_emitted_target_normal_projection_m":actual_minimum})
    return rounded, {"gap_m":gap_m,"maximum_offset_m":maximum_offset_m,
                     "operations":operations,"offset_vertex_count":len(operations),
                     "owner_basis":"offline_target_normal_candidate_requires_exact_geometry_gates",
                     "native_admission":False,"anatomical_admission":False}

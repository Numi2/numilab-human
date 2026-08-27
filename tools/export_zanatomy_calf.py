"""Export the narrow Z-Anatomy right-calf visual supplement from Blender.

Run only inside Blender, for example:

  blender --background Startup.blend --python tools/export_zanatomy_calf.py -- calf.json
  blender --background Startup.blend --python tools/export_zanatomy_calf.py -- \
      calf.json --tendon-subdivision-level 1 --tendon-insertion-depth-mm 8

The output is an offline import interchange.  It contains no simulation
parameters and is consumed by the Human importer before the Python-free native
Core/Metal inspection command runs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


OBJECTS = (
    ("lateral_gastrocnemius", "Lateral head of gastrocnemius.r", "muscle"),
    ("medial_gastrocnemius", "Medial head of gastrocnemius.r", "muscle"),
    ("soleus", "Soleus muscle.r", "muscle"),
    ("calcaneal_tendon", "Calcaneal tendon.r", "tendon"),
    # The narrowly scoped heel overlay is rigidly bound to the existing
    # MyoSim calcn_r body. It keeps the detailed free tendon and its authored
    # calcaneal insertion in one matching source-surface pair; BodyParts3D
    # remains the geometry authority everywhere else.
    ("calcaneus_overlay", "Calcaneus.r", "bone"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not arguments:
        raise RuntimeError("expected an output JSON path after --")
    output = Path(arguments[0]).resolve()
    tendon_subdivision_level = 0
    tendon_insertion_depth_mm = 0.0
    option_arguments = arguments[1:]
    if len(option_arguments) % 2:
        raise RuntimeError("each export option requires a value")
    for option, value in zip(option_arguments[::2], option_arguments[1::2]):
        if option == "--tendon-subdivision-level":
            try:
                tendon_subdivision_level = int(value)
            except ValueError as error:
                raise RuntimeError("tendon subdivision level must be an integer") from error
            if not 0 <= tendon_subdivision_level <= 2:
                raise RuntimeError("tendon subdivision level must be between 0 and 2")
        elif option == "--tendon-insertion-depth-mm":
            try:
                tendon_insertion_depth_mm = float(value)
            except ValueError as error:
                raise RuntimeError("tendon insertion depth must be numeric") from error
            if not 0.0 <= tendon_insertion_depth_mm <= 12.0:
                raise RuntimeError("tendon insertion depth must be between 0 and 12 mm")
        else:
            raise RuntimeError(f"unsupported export option: {option}")
    blend = Path(bpy.data.filepath).resolve()
    if not blend.is_file():
        raise RuntimeError("the saved Z-Anatomy Startup.blend input is required")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    records = []
    for stable_name, object_name, layer in OBJECTS:
        source = bpy.data.objects.get(object_name)
        if source is None or source.type != "MESH":
            raise RuntimeError(f"required Z-Anatomy mesh is absent: {object_name}")
        subdivision = None
        if stable_name == "calcaneal_tendon" and tendon_subdivision_level:
            # The atlas tendon has a visibly faceted closed terminal cap. This
            # is a deterministic derivative of the same licensed source mesh,
            # applied only to the visual supplement before it enters the
            # native C++/Metal path; it never changes MyoSim mechanics.
            subdivision = source.modifiers.new("NumiLab tendon smooth derivative", "SUBSURF")
            subdivision.subdivision_type = "CATMULL_CLARK"
            subdivision.levels = tendon_subdivision_level
            subdivision.render_levels = tendon_subdivision_level
            subdivision.boundary_smooth = "PRESERVE_CORNERS"
            bpy.context.view_layer.update()
        evaluated = source.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            if not mesh.vertices or not mesh.polygons:
                raise RuntimeError(f"Z-Anatomy mesh is empty: {object_name}")
            vertices = [list(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices]
            if stable_name == "calcaneal_tendon" and tendon_insertion_depth_mm:
                # The source's closed terminal cap is visibly serrated.  Its
                # distal 33 mm is not mechanics geometry: make a smooth,
                # source-local visual insertion by carrying the terminal
                # taper 8 mm (or the requested bounded depth) inside the
                # matching calcaneus.  The overlap hides the artificial cap
                # without inventing a visible bridge or changing any MyoSim
                # body, path, tendon, or force parameter.
                distal_z = min(vertex[2] for vertex in vertices)
                transition_z = distal_z + 0.033
                fully_embedded_z = distal_z + 0.010
                denominator = transition_z - fully_embedded_z
                for vertex in vertices:
                    fraction = min(1.0, max(0.0, (transition_z - vertex[2]) / denominator))
                    smooth_fraction = fraction * fraction * (3.0 - 2.0 * fraction)
                    vertex[1] -= 0.001 * tendon_insertion_depth_mm * smooth_fraction
            triangles = []
            for polygon in mesh.polygons:
                if len(polygon.vertices) < 3:
                    raise RuntimeError(f"Z-Anatomy mesh has a degenerate face: {object_name}")
                first = polygon.vertices[0]
                for index in range(1, len(polygon.vertices) - 1):
                    triangles.append([first, polygon.vertices[index], polygon.vertices[index + 1]])
            normals = [Vector((0.0, 0.0, 0.0)) for _ in vertices]
            # Calculate normals from the evaluated world-space triangles.
            # A few source vertices are adjacent only to degenerate Blender
            # faces, so the stored vertex normal can be zero there.
            for first, second, third in triangles:
                a, b, c = (Vector(vertices[index]) for index in (first, second, third))
                face_normal = (b - a).cross(c - a)
                normals[first] += face_normal
                normals[second] += face_normal
                normals[third] += face_normal
            for index, normal in enumerate(normals):
                if normal.length <= 1.0e-12:
                    center = sum((Vector(point) for point in vertices), Vector((0.0, 0.0, 0.0))) / len(vertices)
                    normal = Vector(vertices[index]) - center
                if normal.length <= 1.0e-12:
                    raise RuntimeError(f"Z-Anatomy mesh has no usable normal: {object_name}")
                normals[index] = list(normal.normalized())
            records.append({
                "id": stable_name,
                "object": object_name,
                "layer": layer,
                "vertex_count": len(vertices),
                "triangle_count": len(triangles),
                "vertices_world_m": vertices,
                "normals_world": normals,
                "triangles": triangles,
            })
        finally:
            evaluated.to_mesh_clear()
            if subdivision is not None:
                source.modifiers.remove(subdivision)
                bpy.context.view_layer.update()
    payload = {
        "schema": "numi.human.zanatomy-calf-blender-export.v1",
        "source": {
            "blend_file": blend.name,
            "blend_sha256": sha256(blend),
            "coordinate_system": "Blender evaluated object world metres",
            "calcaneal_tendon_subdivision_level": tendon_subdivision_level,
            "calcaneal_tendon_insertion_depth_mm": tendon_insertion_depth_mm,
        },
        "objects": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

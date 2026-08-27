"""Export the narrow Z-Anatomy right-calf visual supplement from Blender.

Run only inside Blender, for example:

  blender --background Startup.blend --python tools/export_zanatomy_calf.py -- calf.json

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
    # This is a rest-frame registration landmark only.  It is never emitted
    # into the Human tissue payload because the articulated BodyParts3D bone
    # remains the visual skeletal authority.
    ("calcaneus_landmark", "Calcaneus.r", "landmark"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(arguments) != 1:
        raise RuntimeError("expected one output JSON path after --")
    output = Path(arguments[0]).resolve()
    blend = Path(bpy.data.filepath).resolve()
    if not blend.is_file():
        raise RuntimeError("the saved Z-Anatomy Startup.blend input is required")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    records = []
    for stable_name, object_name, layer in OBJECTS:
        source = bpy.data.objects.get(object_name)
        if source is None or source.type != "MESH":
            raise RuntimeError(f"required Z-Anatomy mesh is absent: {object_name}")
        evaluated = source.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            if not mesh.vertices or not mesh.polygons:
                raise RuntimeError(f"Z-Anatomy mesh is empty: {object_name}")
            vertices = [list(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices]
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
    payload = {
        "schema": "numi.human.zanatomy-calf-blender-export.v1",
        "source": {
            "blend_file": blend.name,
            "blend_sha256": sha256(blend),
            "coordinate_system": "Blender evaluated object world metres",
        },
        "objects": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

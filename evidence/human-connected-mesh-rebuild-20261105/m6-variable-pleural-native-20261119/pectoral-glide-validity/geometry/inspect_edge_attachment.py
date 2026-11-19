"""Trace which authored skin/tissue rows block the exact support edge."""
import json
from pathlib import Path

text = Path(__file__).with_name("screen_fixed_edge_tissue_lp.py").read_text()
exec(compile(text[:text.index("orientation = vstack")], __file__, "exec"))
skin_ids = np.asarray(sorted(edge_vertices))
e = posed_skin[np.searchsorted(skin_rows, skin_ids)].astype(float)
face = faces[target[1]]
r = posed[face].astype(float)
n = np.cross(r[1]-r[0],r[2]-r[0])
n /= np.linalg.norm(n)
area_normals = area(posed_skin[emitted_faces[edge_faces]].astype(float))
skin_n = area_normals.sum(axis=0)
skin_n -= (skin_n @ (e[1]-e[0])) * (e[1]-e[0])/np.sum((e[1]-e[0])**2)
skin_n /= np.linalg.norm(skin_n)
body_slots=set(int(i)+1 for i in skin_ids)
with (prior / "native/resting-local-bed-sections.csv").open() as f:
    from csv import DictReader
    support = [row for row in DictReader(f) if int(row["source_geometry_index"]) in body_slots]
seam = np.zeros(len(points), dtype=bool)
seam[clear._exact_coordinate_tied_source_vertex_ids(points.astype("<f4"))] = True
border = np.zeros(len(points), dtype=bool)
border[edges[incidence != 2].ravel()] = True
near = posed[faces].astype(float)
radius = .00015
lo, hi = near.min(1), near.max(1)
selected = (np.all(hi+radius>=e.min(0), axis=1) &
            np.all(lo-radius<=e.max(0), axis=1))
rows=np.unique(faces[selected])
rows=rows[(local[rows]<0) & (((posed[rows]-e[0])@skin_n)>-.00006)]
receipt = dict(
    target=list(target), skin_edge=skin_ids.tolist(),
    end_to_end_length_mm=float(np.linalg.norm(e[1]-e[0])*1000),
    skin_signed_distance_to_witness_triangle_plane_mm=((e-r[0])@n*1000).tolist(),
    triangle_distance_to_skin_norm_plane_mm=((r-e[0])@skin_n*1000).tolist(),
    support_rows=support, local_fixed_tissue_rows=[dict(
        row=int(i), source_seam=bool(seam[i]), source_boundary=bool(border[i]),
        source_end_region=bool(axial[i]<=q[0] or axial[i]>=q[1]),
        outside_linearization=bool(not active[i]),
        distance_above_skin_plane_mm=float((posed[i]-e[0])@skin_n*1000)
    ) for i in rows],
)
Path("seat-edge-attachment-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))

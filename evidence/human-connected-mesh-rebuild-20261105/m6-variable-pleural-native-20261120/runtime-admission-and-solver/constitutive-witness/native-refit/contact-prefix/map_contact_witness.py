"""Map the limiting vertex/triangle to existing fat-layer volume and provenance."""
import csv
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np
from scipy.optimize import brentq

root = Path(sys.argv[1])
q = root/"refit-motion-qualification"
witness = json.loads((q/"trial-022-contact-proof.json").read_text())["contact"]
layout = json.loads((q/"trial-021-limiter-summary.json").read_text())["object_layout"]
left,right,kind,_ = witness["identity"]
assert left==right and kind==0
first = layout[left]["first_node"]
ids = [i-first+1 for i in witness["nodes"][:4]]
p, *face = ids
rows = {}
with (q/"registration/combined-node-map.csv").open() as f:
    for row in csv.DictReader(f):
        i=int(row["joint_node_id"])
        if i in ids:
            rows[i] = row
            if len(rows)==4:
                break
assert len(rows)==4
assert all(row["registered_surface_layer"]=="D" and row["source_partition"]=="fat"
           for row in rows.values())
nearby=set()
face_rows=[]
vertex_ring_count=0
with (q/"registration/facet-ancestry-current.csv").open() as f:
    reader=csv.reader(f)
    columns = next(reader)
    indices = [columns.index(f"joint_node{i}") for i in range(3)]
    for record in reader:
        tri={int(record[i]) for i in indices}
        if p in tri:
            nearby.update(tri-{p})
            vertex_ring_count+=1
        if tri==set(face):
            face_rows.append(dict(zip(columns,record,strict=True)))
assert len(face_rows)==1
neighbors_in_triangle = sorted(nearby.intersection(face))
incident=Counter()
with (root/"constitutive-witness-repaired-tetgen/mesh.ele").open() as f:
    next(f)
    for line in f:
        fields=line.split()
        if len(fields)<5:
            continue
        here=tuple(sorted(set(map(int,fields[1:5])).intersection(ids)))
        if here:
            incident[here]+=1
def stored(which):
    return np.array(witness[f"potential_feature_{which}_high"])[:,:3]+np.array(
        witness[f"potential_feature_{which}_low"])[:,:3]
def point_plane(positions):
    point,*triangle=positions
    v,w,x=triangle
    u,k=w-v,x-v
    normal=np.cross(u,k)
    normal/=np.linalg.norm(normal)
    bary=np.linalg.lstsq(np.stack((u,k),axis=1),point-v,rcond=None)[0]
    return dict(signed_plane_m=float((point-v)@normal),
        projected_triangle_barycentric=[1.-sum(bary),*bary])
a,b=stored("start"),stored("end")
def at(t):
    return point_plane(a+(b-a)*t)
margin=witness["numerical_clearance_offset_m"]
cut=brentq(lambda t: abs(at(t)["signed_plane_m"])-margin,0.,1.)
rest=np.array([[float(rows[i][f"current_{axis}_m"]) for axis in "xyz"] for i in ids])
samples={f"{f:.12g}":at(f) for f in (0.,witness["lane_bound_after"],1./32,cut,1.)}
assert all(min(s["projected_triangle_barycentric"])>0 for s in samples.values())
output=dict(node_ids=ids,authority="registered fat deep boundary D",
            rows=[rows[i] for i in ids],face_ancestry=face_rows[0],
            neighbors_in_triangle=neighbors_in_triangle,
            source_vertex_surface_one_ring_faces=vertex_ring_count,
            nearby_tet_incidence=[dict(shared_witness_nodes=list(key),count=count)
                                  for key,count in sorted(incident.items())],
            stored_rest_measurement=point_plane(rest),sweep_samples=samples,
            diagnostic_margin_crossing_fraction=cut,
            rest_vertex_to_nearest_face_corner_m=float(np.min(np.linalg.norm(rest[0]-rest[1:],axis=1))))
print(json.dumps(output,indent=2))

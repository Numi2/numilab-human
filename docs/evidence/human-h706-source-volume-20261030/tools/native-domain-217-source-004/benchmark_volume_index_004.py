from pathlib import Path
import csv,json,sys,time,resource,hashlib
import numpy as np
import build_native_domain_217 as owner
from prior_face_incidence import prior_face_incidence
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
C=P/"native-physical-parent-mass-330/native-source-identity-control-003"
files=[C/"compiled-parent-cell-mass.csv.nodes.csv",C/"compiled-parent-cell-mass.csv.tets.csv"]
with files[0].open() as f:
    rows=list(csv.DictReader(f))
points=np.asarray([[float(r[k]) for k in ("current_x_m","current_y_m","current_z_m")] for r in rows],np.float32)
with files[1].open() as f:rows=list(csv.DictReader(f))
tets=np.asarray([[int(r[k]) for k in ("parent_node0","parent_node1","parent_node2","parent_node3")] for r in rows],np.int64)
materials=np.asarray([int(r["material_index"]) for r in rows],np.int32)
mesh=dict(points=points,tetrahedra=tets,element_ids=np.arange(len(tets)),material_indices=materials)
del rows
start=time.perf_counter()
index=prior_face_incidence(mesh) if sys.argv[1]=="prior" else owner._tetrahedron_boundary_faces(mesh)
build=time.perf_counter()-start
keys={k for k,v in index.items() if len(v)==1} if sys.argv[1]=="prior" else index.single_occurrence_keys()
start2=time.perf_counter();h=hashlib.sha256()
for key in sorted(keys):
    h.update(json.dumps([key,index[key]],separators=(",",":")).encode())
end=time.perf_counter()
d=dict(mode=sys.argv[1],source_nodes=len(points),source_tetrahedra=len(tets),all_unique_faces=len(index),
       boundary_faces=len(keys),build_seconds=build,query_seconds=end-start2,
       total_index_and_boundary_query_seconds=end-start,output_sha256=h.hexdigest(),
       peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
       physical_admission=False,native_runtime=0,
       limitations=["Single asset-preparation measurement on immutable source330; not a full-volume or GPU runtime benchmark."])
print(json.dumps(d))

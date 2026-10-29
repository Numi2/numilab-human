from pathlib import Path
import csv,json,numpy as np,collections,importlib.util,hashlib
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
f=P/"P/native-detail-bridge-426/effective-tet-weight-witness-009/all-O-effective-tet-weights.csv"
with f.open() as h:r=list(csv.DictReader(h))
bad=[x for x in r if int(x["exceeds_declared_1p5_bound"])]
cells=np.array(sorted({int(x["canonical_selected_parent_tet_row"]) for x in bad}),np.int64)
z=np.load(P/"reduced-physical-boundary-400/attempt-002-parent-ancestry-walk/candidate.npz")
t=z["volume_tetrahedra"];nodes=np.unique(t[cells])
inventory=json.loads((P/"connected-cavity-witness-inventory-448/receipt.json").read_text())
allnodes=set(map(int,nodes))
for x in inventory["bank_witnesses"]:allnodes.update(x["boundary_node_rows"])
spec=importlib.util.spec_from_file_location("m",P/"inventory_connected_cavity_witnesses_448.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
groups=m.components(np.array(sorted(allnodes)),t)
out={"scope":"offline merged repair demand from frozen 007 embedding witnesses and exact 448 bone occupancy; source002 topology; not geometry or physical admission",
 "support_witness_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"failed_detail_rows":len(bad),"unique_selected_cells":len(cells),
 "selected_cell_rows":cells.tolist(),"selected_node_rows":nodes.tolist(),
 "per_region_failed_rows":dict(sorted(collections.Counter(int(x["tet_region_id"]) for x in bad).items())),
 "merged_vertex_star_cavities":[{"index":i,"volume_cell_rows":v,"volume_node_count":len(np.unique(t[v]))} for i,v in enumerate(groups)],
 "physical_admission":False}
d=P/"merged-physical-refinement-demand-454";d.mkdir(exist_ok=False);(d/"receipt.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"failed_rows":len(bad),"cells":len(cells),"nodes":len(nodes),"merged_cavity_cells":[len(v) for v in groups]}))

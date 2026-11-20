import json
from pathlib import Path
import numpy as np
from paired_route_projection import PairedRouteProjection
from reflow_paired_mass import csr
w=Path.cwd()
audit=json.loads((w/"full-paired-route-audit.json").read_text())
m=np.load(w.parent/"resting-clearance-and-atrium-014/fine-current-paired-semantic-cycle-mass.npz")
report=json.loads((w/"exact-p1-paired-route-reflow.json").read_text())
amap=np.full(len(m["group_for_cell"]),-1,np.int32)
group_masks=[]
offset=0
for role,pin in audit["role_reports"].items():
    with np.load(w/pin["region_bounds_file"]) as mask:
        amap[mask["child_rows"]]=mask["child_mask_index"]+offset
        group_masks.extend(mask["group_and_regions"])
        offset+=len(mask["group_and_regions"])
        region_ids=mask["region_ids"]
projection=PairedRouteProjection(m,audit,amap)
label=Path(json.loads((w.parent/"resting-clearance-and-atrium-013/fine-local-current-p1.json").read_text())["parent_inputs"]["source330_labels"]["path"])
with label.open() as stream:
    header=stream.readline().strip().split(",")
    table=np.loadtxt(stream,delimiter=",",dtype=int)
regions=table[:,header.index("anatomical_region_id")][m["source_parent"]]
assert np.array_equal(regions,projection.tuples[:,2]), np.count_nonzero(regions != projection.tuples[:,2])
route=m["mass_flow_kg"]
source=m["carrier_row"]
quota=np.bincount(source,weights=route)
inspect=sorted(report["unfilled_recipient_domains"],key=lambda x:-x["short_kg"])[:10]
points=m["element_mass_kg"]
result=[]
for trial in inspect:
    g=trial["group"]
    cells=projection.cells_of(g)
    given=projection.direct_support(g)
    mask=group_masks[g]
    out=[]
    for cell,typed in zip(cells,given):
        position=projection.centers[cell]
        cols=projection.tuples
        candidates=np.zeros(len(cols),dtype=bool)
        for key in typed:
            candidates|=np.all(cols==key,axis=1)
        v=np.flatnonzero(candidates)
        d=np.linalg.norm(position-projection.supplier_positions[v],axis=1)
        out.append({"cell":int(cell),"exact_types":[list(k) for k in typed],
                    "source_count":len(v),"nearest_m":float(d.min()) if len(v) else None,
                    "carrier_quota_within_limit_kg":float(quota[v[d<=projection.limit]].sum()),
                    "cell_mass_kg":float(points[cell])})
    result.append({"group":g,"cells":len(cells),"exact_supported_cells":sum(row["carrier_quota_within_limit_kg"]>0 for row in out),
                   "exact_supported_cell_mass_kg":sum(row["cell_mass_kg"] for row in out if row["carrier_quota_within_limit_kg"]>0),
                   "cells_detail":out[:4]})
(w/"paired-region-typed-cut-witness.json").write_text(json.dumps(result,indent=2)+"\n")
for r in result:
    print(r["group"],r["cells"],r["exact_supported_cells"],r["cells_detail"][0])

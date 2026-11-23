"""Transfer resolvable bed-normal material lengths from contiguous same-owner tetrahedra."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from run_bed_work import digest


def fit(paths,case,out):
    out.mkdir()
    witness=json.loads((paths/"connected-columns.json").read_text())
    donors=json.loads((case/"skin-donors.json").read_text())
    if (digest(paths/"paths.npz")!=witness["paths"] or digest(case/"registered-bed-columns.csv")!=witness["reference_columns"] or
        digest(case/"skin-donors.npz")!=donors["donors"]):raise ValueError("Material or donor identity drift")
    p=np.load(paths/"paths.npz"); w=np.load(case/"skin-donors.npz")
    rows=p["vertices"];h=p["heights"];area=p["area"]; skin=h[:,1]-h[:,0];fat=h[:,2]-h[:,1]
    reached=p["result"][:,0]==0
    old_skin,old_fat=p["old_skin"],p["old_fat"]
    # Independently registered surfaces must describe the same local tissue neighborhood.
    compatible=reached&(p["outer_owner"]==w["owner"][rows])&(w["fraction"][rows]>=.5)
    compatible&=np.abs(p["source_xyz"][:,2]-h[:,0])<=np.minimum(old_skin+old_fat,skin+fat)
    cosine=np.min(p["vertical_normal_cosine"],axis=1)
    sensitivity={}
    for deg in (30,45,60):
        keep=compatible&(cosine>=np.cos(np.deg2rad(deg)))
        sensitivity[deg]=dict(columns=int(keep.sum()),area_m2=float(area[keep].sum()),
                              projected_skin_volume_delta_m3=float(area[keep]@(skin-old_skin)[keep]),
                              projected_fat_volume_delta_m3=float(area[keep]@(fat-old_fat)[keep]))
    # Limit the one-dimensional approximation to faces with vertical projection >= 1/sqrt(2).
    keep=compatible&(cosine>=np.sqrt(.5))
    mapped={int(v):i for i,v in enumerate(rows) if keep[i]}
    if len(mapped)!=int(keep.sum()):raise ValueError("Duplicate output material row")
    def emit(source,target,subset=False):
        with source.open() as f,target.open("w") as g:
            read=csv.DictReader(f);writer=csv.DictWriter(g,fieldnames=read.fieldnames,lineterminator="\n");writer.writeheader()
            for r in read:
                v=int(r["vertex"])
                if v in mapped:
                    i=mapped[v]
                    if (int(r["row"])!=p["rows"][i] or abs(float(r["z_m"])-p["source_xyz"][i,2])>5e-7):
                        raise ValueError("Changed reference point")
                    r.update(skin_m=format(skin[i],".17g"),fat_m=format(fat[i],".17g"))
                    if "thickness_source_vertex" in r:r["thickness_source_vertex"]="4294967295"
                elif subset:continue
                writer.writerow(r)
    full=case/"registered-bed-columns.csv"
    emit(full,out/"registered-stock.csv",True);emit(full,out/"registered-bed-columns.csv")
    emit(case/"resting-local-bed-footprint.csv",out/"resting-local-bed-footprint.csv")
    geom=case/"resting-local-bed-sections.csv"
    # The table compiler reads section area and identity only. Compliance is regenerated from the updated material columns.
    (out/geom.name).write_bytes(geom.read_bytes())
    for name in ("launch.json","resting-surface-audit.csv"):
        (out/name).write_bytes((case/name).read_bytes())
    i=mapped.get(17764)
    if i is None:raise ValueError("Shoulder correspondence is unqualified")
    report=dict(source=witness["paths"],source_mesh=witness["source_identities"]["TETGEN_ELEMENTS"],
        installed_connectivity=witness["source_identities"].get("INSTALLED_TETRAHEDRA"),
        material_reference_pose=witness["datum_pose"],skin_bindings=donors["influences"],row_count=len(mapped),
        seed=17764,seed_skin_m=float(skin[i]),seed_fat_m=float(fat[i]),seed_cells=int(p["result"][i,1]),
        seed_source_outer_interface_deep_faces=[int(p[k][i]) for k in ("outer_face","interface_face","deep_face")],
        sensitivity=sensitivity,retained_prior_columns=int(len(rows)-len(mapped)),total_projected_area_m2=float(area.sum()),
        stock=digest(out/"registered-stock.csv"),
        scope="Normal thickness assigned over connected registered cells. Existing anatomical, inertial and force owners stay in place. Other columns retain their original reported nearest-row construction.")
    (out/"assignment.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":fit(*(Path(s).resolve() for s in sys.argv[1:]))

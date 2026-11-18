"""Size the atlas gaps requested by the existing native physiology owner."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from inspect_cardiac import TRIAL, sha

def main():
    here = Path(__file__).resolve().parent
    launch=json.loads((TRIAL/"launch.json").read_text())
    argv=launch["argv"]
    receipt=Path(argv[argv.index("--resting-anatomy-receipt")+1])
    field=json.loads(receipt.read_text())["provenance"]["cardiac_geometry_binding"]["common_field"]
    def read(key):
        spec=field[key]
        file=receipt.parent/spec["path"]
        assert sha(file)==spec["sha256"]
        return np.fromfile(file,"<f4").astype(float)
    coeff=read("polynomials").reshape(7,120)
    boxes=read("domain_boxes").reshape(-1,2,8)[:,:,:7]
    degrees=np.asarray([e for e in itertools.product(range(4),repeat=7) if sum(e)<=3])
    powers=np.maximum(degrees[None,:,:]-np.eye(7,dtype=int)[:,None,:],0)
    def monom(q,d):
        return np.prod(np.power(q,d),axis=-1)
    time=[]
    results=[]
    residuals=[]
    q=np.zeros(7)
    with (here/"physiology-dynamic-8000.csv").open() as source:
        for row in csv.DictReader(source):
            target=[float(row[k])*1e-6 for k in ("right_atrium_ml","rv_ml","left_atrium_ml","lv_ml")]
            target=np.asarray(target+field["material_target_volumes_m3"])/field["source_reference_volumes_m3"]
            for _ in range(12):
                r=coeff @ monom(q,degrees)-target
                if abs(r).max() < 1e-9:
                    break
                jac=coeff @ (monom(q,powers)*degrees.T).T
                q -= np.linalg.solve(jac,r)
            time.append(float(row["time_s"]))
            results.append(q.copy())
            residuals.append(float(abs(r).max()))
    ts=np.array(time)
    trajectory=np.array(results)
    delta=np.maximum(np.maximum(boxes[:,0]-trajectory[:,None,:],trajectory[:,None,:]-boxes[:,1]),0)
    distance=np.max(delta,axis=2)
    nearest=np.argmin(distance,axis=1)
    misses=distance[np.arange(len(ts)),nearest]
    exposed=misses>1e-7
    worst=np.argmax(misses)
    report={"samples":len(ts),"last_s":float(ts[-1]),"samples_outside_retained_domains":int(exposed.sum()),
            "max_residual":max(residuals),"q_min":trajectory.min(axis=0).tolist(),
            "q_max":trajectory.max(axis=0).tolist(),"nearest_box_at_max_gap":int(nearest[worst]),
            "max_gap":float(misses[worst]),"max_gap_s":float(ts[worst]),
            "worst_case_q":trajectory[worst].tolist(),
            "worst_case_gap_by_coordinate":delta[worst,nearest[worst]].tolist()}
    if exposed.any():
        report["outside_box_frequency"]={str(k):int((nearest[exposed]==k).sum()) for k in np.unique(nearest[exposed])}
    print(json.dumps(report,indent=2))
    (here/"cardiac-range.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(here/"cardiac-range.npz",trajectory=trajectory,time=ts,
                        nearest=nearest,miss=misses)
if __name__=="__main__":
    main()

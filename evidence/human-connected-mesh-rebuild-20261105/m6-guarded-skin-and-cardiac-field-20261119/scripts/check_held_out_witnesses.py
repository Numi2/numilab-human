"""Classify retained singleton witness candidates via the exact owner predicates."""
import csv
from collections import Counter
import json
from pathlib import Path

import numpy as np

from correct_cardiac_field_soft import geometry,position
from trim_skin_witnesses import exact,positioned_records

def main():
    _,rows,faces,windows=geometry(Path("cardiac-held-out.bin"))
    with open("cardiac-held-out-witnesses.csv") as f:
        witnesses=list(csv.DictReader(f))
    with open("cardiac-held-out.log") as f:
        declared={int(w):int(area)+int(cross)+int(one)+int(two)+int(duplicate)
                  for w,_pairs,area,cross,one,two,duplicate in (line.split() for line in f)}
    owners=np.load(Path("../resting-clearance-and-atrium-003/common-cardiac-face-owners.npz"))["owner"]
    out=[]
    for i,(lo,hi) in enumerate(windows):
        assert np.array_equal(lo,hi)
        pairs=np.asarray([(int(x["first_face"]),int(x["second_face"]))
                     for x in witnesses if int(x["window"])==i and int(x["second_face"])>=0],
                     dtype=int).reshape(-1,2)
        ids=np.unique(pairs)
        hit_count=0
        by_owner=Counter()
        if len(ids):
            xyz=(position(rows,lo)*.001).astype("<f4")
            records={r[3]:r for r in positioned_records(xyz,faces,ids)}
            check=exact._audit_record_pairs(((records[a],records[b]) for a,b in pairs),same_surface=True)
            hit_count=check["count"]
            by_owner.update("/".join(map(str,sorted((owners[a],owners[b]))))
                            for a,b in check["triangle_pairs"])
        report=dict(index=i,actual_crossings_detected=hit_count,window_witnesses_declared=declared[i],
                    inspected_witness_pairs=len(pairs),output_cap_truncated=len(pairs)<declared[i],
                    owner_pairs=dict(sorted(by_owner.items())))
        out.append(report)
        print(report,flush=True)
    assert any(x["actual_crossings_detected"]>0 for x in out)
    Path("cardiac-held-out-exact.json").write_text(json.dumps(out,indent=2)+"\n")

if __name__=="__main__":
    main()

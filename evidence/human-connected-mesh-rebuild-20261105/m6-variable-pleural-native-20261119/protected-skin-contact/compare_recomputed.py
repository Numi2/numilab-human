"""Compare native accepted geometry and skin/fat support in matching resting invocations."""
import csv
import json

import numpy as np

from run_support_dependency_preflight import HERE, PREVIOUS, RUN, read, write, digest
from fit_recooked_skin import dependencies
from fit_clearance_patches import decode
from compare_local_clearance import report

candidate=HERE/"recomputed-support-native"
frozen=PREVIOUS/"source-clearance-patches.nhskin"
p,faces,_,old,pinned,_=dependencies(decode(frozen.read_bytes()))
a=RUN/"step-8000-skin-exact.json"
b=candidate/"step-8000-skin-exact.json"
A,S0,control=report(a,faces,old)
B,S1,test=report(b,faces,old)
assert control["accepted_step"]==test["accepted_step"]==8000 and A.keys()==B.keys()
assert len(p)==61950 and pinned.sum()==2637
C=read(candidate/"curves.json")
c0=read(RUN/"geometry-comparison.json")["candidate"]
assert c0["pack_sha256"]==control["pack_sha256"]
control.update({k:c0[k] for k in ("tissue_curve_length_m","tissue_curve_components","self_curve_length_m","self_curve_components")})
test.update(tissue_curve_length_m=C["tissue_curve_length_m"],
            tissue_curve_components=sum(v["connected_curve_components"] for v in C["tissue"].values()),
            self_curve_length_m=C["self_crossing"]["length_m"],self_curve_components=C["self_crossing"]["connected_curve_components"])
retained=[(k,len(A[k]-B[k]),len(B[k]-A[k])) for k in sorted(A)]
summary=dict(
    control=control,candidate=test,
    tissue_pair_witnesses_added=sum(add for _,_,add in retained),
    tissue_pair_witnesses_removed=sum(remove for _,remove,_ in retained),
    self_pair_witnesses_added=len(S1-S0),self_pair_witnesses_removed=len(S0-S1),
    fully_protected_tissue_witnesses_removed=sum(old[faces[i]].all() for k in A for i,_ in A[k]-B[k]),
    fully_protected_tissue_witnesses_added=sum(old[faces[i]].all() for k in B for i,_ in B[k]-A[k]),
    cleared_banks=[k for k in sorted(A) if A[k] and not B[k]],
    target_bank_witnesses=[dict(bank=k,control_pairs=len(A[k]),candidate_pairs=len(B[k])) for k in ((51005,119),(51004,89))],
    banks=[dict(bank=k,removed=rm,added=ad) for k,rm,ad in retained if rm or ad],
    total_exact_tissue_pair_delta=test["tissue_pairs"]-control["tissue_pairs"],
    total_exact_tissue_curve_delta_m=test["tissue_curve_length_m"]-control["tissue_curve_length_m"],
    qualification="terminal captured state only, active 32-section support, no full shared skin/fat tissue owner")
# Integer counts produced by array comparisons must use JSON-compatible int.
for k in ("fully_protected_tissue_witnesses_added","fully_protected_tissue_witnesses_removed"):
    summary[k]=int(summary[k])
names=("resting-coupled.csv","resting-local-bed-sections.csv","resting-com-support-impulses.csv","resting-com-momentum-diagnostic.csv")
summary["repeatable_matched_assets_but_changed_physical_trace"]={
    n:digest(RUN/"native"/n)!=digest(candidate/"native"/n) for n in names}
support_receipt=read(HERE/"native-recomputed-sections.json")
assert all(v["areas_changed"]==2 for v in support_receipt.values())
assert all(v["maximum_fractional_delta"]<0.001 for v in support_receipt.values())
assert test["tissue_pairs"]<control["tissue_pairs"]
assert test["tissue_curve_length_m"]<control["tissue_curve_length_m"]
write(candidate/"geometry-comparison.json",summary)
print(json.dumps({k:v for k,v in summary.items() if k!="banks"},indent=2),flush=True)

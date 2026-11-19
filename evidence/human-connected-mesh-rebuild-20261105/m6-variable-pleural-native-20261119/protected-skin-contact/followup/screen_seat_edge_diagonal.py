"""Check a local topology change at the edge between two real bed witnesses."""
import numpy as np
from fit_recooked_skin import SKIN, NATIVE, RUN, decode, area, Pose, SELF
from fit_thickness_dependency_repair import capture_in_memory, owner
from run_support_dependency_preflight import HERE, read, write, digest

owner.capture=capture_in_memory
path=HERE/"recomputed-support-skin.nhskin"
skin=decode(path.read_bytes())
p, faces=skin["verts"][:,:3],skin["inds"]
edge=(6889,7240)
selected=np.flatnonzero(np.isin(faces,edge).sum(axis=1)==2)
assert len(selected)==2 and 14040 in selected
one,two=faces[selected]
a,b=next((int(one[j]),int(one[(j+1)%3])) for j in range(3)
         if set((one[j],one[(j+1)%3]))==set(edge))
assert any(two[j]==b and two[(j+1)%3]==a for j in range(3))
c=int(next(j for j in one if j not in edge))
d=int(next(j for j in two if j not in edge))
assert c!=d and not np.any(np.isin(faces,(c,d)).sum(axis=1)==2)
proposed=faces.copy()
proposed[selected]=((c,d,b),(d,c,a))
key=(51005,121)
report = dict(source_sha256=digest(path), faces=selected.tolist(), fixed_edge=edge,
              before=faces[selected].tolist(), after=proposed[selected].tolist(), checks=[])

def check(pose, label):
    before_normals=area(pose.skin[faces[selected]].astype(float))
    after_normals=area(pose.skin[proposed[selected]].astype(float))
    average=before_normals.sum(axis=0)
    relative=(after_normals @ average)/np.dot(average,average)
    before=pose.targets_for_patch(selected,pose.skin[faces[selected]])
    before_curve=pose.curve_stats(key,before[key],pose.skin)[0] if before[key] else 0.
    pose.faces=proposed
    after=pose.targets_for_patch(selected,pose.skin[proposed[selected]])
    report["checks"].append(dict(
        pose=label, normal_projection=relative.tolist(), self_before=len(before[SELF]),self_after=len(after[SELF]),
        target_before=len(before[key]),target_after=len(after[key]),
        target_curve_before=before_curve,
        target_curve_after=pose.curve_stats(key,after[key],pose.skin)[0] if after[key] else 0.,
        other_regressions={str(k):len(rows-before[k]) for k,rows in after.items() if k!=key and rows-before[k]},
    ))

check(Pose.source_pose(p,faces),"source")
original=decode(SKIN.read_bytes())
for step, run in ((2015,RUN),(8000,HERE/"recomputed-support-native")):
    data=read(run/f"step-{step}-skin-exact.json")
    witnesses={(r["semantic"],r["stable_id"]):set(map(tuple,r["triangle_pairs"]))
               for r in data["surfaces"] if r["count"]}
    pose=Pose(run/f"native/accepted-geometry/step-{step}.mrvpack",skin,faces,witnesses)
    assert pose.sha==data["accepted_pack_sha256"]
    if step==2015:
        add=p.astype(float)-original["verts"][:,:3]
        pose.skin=(pose.skin.astype(float)+np.einsum("vij,vj->vi",pose.j,add)).astype("<f4")
    check(pose,step)
    del pose
report["archival_status"] = "screened local topology proposal; no native integration or clearance claim"
write(HERE/"seat-edge-diagonal-check-v2.json",report)

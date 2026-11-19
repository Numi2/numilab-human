"""Keep only source edits that add no exact witnesses on the retained poses."""
import argparse
import json

import numpy as np

from fit_connected_skin_offset import SOURCE, HERE, PREVIOUS, PRIOR, BASE, capture
from fit_connected_skin_offset import decode, axes, Pose, projected_guard, hash_file
from fit_connected_skin_offset import positioned_records, audit_source, exact
from fast_skin_intersection_audit import install_for_float32


def compare(report, variant):
    control = {(r["semantic"],r["stable_id"]): {tuple(p) for p in r["triangle_pairs"]}
               for r in report["surfaces"]}
    trials = {(r["semantic"],r["stable_id"]): {tuple(p) for p in r["triangle_pairs"]}
              for r in variant["surfaces"]}
    assert control.keys() == trials.keys()
    return [(key, *pair) for key, pairs in trials.items() for pair in sorted(pairs-control[key])]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--oriented", action="store_true")
    cli.add_argument("--fast-audit",action="store_true")
    args = cli.parse_args()
    if args.fast_audit:
        install_for_float32()
    source = HERE.parent/"resting-clearance-and-atrium-002/skin-rebound.nhskin"
    model, guarded = (decode(p.read_bytes()) for p in (source,SOURCE))
    xyz, faces = model["verts"][:,:3], model["inds"]
    assert np.array_equal(faces,guarded["inds"])
    assert np.array_equal(model["weights"],guarded["weights"])
    original = guarded["verts"][:,:3].astype(float)-xyz.astype(float)
    baseline_run = HERE/"cardiac-split-native-8000"
    guard_run = HERE/"cardiac-split-native-8000-guarded"

    def report(run):
        return json.loads((HERE/(run.name+"-skin-exact.json")).read_text())

    protected = np.load(PREVIOUS/"bed-section-pins.npz")["vertices"]
    assert not np.any(original[protected])
    new = compare(report(baseline_run),report(guard_run))
    assert len(new)==5
    original[np.unique(faces[[r[1] for r in new]])] = 0
    proposal = original.copy()
    if args.oriented:
        broad = np.load(HERE/"skin-connected-oriented.npz")["displacement_m"]
        proposal += .45*broad
        proposal[protected] = 0
    active = np.any(proposal != 0,axis=1)
    def case(run, step):
        return run/"native/accepted-geometry"/f"step-{step}.mrvpack"
    states = [
        (BASE, json.loads((HERE.parent/"resting-surface-clearance-001/initial.json").read_text())),
        (case(PRIOR/"wider-release-variable-4",2000), json.loads((PRIOR/"wider-release-variable-4-exact.json").read_text())),
        (case(baseline_run,8000),report(baseline_run)),
    ]
    poses=[]
    for path, data in states:
        record, current, topology = capture(path)
        assert np.array_equal(faces,topology)
        jacobian = np.einsum("vb,bdc->vdc",model["weights"],axes(model,record)[0])
        proposal_pose = (current + np.einsum("vij,vj->vi",jacobian,proposal)).astype("<f4")
        poses.append(Pose(path,data,proposal_pose,active))

    def rows_discarding(bad):
        return np.unique(faces[np.fromiter(bad,dtype=int)].ravel())

    tree = exact._prepare_surface_aabb(positioned_records(xyz,faces,np.arange(len(faces))))
    recent_source = set()
    history = []
    for iteration in range(24 if args.oriented else 8):
        invalid=set.union(*(projected_guard(p,active) for p in poses))
        while invalid:
            # Restoring a failing triangle can expose an inversion on its
            # neighbors. Finish the cheap area guard before broad queries.
            retire=rows_discarding(invalid)
            assert active[retire].any()
            active[retire]=False
            invalid=set.union(*(projected_guard(p,active) for p in poses))
        improved=[]
        for pose in poses:
            result,bad = pose.audit(active,check_self=False)
            invalid.update(bad)
            improved.append(result)
        if not invalid:
            improved=[]
            for pose in poses:
                result,bad = pose.audit(active,check_self=True)
                invalid.update(bad)
                improved.append(result)
            pts=(xyz+np.where(active[:,None],proposal,0.)).astype("<f4")
            invalid_source,_ = audit_source(xyz,pts,faces,active,tree,recent_source)
            invalid.update(invalid_source)
        entry=dict(iteration=iteration,active_source_rows=int(active.sum()),
                   poses=improved,invalid_triangles=len(invalid))
        print(entry,flush=True)
        history.append(entry)
        if not invalid:
            if not active.any() or not all(r["added"]==0 and r["self_added"]==0 for r in improved):
                raise RuntimeError("empty or invalid skin proposal")
            if improved[-1]["removed"]==0:
                raise RuntimeError("the proposal did not improve the late pose")
            stem="trimmed-oriented-pressure-skin" if args.oriented else "trimmed-guarded-pressure-skin"
            np.savez_compressed(HERE/(stem+".npz"),displacement_m=np.where(active[:,None],proposal,0.))
            (HERE/(stem+".json")).write_text(json.dumps(history,indent=2)+"\n")
            return
        retire=rows_discarding(invalid)
        assert active[retire].any()
        active[retire]=False
    (HERE/"trimmed-oriented-pressure-skin-failed.json").write_text(json.dumps(history,indent=2)+"\n")
    raise RuntimeError("candidate still creates new exact witnesses")


if __name__=="__main__":
    main()

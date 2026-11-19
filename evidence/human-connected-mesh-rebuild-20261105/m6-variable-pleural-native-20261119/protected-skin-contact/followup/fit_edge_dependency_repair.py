"""Screen source repairs near guarded skin using the native bed dependencies."""
import csv
from collections import Counter
from contextlib import contextmanager
import hashlib
import mmap
import subprocess
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.sparse import coo_matrix, csgraph, diags
from scipy.sparse.linalg import spsolve

from run_support_dependency_preflight import HERE, PREVIOUS, RUN, digest, read, write
sys.path.insert(0,str(PREVIOUS))
import fit_monotone_contour_patches as owner
from fit_monotone_contour_patches import Pose
from fit_clearance_patches import SELF, area, clear, decode, mesh_edges, oriented

SKIN=PREVIOUS/"source-clearance-patches.nhskin"
NATIVE=HERE/"bed-dependency-control/native"
TABLE=Path("/Users/numi/Work/skin-fat-native-integration-209/apps/NumiHumanWholeBodySkinReduction.hpp")
LIMIT_M=.0035


def dependencies(skin):
    p=skin["verts"][:,:3]
    f=skin["inds"]
    old=PREVIOUS.parent/"resting-clearance-and-atrium-004/bed-section-pins.npz"
    with np.load(old,allow_pickle=False) as z:
        old_guards=z["vertices"]
    e,n=mesh_edges(f)
    static=np.zeros(len(p),bool)
    static[e[n!=2].ravel()]=True
    static[clear._exact_coordinate_tied_source_vertex_ids(p)]=True
    legacy=static.copy()
    legacy[old_guards]=True
    assert legacy.sum()==9588
    csv_path=NATIVE/"resting-bed-section-footprint.csv"
    d=np.genfromtxt(csv_path,names=True,delimiter=",")
    assert np.array_equal(d["vertex"],np.arange(len(p)))
    assert np.isfinite(np.stack([d[k] for k in ("x","y","z","region_min","tributary_area")])).all()
    sections=NATIVE/"resting-local-bed-sections.csv"
    control=RUN/"native/resting-local-bed-sections.csv"
    assert digest(sections)==digest(control)
    with sections.open() as s:
        anchor=np.array([int(row["source_geometry_index"]) for row in csv.DictReader(s)])
    assert len(set(map(int,anchor)))==len(anchor)==32
    pinned=static.copy()
    pinned[anchor]=True
    pinned[anchor-1]=True
    sample_rows=TABLE.read_text().split("kWholeBodySkinSourceVertices={",1)[1].split("};",1)[0]
    samples=sorted({int(i) for i in re.findall(r"\b(\d+)(?:u)?(?=\s*[,}])",sample_rows+',')
                    if int(i)<len(p)})
    assert len(samples)>1000
    active=d["tributary_area"]>0.
    owners=d["nearest_thickness_vertex"][active].astype(int)
    assert set(owners)<=set(samples)
    pinned[samples]=True
    # In this trial the native owner recomputes bed projected areas from
    # output positions. The source attachment rows stay fixed.
    return p,f,e,legacy,pinned,dict(
        footprint_sha256=digest(csv_path),sections_sha256=digest(sections),
        old_pins_sha256=digest(old),source_table_sha256=digest(TABLE),
        old_protected_vertices=int(legacy.sum()),new_protected_vertices=int(pinned.sum()),
        exact_contributors=int(active.sum()),source_thickness_samples=len(samples),
        old_guards_eligible_for_screen=int((legacy&~pinned).sum()),maximum_displacement_m=LIMIT_M
    )


@contextmanager
def capture_in_memory(path):
    metadata = read(path.with_suffix(".receipt.json"))
    recovery = read(Path(str(path)+".zst.recovery.json"))
    archive = Path(str(path)+".zst")
    with mmap.mmap(-1, recovery["bytes"]) as mapping, subprocess.Popen(
        ["/opt/homebrew/bin/zstd", "-dc", str(archive)], stdout=subprocess.PIPE
    ) as proc:
        with memoryview(mapping) as buf:
            length = proc.stdout.readinto(buf)
            assert length == len(buf) and proc.stdout.read(1) == b""
            assert proc.wait()==0 and hashlib.sha256(buf).hexdigest()==metadata["pack_file_sha256"]
        yield mapping, {r["body_index"]:r for r in metadata["accepted_body_poses"]}, metadata


def main():
    owner.capture = capture_in_memory
    original=decode(SKIN.read_bytes())
    source=decode((HERE/"recomputed-support-skin.nhskin").read_bytes())
    p0,f,edges,old,pinned,proof=dependencies(original)
    original_pins=pinned.copy()
    release=read(HERE/"native-index-edge-dependency-repair.json")["selected_thickness_positions"]
    assert len(release)==26
    pinned[release]=False
    p=source["verts"][:,:3]
    assert np.array_equal(source["inds"],f)
    prior_delta=p.astype(float)-p0
    proof["skin_sha256"]=digest(SKIN)
    near_old=np.all(original_pins[f],axis=1)
    poses=[]
    blocked=Counter()
    exposed=Counter()
    for step in (2015,8000):
        run=RUN if step==2015 else HERE/"recomputed-support-native"
        capture=run/f"native/accepted-geometry/step-{step}.mrvpack"
        report=read(run/f"step-{step}-skin-exact.json")
        witnesses={(r["semantic"],r["stable_id"]):set(map(tuple,r["triangle_pairs"]))
                   for r in report["surfaces"] if r["count"]}
        bounded={k:{(a,b) for a,b in rows if near_old[a]} for k,rows in witnesses.items()}
        available={k:{(a,b) for a,b in rows if not pinned[f[a]].all()}
                   for k,rows in bounded.items()}
        if step==8000:
            for k in bounded:
                blocked[k[0]]+=len(bounded[k])
                exposed[k[0]]+=len(available[k])
        pose=Pose(capture,source,f,witnesses)
        assert pose.sha==report["accepted_pack_sha256"]
        if step==2015:
            pose.skin=(pose.skin.astype(float)+np.einsum("vij,vj->vi",pose.j,prior_delta)).astype("<f4")
            pose.init=pose.skin.copy()
        poses.append((pose,available))
    candidates=sorted({k for _,r in poses for k,pairs in r.items() if pairs},
                      key=lambda k:(-sum(len(a.get(k,())) for _,a in poses),k))
    proof["late_previously_fully_protected_pairs"]=dict(blocked)
    proof["late_previously_fully_protected_pairs_with_proposal_coordinates"]=dict(exposed)
    proof["pose_pack_sha256"]=[pose.sha for pose,_ in poses]
    write(HERE/"edge-native-dependency-screen.json",proof)
    print(proof,flush=True)
    print("target banks",len(candidates),flush=True)

    dirs=np.zeros_like(p,dtype=float)
    for axis in range(3):
        np.add.at(dirs,f[:,axis],area(p[f].astype(float)))
    dirs/=np.maximum(np.linalg.norm(dirs,axis=1)[:,None],1.e-20)
    dirs+=source["verts"][:,3:6]
    dirs/=np.maximum(np.linalg.norm(dirs,axis=1)[:,None],1.e-20)
    a,b=edges.T
    weight=1./np.maximum(np.linalg.norm(p[a]-p[b],axis=1)*1000.,.1)
    L=csgraph.laplacian(coo_matrix((np.r_[weight,weight],(np.r_[a,b],np.r_[b,a])),
                     shape=(len(p),len(p))).tocsr()).tocsr()
    source_pose=Pose.source_pose(p,f)
    delta=np.zeros_like(p,dtype=float)
    results=[]
    rejected=[]
    for index,key in enumerate(candidates):
        touched=np.zeros(len(p),bool)
        for _,rows in poses:
            target=rows.get(key,set())
            if target:
                touched[f[[i for i,_ in target]]]=True
        touched[pinned]=False
        region=touched.copy()
        for _ in range(3):
            region[edges[region[edges].any(axis=1)].ravel()]=True
        region[pinned]=False
        faces=np.flatnonzero(region[f].any(axis=1))
        ids=np.flatnonzero(region)
        req=np.zeros_like(p,dtype=float)
        for pose,rows in poses:
            for v,shift in pose.witness_support(key,rows.get(key,set()),region,dirs):
                shift*=min(1.,LIMIT_M/max(np.linalg.norm(shift),1.e-15))
                if np.linalg.norm(shift)>np.linalg.norm(req[v]):
                    req[v]=shift
        if not np.any(req):
            rejected.append(dict(bank=key, cause="no_outward_separator"))
            continue
        weights=15.*np.any(req[ids]!=0,axis=1)
        shift=spsolve(L[ids][:,ids]+diags(weights+.35),weights[:,None]*req[ids])
        assert np.isfinite(shift).all()
        prev=source_pose.targets_for_patch(faces,source_pose.skin[f[faces]])[SELF]
        old_sets=[pose.targets_for_patch(faces,pose.skin[f[faces]]) for pose,_ in poses]
        if not any(x[key] for x in old_sets):
            continue
        volumes=[pose.curve_stats(key,hits[key],pose.skin)
                 for (pose,_),hits in zip(poses,old_sets)]
        failures=Counter()
        for scale in (1.2,.8,.45,.25,.12):
            proposal=delta.copy()
            proposal[ids]+=scale*shift
            if np.max(np.linalg.norm(proposal[ids],axis=1))>=LIMIT_M:
                failures["displacement"]+=1
                continue
            src=(p.astype(float)+proposal).astype("<f4")
            if not oriented(p[f[faces]],src[f[faces]]):
                failures["source_orientation"]+=1
                continue
            if source_pose.targets_for_patch(faces,src[f[faces]])[SELF]-prev:
                failures["source_fold"]+=1
                continue
            staged=[]
            dif=0.
            sample=[]
            for (pose,_),baseline,(length,curves) in zip(poses,old_sets,volumes):
                new=pose.skin.copy()
                new[ids]=(pose.init[ids].astype(float)+np.einsum("vij,vj->vi",pose.j[ids],proposal[ids])).astype("<f4")
                tri=new[f[faces]]
                if not oriented(pose.init[f[faces]],tri):
                    failures["pose_orientation"]+=1
                    break
                hits=pose.targets_for_patch(faces,tri)
                if any(k!=key and not h.issubset(baseline[k]) for k,h in hits.items()):
                    failures["new_other_crossing"]+=1
                    break
                after,num=pose.curve_stats(key,hits[key],new)
                if num>curves or after>length+1.e-9:
                    failures["target_curve_regression"]+=1
                    break
                if length<=1.e-9 and not hits[key].issubset(baseline[key]):
                    break
                dif+=length-after
                sample.append(dict(step=pose.step,before_pairs=len(baseline[key]),
                                   after_pairs=len(hits[key]),before_length_m=length,
                                   after_length_m=after,before_curves=curves,after_curves=num))
                staged.append(new)
            # Credit retained geometric extent. Changed triangulation pair IDs
            # within the affected bank are reported separately.
            if len(staged)!=len(poses) or dif<.00035:
                failures["insufficient_clearance"]+=1
                continue
            delta=proposal
            source_pose.skin=src
            for (pose,_),new in zip(poses,staged):
                pose.skin=new
            result=dict(bank=key,scale=scale,rows_changed=int(np.any(shift!=0,axis=1).sum()),samples=sample)
            results.append(result)
            print(result,flush=True)
            break
        else:
            rejected.append(dict(bank=key, failures=dict(failures)))
        if index%12==11:
            print("screened",index+1,"of",len(candidates),flush=True)
    assert not np.any(delta[pinned])
    data=HERE/"edge-native-source-delta.npz"
    np.savez_compressed(data,delta=delta,protected=np.flatnonzero(pinned))
    proof.update(delta_sha256=digest(data),accepted=results, rejected=rejected,
                 changed_source_vertices=int(np.any(delta!=0,axis=1).sum()),
                 changed_legacy_guard_vertices=int(np.any(delta[old]!=0,axis=1).sum()))
    write(HERE/"edge-native-clearance-screen.json",proof)
    print("retained",len(results),flush=True)


if __name__=="__main__":
    main()

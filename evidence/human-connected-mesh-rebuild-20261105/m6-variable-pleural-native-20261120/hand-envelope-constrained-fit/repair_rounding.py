"""Verify axial coordinate rounding remedies against every incident tetrahedron."""
import numpy as np

from envelope_lp import FLOOR_DET,FLOOR_QUALITY,tet_metrics


def failures(points,tets):
    det,diam,_=tet_metrics(points[tets].astype(float))
    q=det/diam**3
    bad=(det<=FLOOR_DET)|(q<FLOOR_QUALITY)
    return bad


def repair(points,tets,initial,eligible,max_passes=4):
    proposal=points.copy()
    count=len(proposal)
    # Keep all touches of a changed vertex, including immutable neighbors.
    # Stable slot ordering makes equal-cost rounding choices reproducible.
    v=tets.ravel()
    order=np.argsort(v,kind="stable")
    offsets=np.r_[0,np.cumsum(np.bincount(v,minlength=count))]
    incident=order//4
    before=failures(proposal,tets)
    bad=before.copy()
    changes=[]
    for _ in range(max_passes):
        acted=False
        for cell in np.flatnonzero(bad):
            if not bad[cell]:
                continue
            best=None
            for index in tets[cell]:
                if not eligible[index]:
                    continue
                affected=incident[offsets[index]:offsets[index+1]]
                old=bad[affected]
                total=int(old.sum())
                current=proposal[index,0]
                for direction in (-np.inf,np.inf):
                    value=current
                    for ulp in range(1,5):
                        value=np.nextafter(value,np.float32(direction),dtype=np.float32)
                        if abs(float(value)-float(initial[index,0]))>0.00501:
                            continue
                        proposal[index,0]=value
                        trial=failures(proposal,tets[affected])
                        proposal[index,0]=current
                        improvement=total-int(trial.sum())
                        if improvement<=0:
                            continue
                        score=(improvement,-ulp)
                        if best is None or score>best[0]:
                            best=(score,index,value,affected,trial,current)
            if best is not None:
                _,index,value,affected,trial,current=best
                proposal[index,0]=value
                bad[affected]=trial
                changes.append((int(index),float(current),float(value)))
                acted=True
        if not acted or not bad.any():
            break
    after=failures(proposal,tets)
    assert np.array_equal(after,bad)
    assert np.array_equal(proposal[:,1:],points[:,1:])
    assert np.array_equal(proposal[~eligible],points[~eligible])
    return proposal,{"bad_before":int(before.sum()),"bad_after":int(after.sum()),
                     "changes":changes}

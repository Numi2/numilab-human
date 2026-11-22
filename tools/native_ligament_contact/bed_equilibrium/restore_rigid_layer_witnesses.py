"""Restore eligible unloaded neighbors of exact layer witnesses to the rigidly seated material reference."""
import json,shutil,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from screen_registered_dense_posture import load_geometry,measure
from recover_material_thickness import audit
from untangle_registered_donor_field import differentials
from construct_registered_donor_field import errors
from evaluate_registered_bed_transfer import digest

def main(full,donor,seed,out):
    x,target,owner,t,mat,outer,_,pins=load_geometry(full,donor)
    if 'root_translation_m' not in pins:raise ValueError('this construction requires rigid placement of the source skeleton')
    baseline=x+pins['root_translation_m'];y=x+np.load(seed/'displacement.npy')
    receipt=json.loads((seed/'loaded-registration.json').read_text());witness=json.loads((seed/'constructed-layers.json').read_text())
    if receipt['source']!=pins or witness['inputs']['CONSTRUCTED_NODE_SHA256']!=digest(seed/'current.node'):raise ValueError('unmatched exact witnesses')
    z=np.load(seed/'construction-boundary.npz');locked=z['node'];bed=np.flatnonzero(outer&(y[:,2]==0))
    faces=np.array([v for group in witness['witness_node_rows'].values() for pair in group for v in pair],dtype=int)
    points=np.unique(faces);tree=cKDTree(y[points]);dist=tree.query(y)[0];max_r=min(dist[locked].min(),dist[bed].min())
    start=y.copy();rr=np.concatenate([differentials(x,t[k:k+32768])[0] for k in range(0,len(t),32768)])
    best=None
    for core in (0.,.001,.003,.007):
        for edge in (.001,.003,.007,.012):
            if core+edge>=max_r:continue
            a=np.clip((dist-core)/edge,0,1);w=(1-a)**3*(1+3*a+6*a*a)
            trial=(y+(baseline-y)*w[:,None]).astype('f4').astype(float)
            trial[dist<=core]=baseline[dist<=core].astype('f4').astype(float)
            if not np.array_equal(trial[locked],y[locked]) or np.any(trial[:,2]<0) or not np.array_equal(y[bed],trial[bed]):continue
            if not np.array_equal(trial[points]-baseline[points],np.zeros((len(points),3))):continue
            selected=np.any(trial!=y,axis=1);incident=selected[t].any(1)
            failed=audit(trial,t[incident],rr[incident])[3].sum()
            key=(int(failed),int(selected.sum()))
            if best is None or key<best[0]:best=(key,trial)
    if best is None or best[0][0]:raise ValueError(('no admissible restored pose',None if not best else best[0]))
    y=best[1];j,h,det,bad=audit(y,t,rr);assert not bad.any()
    out.mkdir(exist_ok=False);np.save(out/'displacement.npy',y-x);shutil.copyfile(seed/'construction-boundary.npz',out/'construction-boundary.npz')
    result=dict(source=pins,failed=0,quality=measure(y,owner,t,mat),correction_to_seed=errors(y,start),moved_nodes=best[0][1],
                nearest_anchor_or_touching_skin_m=float(max_r),matched_original_rigid_nodes=len(points),
                scope='Source rigid points restored on exact loaded-layer witnesses. Requires a new complete layer audit and native physics trial.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(best[0],len(points),max_r)
if __name__=='__main__':main(*(Path(v).resolve() for v in sys.argv[1:]))

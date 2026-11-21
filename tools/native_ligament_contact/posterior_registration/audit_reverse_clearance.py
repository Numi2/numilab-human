"""Use the independently welded double-precision closed muscle to test the accepted cortical vertex positions."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from cortical_distance import CorticalSurface
from material_context import sha

def run(assets,surface,snapshot,lib,out):
    with snapshot.open() as f:rows=list(csv.DictReader(f))
    def field(k):return {int(r['index']):np.array([float(r[a]) for a in 'xyzw']) for r in rows if r['field']==k}
    m=field('material_node');pos=field('relative_position');rot={b:Rotation.from_quat(q) for b,q in field('orientation').items()}
    press=field('normal_and_force');shapes=field('reverse_shape');parents=field('reverse_parent_nodes')
    pts=np.stack([m[i][:3] for i in range(len(m))]);d=json.loads((assets/'manifest.json').read_text());base=d['closed_reverse']['first']
    pool=np.fromfile(assets/'pool-material-points.bin','f4').reshape(-1,7,4)
    start=0;res={};worst_force_gap=0;max_normal_delta=0;min_area=float('inf')
    for side,(sid,fem) in enumerate(((29,131),(30,145))):
        ref=np.load(surface/f'{sid}.npz');ids=ref['node'].astype(int)+start;start+=1190
        geom=np.einsum('ni,nij->nj',ref['vertex_shape'],pts[ids]);faces=ref['face']
        tri=geom[faces];min_area=min(min_area,.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).min())
        q=CorticalSurface(lib,geom,faces)
        for body in (1,fem,128):
            ix=np.flatnonzero((pool[base:,3,1].copy().view('u4')==fem)&(pool[base:,3,2].copy().view('u4')==body))+base
            cortex=rot[128].inv().apply(rot[body].apply(pool[ix,2,:3])+pos[body][:3]);gap,n,clos,_=q.query(cortex)
            minimum=int(gap.argmin());res[f'{sid}/{body}']=dict(points=len(ix),minimum_m=gap[minimum],deepest_point_id=int(ix[minimum]),below_half_micron=int(sum(gap<-.5e-6)))
            for j,i in enumerate(ix):
                if i not in press or press[i][3]==0:continue
                if i not in shapes:raise ValueError('Missing published reverse ancestry')
                p=shapes[i]@pts[parents[i].astype(int)]
                diff=np.linalg.norm(p-clos[j]);worst_force_gap=max(worst_force_gap,diff)
                if abs(gap[j])>1.e-7:max_normal_delta=max(max_normal_delta,np.linalg.norm(press[i][:3]+n[j]))
    out.write_text(json.dumps(dict(accepted_step=int(rows[0]['accepted_step']),snapshot=sha(snapshot),minimum_material_face_area_m2=min_area,maximum_loaded_feature_position_disagreement_m=worst_force_gap,maximum_loaded_normal_disagreement_above_0_1_um=max_normal_delta,pairs=res),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))

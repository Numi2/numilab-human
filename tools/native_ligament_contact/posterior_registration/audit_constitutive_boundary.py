"""Compare the original P1 material boundary before rounding with the emitted cortex."""
import csv,json,sys
from pathlib import Path
import numpy as np
from audit_finite_cortical_crossings import load_vertices
from count_accepted_cortical_crossings import dyadic,plane,straddles
from compare_hand_interfaces import query
from material_context import sha

def audit(part,assets,run,predicate):
    spec=json.loads((assets/'manifest.json').read_text());banks=spec['closed_reverse']
    ct=run/'accepted-pelvic-surfaces.csv';st=run/'accepted-pelvic-state.csv'
    step,surf=load_vertices(ct,{'material_vertex':sum(r['vertices'] for r in banks['sides']),'cortical_vertex':banks['unique_vertices']})
    with st.open() as f:
        delta={int(r['index']):np.array([r[k] for k in 'xyz'],dtype='f4').astype(float)
            for r in csv.DictReader(f) if r['field']=='material_delta' and int(r['accepted_step'])==step}
    ref=np.fromfile(assets/'pelvic-reference-nodes.bin','f4').reshape(-1,3,4).astype(float)
    nodes=ref[:,0,:3]+ref[:,1,:3]+np.stack([delta[i] for i in range(len(ref))])
    faces=np.fromfile(assets/'cortical-certificate-faces.bin','u4').reshape(-1,4)
    point=np.fromfile(assets/'pool-material-points.bin','f4').reshape(-1,7,4)[banks['first']:banks['first']+banks['unique_vertices']]
    ids=point[:,3].copy().view('u4')
    result={}
    for side,donor in enumerate((131,145)):
        region=spec['contact_flexibility'][side];m=region['muscle'];source=part/'conforming-boundary'/f'{m}.npz';mesh=np.load(source)
        if sha(source)!=spec['surface_material_reference']['sides'][side]['source']:raise ValueError('Mismatched material boundary')
        xyz=np.einsum('ij,ijk->ik',mesh['vertex_shape'],nodes[mesh['node']+region['first_node']])
        mf=mesh['face'];v0=banks['sides'][side]['first_vertex'];v1=v0+len(xyz)
        drift=np.linalg.norm(xyz-surf['material_vertex'][v0:v1],axis=1)
        for owner in (1,donor,128):
            f=faces[(faces[:,3]==side)&(ids[faces[:,0],2]==owner),:3]
            if not len(f):raise ValueError('Missing registered cortical faces')
            unique,bf=np.unique(f,return_inverse=True);bf=bf.reshape(f.shape)
            v=np.vstack((xyz,surf['cortical_vertex'][unique]));ft=np.vstack((mf,bf+len(xyz)))
            hits=[(int(i),int(j)-len(mf)) for i,j in query(predicate,v,ft,np.arange(len(ft))<len(mf)) if i<len(mf)<=j]
            lattice=dyadic(v);cross=[]
            for i,j in hits:
                x=[lattice[k] for k in mf[i]];b=[lattice[k+len(xyz)] for k in bf[j]]
                if straddles(x,b[0],plane(*b)) and straddles(b,x[0],plane(*x)):cross.append([i,j])
            result[f'{m}/{owner}']={'transverse_pairs':len(cross),'touch_pairs':len(hits)-len(cross),
                'first_pairs':sorted(cross)[:4],'max_boundary_emission_error_m':float(drift.max())}
    return {'accepted_step':step,'material_capture':sha(st),'cortex_capture':sha(ct),
            'representation':'binary64 original P1 weights on split reference plus native material delta',
            'banks':result,'total_transverse':sum(g['transverse_pairs'] for g in result.values())}
if __name__=='__main__':
    partition,assets,run,pred,out=map(Path,sys.argv[1:])
    out.write_text(json.dumps(audit(partition,assets,run,pred),indent=2)+'\n')

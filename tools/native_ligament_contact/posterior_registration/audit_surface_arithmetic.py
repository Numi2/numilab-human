"""Hold physical state and cortical vertices fixed and round the P1 surface once."""
import csv,json,sys,tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from qualify_material_motion import frame
from audit_finite_cortical_crossings import tests
from material_context import sha
def check(part,assets,ref,run,pred,out):
    sides=json.loads((assets/'manifest.json').read_text())['closed_reverse']['sides']
    poses=json.loads(ref.read_text())['initial_anatomical_registration']['body_poses']
    p=frame(next(row for row in poses if row['body_index']==128))
    rotation=Rotation.from_quat(p[3:]).inv()
    bind=np.fromfile(assets/'closed-material-vertices.bin','f4').reshape(-1,2,4)
    wt=bind[:,0].astype(float);idx=bind[:,1].copy().view('u4')
    s=run/'accepted-pelvic-state.csv';cap=run/'accepted-pelvic-surfaces.csv'
    with s.open() as f:delta=np.array([[row[k] for k in 'xyz'] for row in csv.DictReader(f) if row['field']=='material_delta'],'f4').astype(float)
    xyz=np.vstack([rotation.apply(np.load(part/'conforming-boundary'/f'{sid}.npz')['vertices']-p[:3]) for sid in (29,30)])
    x=(xyz+np.einsum('ij,ijk->ik',wt,delta[idx])).astype('f4')
    out.mkdir(exist_ok=True);step=None;old=[]
    with tempfile.TemporaryDirectory() as temp:
        test=Path(temp)/'rounded-once.csv'
        with cap.open() as source,test.open('w') as dest:
            reader=csv.DictReader(source);writer=csv.DictWriter(dest,fieldnames=reader.fieldnames);writer.writeheader()
            for row in reader:
                step=int(row['accepted_step'])
                if row['field']=='material_vertex':
                    i=int(row['index']);assert i==len(old)
                    old.append([row[k] for k in 'xyz'])
                    for k,q in zip('xyz',x[i]):row[k]=format(q,'.9g')
                writer.writerow(row)
        tests(assets,test,pred,out/'rounded-once-crossings.json')
    error=np.linalg.norm(np.asarray(old,'f4').astype(float)-x.astype(float),axis=1)
    result={'accepted_step':step,'source_capture':sha(cap),'state':sha(s),'reference_pose':sha(ref),
            'max_position_difference_m':float(max(error)),'rms_position_difference_m':float(np.sqrt(np.mean(error**2))),
            'actual_native_crossings':json.loads((run/'native-crossings.json').read_text())['total_transverse'],
            'single_rounding_counterfactual_crossings':json.loads((out/'rounded-once-crossings.json').read_text())['total_transverse']}
    (out/'arithmetic.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':check(*map(Path,sys.argv[1:]))

"""Quantify rigid-transform rounding at an unchanged accepted skeleton and material state."""
import csv,json,sys,tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from audit_finite_cortical_crossings import tests
from material_context import sha
def run(assets,run,pred,out):
    s=run/'accepted-pelvic-state.csv';cap=run/'accepted-pelvic-surfaces.csv'
    with s.open() as f:rows=list(csv.DictReader(f))
    def field(name,columns):
        return {int(r['index']):np.array([r[k] for k in columns],'f4').astype(float) for r in rows if r['field']==name}
    pos=field('relative_position','xyz');ori=field('orientation','xyzw')
    frames={i:Rotation.from_quat(q) for i,q in ori.items()}
    m=json.loads((assets/'manifest.json').read_text())['closed_reverse'];start=m['first']
    pts=np.fromfile(assets/'pool-material-points.bin','f4').reshape(-1,7,4)[start:start+m['unique_vertices']]
    b=pts[:,3].copy().view('u4')[:,2];x=pts[:,2,:3].astype(float).copy()
    for i in np.unique(b):
        if i!=128:x[b==i]=frames[128].inv().apply(pos[i]+frames[i].apply(x[b==i]))
    x=x.astype('f4');old=[];out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'rounded-rigid.csv'
        with cap.open() as source,path.open('w') as dest:
            reader=csv.DictReader(source);writer=csv.DictWriter(dest,fieldnames=reader.fieldnames);writer.writeheader()
            for row in reader:
                if row['field']=='cortical_vertex':
                    i=int(row['index']);assert i==len(old);old.append([row[z] for z in 'xyz'])
                    for z,q in zip('xyz',x[i]):row[z]=format(q,'.9g')
                writer.writerow(row)
        tests(assets,path,pred,out/'counterfactual.json')
    d=np.linalg.norm(np.array(old,'f4').astype(float)-x.astype(float),axis=1)
    c=json.loads((out/'counterfactual.json').read_text())
    result={'accepted_step':int(rows[0]['accepted_step']),'state':sha(s),'surface':sha(cap),'actual_pairs':json.loads((run/'native-crossings.json').read_text())['total_transverse'],'counterfactual_pairs':c['total_transverse'],'maximum_displacement_m':float(max(d)),'maximum_by_body_m':{str(i):float(max(d[b==i])) for i in np.unique(b)}}
    (out/'arithmetic.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))

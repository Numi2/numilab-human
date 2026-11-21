"""Differential regression tests for inherited triangle certificates."""
import csv,json,struct,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from refine_heart import write_subset

def run(probe,atlas,witness,pairs=None):
    cmd=[probe,atlas,witness,'split-tube']
    if pairs is not None:cmd.append(pairs)
    rows=subprocess.check_output(cmd,text=True).splitlines()
    assert len(rows)==1
    return list(map(int,rows[0].split()))[2:7]

def test(parent,leaf,full,out):
    a=[(0.,0.,0.),(1.,0.,0.),(0.,1.,0.)]
    cases=[
        ('separated',[(0.,0.,.1),(1.,0.,.1),(0.,1.,.1)],[(0,1,2),(3,4,5)],None,True),
        ('interior crossing',[(.1,.1,-.1),(.2,.1,-.1),(.1,.2,-.1)],[(0,1,2),(3,4,5)],[(0.,0.,0.)]*3+[(0.,0.,1.6)]*3,False),
        ('coincident distinct vertices',a,[(0,1,2),(3,4,5)],None,False),
        ('duplicate face',[],[(0,1,2),(0,1,2)],None,False),
        ('shared edge',[(0.,-1.,0.)],[(0,1,2),(1,0,3)],None,True),
        ('collapsed triangle',[(0.,0.,0.)],[(0,1,3)],None,False)]
    ruled=[]
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp);atlas=d/'geometry.bin';pairs=d/'pairs.bin';table=np.zeros((4,7),dtype='<f4');table[0,0]=-.125;table[1,0]=.125
        for name,other,faces,offset,good in cases:
            vs=a+other;v=np.zeros((len(vs),8,3),dtype='<f4');v[:,0]=vs;v[:,1]=.01 if offset is None else offset
            write_subset(atlas,v,np.array(faces,'<u4'),[(table,0,[])])
            result=run(full,atlas,d/'full.csv');wide=run(parent,atlas,d/'parent.csv')
            witnesses=list(csv.DictReader((d/'parent.csv').open()))
            assert len(witnesses)==sum(wide)
            pending=np.array([(int(w['first_face']),int(w['second_face'])) for w in witnesses if w['reason']!='face'],dtype='<u4').reshape(-1,2)
            pairs.write_bytes(struct.pack('<I',len(pending))+pending.tobytes())
            narrow=run(leaf,atlas,d/'leaf.csv',pairs) if any(wide) else [0]*5
            assert (not any(narrow))==good and (not any(result))==good
            ruled.append(dict(case=name,full=result,parent=wide,inherited=narrow))
    out.write_text(json.dumps(ruled,indent=2)+'\n');print(len(ruled),'matched geometry tests')
if __name__=='__main__':test(*map(Path,sys.argv[1:]))

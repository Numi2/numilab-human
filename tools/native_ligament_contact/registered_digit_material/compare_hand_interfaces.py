"""Compare all changed tissue triangles against all three layers and registered bones, with stable face indices."""
from pathlib import Path
import subprocess,sys,json,tempfile
import numpy as np
from scipy.spatial import cKDTree
from material_context import get_layers,read_inputs

def query(exe,x,f,changed):
    with tempfile.TemporaryDirectory() as folder:
        src=Path(folder)/'input';dst=Path(folder)/'hits'
        with src.open('wb') as stream:
            for a,t in (([len(x),len(f)],'u4'),(x,'f8'),(f,'u4'),(changed,'u1')):
                np.asarray(a,dtype=t).tofile(stream)
        subprocess.run([exe,src,dst],check=True)
        pairs=np.fromfile(dst,'u4').reshape(-1,2)
        if len(np.unique(pairs,axis=0))!=len(pairs):raise ValueError('Duplicate face pair')
        return set(map(tuple,pairs.tolist()))

def run(w,manifest,c,material,trial,exe,out):
    x=np.load(material)['x']; y=np.load(trial)
    tree=cKDTree(x); layers=get_layers(w/'qualified/native-coupled-006/native/resting-human.mrvpack',w/'qualified/motion-008/step-8.receipt.json')
    f=[];names=[];offset=[0];extra=[];nv=len(x)
    for k in ('outer','inner','deep'):
        v,t=layers[k];dist,ids=tree.query(v)
        if dist.max()!=0 or len(np.unique(ids))!=len(v):raise ValueError('Boundary ancestry changed')
        names.append(k);f.append(ids[t]);offset.append(offset[-1]+len(t))
    t=np.concatenate(f);m=np.any(np.any(x[t]!=y[t],axis=2),axis=1)
    _,_,bones,_,_=read_inputs(w/'current-package.json',manifest,c/'emitted-native/stderr.log')
    holes=np.unique(t[m]);reach=np.concatenate([x[holes],y[holes]]);lo=reach.min(0);hi=reach.max(0)
    for sid,(v,faces) in sorted(bones.items()):
        v=v.astype('f4')
        if np.any(v.max(0)<lo) or np.any(v.min(0)>hi):continue
        names.append(f'bone_{sid}');f.append(faces+nv);extra.append(v)
        nv+=len(v);offset.append(offset[-1]+len(faces))
    faces=np.concatenate(f);edited=np.pad(m,(0,len(faces)-len(m)))
    # Test omitted bones against a conservative box around every moved triangle,
    # including vertices of neighboring faces that are stationary.
    old=query(exe,np.concatenate([x]+extra),faces,edited)
    new=query(exe,np.concatenate([y]+extra),faces,edited)
    starts=np.array(offset)
    def describe(pairs):
        grouped={}
        for a,b in sorted(pairs):
            i,j=(int(k) for k in np.searchsorted(starts,[a,b],side='right')-1)
            grouped.setdefault(names[i]+'/'+names[j],[]).append([int(a-starts[i]),int(b-starts[j])])
        return grouped
    result=dict(scope='Changed material faces against all physical skin/fat layers and all AABB-eligible registered bones in the matched entrance pose',groups=names,changed_triangles=int(m.sum()),baseline=len(old),proposed=len(new),introduced=describe(new-old),resolved=describe(old-new),persistent=describe(old&new))
    out.write_text(json.dumps(result,separators=(',',':'))+'\n')
    print({k:result[k] for k in ('changed_triangles','baseline','proposed')},'introduced',len(new-old),'removed',len(old-new),flush=True)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))

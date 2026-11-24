"""Retriangulate compatible skin/fat cavities in both registered configurations."""
from pathlib import Path
from itertools import combinations
import json,sys
import numpy as np
from material_cavity_flips import replace_pass,boundary
from material_star_cavity import grow_pass
from supported_material_admission import geom
from compile_connected_joint import digest

def run(job,out):
    mat=job/'interval-certified-scene/material';loaded=job/'verified-connected-material/current.node'
    read=lambda p:np.loadtxt(p,skiprows=1,usecols=(1,2,3)).astype('f4').astype('f8')
    x,y=read(mat/'mesh.node'),read(loaded)
    rows=np.loadtxt(mat/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    t=rows[:,:4]-1;role=rows[:,4];out.mkdir()
    def score(cell,both=False):
        v,q=geom(x,cell);w,s=geom(y,cell);margin=np.zeros(len(cell))
        ok=(v>1e-18)&(w>1e-18)&(q>=1e-5)&(s>=1e-5)&(w>.4*v)&(w<2.5*v)
        if np.any(ok):
            n=cell[ok];F=np.linalg.solve(x[n[:,1:]]-x[n[:,:1]],y[n[:,1:]]-y[n[:,:1]])
            sv=np.linalg.svd(F,compute_uv=False);margin[ok]=np.minimum(2/sv[:,0],2*sv[:,-1])
        return (margin,ok) if both else margin
    margins=np.concatenate([score(t[i:i+16384]) for i in range(0,len(t),16384)])
    severe=np.flatnonzero(margins<1)
    if len(severe)!=448 or np.min(margins)<=0:raise ValueError('Source no longer matches coupled control')
    patch=[];ancestry=[];res={}
    for material in (15,16):
        sel=severe[role[severe]==material]
        near=np.unique(t[sel]);inpatch=(role==material)&np.isin(t,near).any(1)
        r=t[inpatch].copy();orig=r.copy()
        outside=t[~inpatch];members=np.unique(r)
        ext=outside[np.sum(np.isin(outside,members),axis=1)>=2]
        forbidden={e for c in ext for e in combinations(sorted(map(int,c)),2)}
        def quality(c):return score(c)
        history=[]
        for iteration in range(12):
            candidate,ops=replace_pass(x,r,forbidden,quality)
            if not ops:candidate,ops=grow_pass(x,r,forbidden,quality)
            if not ops:break
            valid=[]
            for old,new in ops:
                v,ok=score(new,True)
                if ok.all() and (v<1).sum()<=(score(old)<1).sum() and boundary(old)==boundary(new):valid.append((old,new))
            if not valid:break
            remove={tuple(map(int,c)) for old,_ in valid for c in old}
            keep=np.array([tuple(map(int,c)) not in remove for c in r])
            if (~keep).sum()!=len(remove):raise ValueError('Overlapping cavity removals')
            r=np.vstack([r[keep],*(n for _,n in valid)])
            for old,new in valid:ancestry.append(dict(material=material,old=old.tolist(),new=new.tolist()))
            m=score(r);history.append(dict(cavities=len(valid),severe=int((m<1).sum()),maximum_stretch_or_reciprocal=float(2/m.min())))
            print(material,iteration,history[-1],flush=True)
            if not (m<1).any():break
        if boundary(orig)!=boundary(r) or not np.array_equal(np.unique(orig),np.unique(r)) or np.min(score(r))<=0:raise ValueError('Failed constrained boundary or cell test')
        patch.append((np.flatnonzero(inpatch),orig,r))
        res[str(material)]=dict(start=int(len(sel)),remaining=int((score(r)<1).sum()),before_cells=len(orig),after_cells=len(r),history=history)
    o={'reference':digest(mat/'mesh.node'),'loaded':digest(loaded),'elements':digest(mat/'mesh.ele'),'materials':res,'cavities':len(ancestry)}
    (out/'result.json').write_text(json.dumps(o,indent=2)+'\n');(out/'ancestry.json').write_text(json.dumps(ancestry,separators=(',',':'))+'\n')
    np.savez_compressed(out/'patches.npz',**{f'{k}{i}':v for i,p in enumerate(patch) for k,v in zip(['source','before','after'],p)})
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))

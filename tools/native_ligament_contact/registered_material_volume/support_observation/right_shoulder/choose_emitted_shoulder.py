"""Select independently admissible source displacements and certify their mutual occupancy together."""
from pathlib import Path
import json,sys
import numpy as np
from manifold3d import Error
from fit_registered_assembly import layer_surface
from measure_assembly_contacts import canonical
from prepare_shoulder import MOVING,read
from compile_connected_joint import digest

def solid(x,f):
    m,log=canonical(x,f)
    if m is None:raise ValueError('Invalid moved muscle')
    return m,x.min(0),x.max(0)
def overlaps(a,b):
    x,l,h=a;y,ll,hh=b
    if np.any(h<ll)or np.any(hh<l):return 0.
    if y is None:raise ValueError('Unresolved fixed tissue in candidate footprint')
    z=x^y
    if z.status()!=Error.NoError or z.volume()< -1e-6:raise ValueError('Intersection unresolved')
    return max(0.,z.volume())
def choose(j,reg,opts,out):
    out.mkdir(exist_ok=True);meta=read(opts/'results.json');org=np.array(read(reg/'registration.json')['origin']);levels=meta['levels'];banks={}
    for sid in MOVING:
        cells={}
        for idx in range(len(levels)):
            fp=opts/f'{sid}-{idx}.npz'
            if not fp.exists():continue
            with np.load(fp)as z:
                cells[idx]=solid((z['accepted'][:len(z['x'])]-org)*1000,z['f'])
        if 0 not in cells:raise ValueError('No identity control')
        banks[sid]=cells
    L=np.min([l for c in banks.values()for _,l,h in c.values()],axis=0);H=np.max([h for c in banks.values()for _,l,h in c.values()],axis=0)
    fixed=layer_surface(j,org,L,H);refs=np.load(reg/'neighbors.npz')
    for key in refs.files:
        if not key.endswith('-v'):continue
        kind,b=map(int,key[:-2].split('-'))
        if not kind:continue
        pts=refs[key];l=pts.min(0);h=pts.max(0)
        if np.any(l>H)or np.any(h<L):continue
        m,_=canonical(pts,refs[key[:-1]+'f']);fixed[kind,b]=m,l,h
    unary={};pairs={};valid={};stats={}
    for n,c in banks.items():
        v=np.full(len(levels),np.inf);admit=[]
        for k,body in c.items():
            if abs(body[0].volume()/c[0][0].volume()-1)>1e-5:continue
            try:q=np.array([overlaps(body,b)for b in fixed.values()])
            except ValueError:
                if k==0:raise
                continue
            if k==0:base=q
            if np.any(q>base+1e-4):continue
            v[k]=q.sum();admit.append(k)
        valid[n]=admit;unary[n]=v
    for i,sid in enumerate(MOVING):
        for other in MOVING[i+1:]:
            mat=np.full((len(levels),len(levels)),np.inf)
            a,b=banks[sid],banks[other];base=overlaps(a[0],b[0])
            for k in valid[sid]:
                for l in valid[other]:
                    v=overlaps(a[k],b[l])
                    if v<=base+1e-4:mat[k,l]=v
            pairs[sid,other]=mat
    best=[float('inf'),None];combinations=[0]
    def select(indices,cost):
        i=len(indices)
        if i==len(MOVING):
            combinations[0]+=1
            if cost<best[0]:best[:]=[cost,indices]
            return
        a=MOVING[i]
        for k in valid[a]:
            value=cost+unary[a][k]+sum(pairs[b,a][j,k] for b,j in zip(MOVING,indices))
            if np.isfinite(value):select(indices+[k],value)
    select([],0.);baseline=sum(v[0]for v in unary.values())+sum(m[0,0]for m in pairs.values())
    if not np.isfinite(best[0])or best[0]>baseline+1e-7:raise ValueError('No admissible paired state')
    chosen={};quality={}
    for k,n in zip(best[1],MOVING):
        z=np.load(opts/f'{n}-{k}.npz');xyz=z['accepted'][:len(z['x'])];quality[n]=json.loads(str(z['quality']));chosen[n]=levels[k]
        np.savez_compressed(out/f'bank-{n}.npz',x=z['x'],w=z['w'],f=z['f'],native=xyz,owners=z['owners'])
        np.savez_compressed(out/f'cells-{n}.npz',**{key:z[key]for key in ['t','reference','rig_reference','opening','accepted']})
    res=dict(source=digest(opts/'results.json'),native=meta['native'],chosen=chosen,source_levels=levels,unary_choices=valid,admissible_combinations=combinations[0],reference_pair_sum_mm3=baseline,emitted_pair_sum_mm3=best[0],metrics=quality,maximum_per_pair_increase_mm3=1e-4,maximum_relative_registered_volume_change=1e-5)
    (out/'results.json').write_text(json.dumps(res,allow_nan=False,indent=2)+'\n');print('joint source',chosen,baseline,best[0])
if __name__=='__main__':choose(*(Path(x).resolve()for x in sys.argv[1:]))

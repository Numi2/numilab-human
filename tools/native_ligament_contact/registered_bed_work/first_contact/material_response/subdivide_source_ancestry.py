"""Clip each donor against individual refined children in their material reference frame."""
from pathlib import Path
import sys,json
import numpy as np
from build_carpal_cavity import FACES
from partition_carpal_material import solid
from fit_carpal_clearance import geom
from compile_connected_joint import digest


def run(mesh,material,flow,refinement,conditioned,output):
    z=np.load(material);base=np.load(refinement/'domain.npz');new=np.load(conditioned)
    split=np.load(refinement/'subdivision.npz');parent=split['parent']
    src=np.load(flow/'ancestry/overlap.npz');receipt=json.loads((flow/'ancestry/result.json').read_text())
    if receipt['mesh']!=digest(material) or receipt['mapping']!=digest(flow/'ancestry/overlap.npz'):raise ValueError('Wrong parent map')
    if not np.array_equal(new['t'],base['t']) or not np.array_equal(new['role'],z['role'][parent]):
        raise ValueError('Lost material ownership')
    # Exact edge ancestry is a linear pullback even when the emitted midpoint rounded off an edge.
    frac=json.loads((refinement/'result.json').read_text())['fraction']
    x=np.zeros_like(base['x'],dtype='f8');x[:len(z['x'])]=z['x'].astype(float)
    for a,b,j in split['nodes']:x[j]=x[a]+frac*(x[b]-x[a])
    t=new['t'];quad=x[t]
    cv0,_=geom(quad);cv,_=geom(new['x'].astype(float)[t]);rv,_=geom(z['x'].astype(float)[z['t']])
    if np.max(abs(np.bincount(parent,weights=cv0)-rv)/rv)>5e-8:raise ValueError('Incomplete parent subdivision')
    # Strict positive determinants over the entire conditioning path, including Float32 subdivision emission.
    worst=np.inf
    for q0,q1 in [(quad,base['x'].astype(float)[t]),(base['x'].astype(float)[t],new['x'].astype(float)[t])]:
        A=q0[:,1:]-q0[:,:1];D=q1[:,1:]-q1[:,:1]-A
        def coef(m):
            u,v,w=(D[:,i] if m&(1<<i) else A[:,i] for i in range(3))
            return np.einsum('ij,ij->i',u,np.cross(v,w))/6.
        c=np.array([coef(0),sum(coef(i) for i in (1,2,4)),sum(coef(i) for i in (3,5,6)),coef(7)])
        low=np.minimum(c[0],c.sum(0))
        for i in np.flatnonzero((D!=0.).any((1,2))):
            for root in np.roots((3*c[3,i],2*c[2,i],c[1,i])):
                if abs(root.imag)<1e-12 and 0<root.real<1:
                    low[i]=min(low[i],np.polynomial.polynomial.polyval(root.real,c[:,i]))
        worst=min(worst,low.min())
    if worst<=0:raise ValueError('Invalid continuous material volume')
    selected=np.bincount(parent)>1
    ce=src['child'];affected=selected[ce]
    if np.any(src['edge_is_advected'][affected]):
        raise ValueError('Split includes advected source. Recover its earlier material frame before clipping.')
    stable=ce[~affected]
    moved=np.any(new['x'].astype(float)[t[:len(z['t'])]]!=z['x'].astype(float)[z['t']],axis=(1,2))
    p=list(src['parent'][~affected]);c=list(stable)
    w=list(src['volume_mm3'][~affected]*(cv[stable]/rv[stable]))
    mode=list(src['edge_is_advected'][~affected]|moved[stable])
    if not np.array_equal(parent[:len(z['t'])],np.arange(len(z['t']))):raise ValueError('Unknown subdivision order')
    elem=np.loadtxt(mesh/'mesh.ele',skiprows=1,dtype='i4')
    coords=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=(1,2,3))
    center=x.mean(0);donors={};old=np.asarray(src['parent'])
    for j in np.flatnonzero(selected[parent]):
        v=solid((quad[j]-center)*1000,FACES)
        for k in old[ce==parent[j]]:
            if elem[k,5]!=new['role'][j]:raise ValueError('Material switch')
            if k not in donors:donors[k]=solid((coords[elem[k,1:5]-1]-center)*1000,FACES)
            hit=v^donors[k]
            if hit.status()!=v.status():raise ValueError('Invalid donor/child clipping')
            volume=hit.volume()
            if volume>0:p.append(k);c.append(j);w.append(volume*cv[j]/cv0[j]);mode.append(True)
    out=output/'ancestry';out.mkdir(parents=True)
    coverage=np.bincount(c,weights=w,minlength=len(t));missing=np.flatnonzero(coverage<=0)
    file=out/'overlap.npz'
    np.savez_compressed(file,parent=p,child=c,volume_mm3=w,edge_is_advected=mode,parent_ids=src['parent_ids'],parent_volume_mm3=src['parent_volume_mm3'],child_volume_mm3=cv*1e9)
    doc=dict(mesh=digest(conditioned),mapping=digest(file),parent_mesh=digest(material),source_donors=digest(flow/'ancestry/overlap.npz'),
             subdivision=digest(refinement/'subdivision.npz'),minimum_path_volume_m3=float(worst),split_parents=int(selected.sum()),split_children=int(selected[parent].sum()),uncovered_children=missing.tolist())
    (out/'result.json').write_text(json.dumps(doc,indent=2)+'\n')
    print(doc['split_parents'],doc['split_children'],'uncovered',doc['uncovered_children'])
    if len(missing):raise ValueError('Material source coverage incomplete')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))

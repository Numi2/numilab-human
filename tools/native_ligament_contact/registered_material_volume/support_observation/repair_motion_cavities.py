"""Retriangulate like-labelled cavities at fixed reference, loaded and accepted coordinates."""
import json,sys
from itertools import combinations
from pathlib import Path
import numpy as np
from material_cavity_flips import replace_pass,boundary
from material_star_cavity import grow_pass
from supported_material_admission import geom
from build_bed_audit import sha

def run(v,case,quality,out):
    out.mkdir();raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype=np.int32)
    t=raw[:,:4]-1;lab=np.loadtxt(v/'current-child.labels.csv',skiprows=1,delimiter=',',dtype=np.int64)
    if not np.array_equal(lab[:,0],np.arange(len(t))) or not np.array_equal(lab[:,1],raw[:,4]):raise ValueError('Mismatched material rows')
    points=[np.loadtxt(f,skiprows=1,usecols=(1,2,3)) for f in (v/'mesh.node',v/'current.node',case/'physical-accepted.node')]
    q=json.loads(quality.read_bytes());rows=np.array([c['row']for r in q['metrics'].values()for c in r['cells']if c['quality'][2]<1e-5]);keys=np.unique(lab[rows,1:],axis=0);nodes=np.unique(t[rows]);near=np.flatnonzero(np.isin(t,nodes).any(1))
    # Any new flip edge connects vertices in these stars. This second ring includes
    # every existing occurrence of any possible proposed edge, including other labels.
    ring=np.flatnonzero(np.isin(t,np.unique(t[near])).any(1))
    def margin(cells):
        constraints=[]
        for k,x in enumerate(points):
            volume,shape=geom(x,cells);constraints.extend([volume/1e-18,shape/1e-5])
            if k==0:rest=volume
            else:
                ratio=np.divide(volume,rest,out=np.zeros_like(volume),where=rest>0);constraints.extend([ratio/(.4 if k==1 else .35),np.divide(2.5,ratio,out=np.zeros_like(ratio),where=ratio>0)])
        return np.min(constraints,axis=0)
    edits=[];used=set();bad=set(map(int,rows));global_wall=np.load(v/'material-walls.npz')
    protected=set()
    for k in 'OIDC':
        faces=global_wall[k]; protected.update(tuple(sorted(map(int,f)))for f in faces[np.isin(faces,np.unique(t[near])).any(1)])
    for key in keys:
        mask=np.all(lab[near,1:]==key,axis=1);ids=near[mask];others=np.setdiff1d(ring,ids)
        block={e for c in t[others] for e in combinations(sorted(map(int,c)),2)}
        local=t[ids]; _,changes=replace_pass(points[0],local,block,margin)
        _,stars=grow_pass(points[0],local,block,margin)
        changes.extend(stars)
        index={tuple(sorted(map(int,c))):int(r) for c,r in zip(local,ids)}
        for old,new in changes:
            src=[index[tuple(sorted(map(int,c)))] for c in old]; verts=set(map(int,old.ravel()))
            if used & verts or not bad.intersection(src) or np.any(margin(new)<1):continue
            boundary_faces={tuple(sorted(f)) for f in boundary(old)}
            interior={tuple(sorted(f))for c in old for f in combinations(map(int,c),3)}-boundary_faces
            if protected.intersection(interior):continue
            if boundary(old)!=boundary(new):raise ValueError('Boundary mismatch')
            checks=[]
            for x in points:
                ov=geom(x,old)[0].sum();vv=geom(x,new)[0].sum()
                if abs(vv-ov)>1e-9*ov:raise ValueError('Cavity changes volume')
                checks.append(float(min(geom(x,new)[1])))
            edits.append(dict(old=src,new=new.tolist(),label=key.tolist(),minimum_shape=checks));used.update(verts);bad.difference_update(src)
    record=dict(topology=sha(v/'mesh.ele'),source_coordinates=sha(v/'mesh.node'),loaded_coordinates=sha(v/'current.node'),accepted_coordinates=sha(case/'physical-accepted.node'),original_failures=len(rows),remaining_on_recorded_positions=sorted(bad),patches=edits)
    (out/'cavities.json').write_text(json.dumps(record,indent=2)+'\n');print(len(rows),len(bad),len(edits),flush=True)
if __name__=='__main__':run(*(Path(x).resolve()for x in sys.argv[1:]))

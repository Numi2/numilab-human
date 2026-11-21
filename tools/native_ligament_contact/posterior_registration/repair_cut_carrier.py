"""Refit the background tetrahedralization around a measured cut-material collapse, keeping every registered vertex and boundary face."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from condition_material_cells import q,boundary,det,FACE
from material_context import sha

def match(a,b):return {tuple(sorted(x)) for x in a}=={tuple(sorted(x)) for x in b}
def run(source,carrier,output,side,cell):
    output.mkdir(exist_ok=True);summary={}
    for sid in (29,30):
        p=np.load(source/'projected'/f'{sid}.npz');v=p['vertices'].astype(float);t=p['cells'].astype(int)
        oldpath=carrier/str(sid)/'source.1.ele';original=np.loadtxt(oldpath,skiprows=1,dtype=int)[:,1:5]
        assert np.array_equal(t,original)
        tgt=output/str(sid);tgt.mkdir(exist_ok=True)
        path=tgt/'source.1.ele'
        if sid!=side:path.write_bytes(oldpath.read_bytes());continue
        incident={}
        for i,el in enumerate(t):
            for f in FACE:incident.setdefault(tuple(sorted(el[list(f)])),[]).append(i)
        near={i for f in FACE for i in incident[tuple(sorted(t[cell,list(f)]))] if i!=cell}
        starts=[{cell,*a} for k in range(1,len(near)+1) for a in itertools.combinations(near,k)]
        for ring in list(starts):
            more={i for j in ring for f in FACE for i in incident[tuple(sorted(t[j,list(f)]))]}-ring
            starts.extend([ring|{i} for i in more]);starts.append(ring|more)
        starts.extend([set(np.flatnonzero(np.isin(t,e).sum(1)==len(e))) for k in (1,2) for e in itertools.combinations(t[cell],k)])
        best=None;before=q(v,t[[cell]])[0]
        for indices in starts:
            ix=np.array(sorted(indices));old=t[ix];faces=boundary(old);minimum=q(v,old).min()
            for anchor in np.unique(old):
                fs=faces[np.all(faces!=anchor,axis=1)]
                new=np.column_stack((np.full(len(fs),anchor),fs))
                if not len(new) or not match(boundary(new),faces):continue
                qual=q(v,new);lowest=qual.min()
                if lowest<=max(minimum*1.01,before*1.5):continue
                if not np.isclose(det(v,new).sum(),det(v,old).sum(),rtol=1e-10,atol=1e-21):continue
                if best is None or lowest>best[0]:best=lowest,ix,new
        if best is None:raise ValueError('No admissible existing-vertex cavity')
        lowest,ix,new=best;keep=np.ones(len(t),bool);keep[ix]=False;repaired=np.vstack((t[keep],new))
        assert match(boundary(repaired),boundary(t)) and det(v,repaired).min()>0
        np.testing.assert_allclose(det(v,repaired).sum(),det(v,t).sum(),rtol=2e-14)
        with path.open('w') as f:
            f.write(f'{len(repaired)} 4 0\n');np.savetxt(f,np.column_stack((np.arange(len(repaired)),repaired)),fmt='%d')
        summary[str(sid)]=dict(source=sha(oldpath),target=sha(path),witness=int(cell),removed=ix.tolist(),replacement=new.tolist(),target_quality_before=float(before),minimum_replacement_quality=float(lowest),boundary_faces=len(boundary(t)),cell_count_before=len(t),cell_count_after=len(repaired))
    (output/'repair.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    for name in ('source','carrier','output'):a.add_argument(name,type=Path)
    a.add_argument('side',type=int);a.add_argument('cell',type=int);run(**vars(a.parse_args()))

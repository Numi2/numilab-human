"""Prepare source-identified nodal masses and donor contributions for the existing native partitioner."""
import json,sys
from pathlib import Path
import numpy as np
from qualify_material_motion import frame,local
from pose_glutes import parts
from pelvic_specimen import properties
from material_context import sha

def prepare(cfg,receipt,projected,material,out):
    args=json.loads(cfg.read_text())['argv'];asset=Path(args[args.index('--soft-tissue-payload')+1])
    poses={r['body_index']:frame(r) for r in json.loads(receipt.read_text())['initial_anatomical_registration']['body_poses']}
    density=properties(material)['density'];summary={};offset=0;file=out/'mass-reference.txt';out.mkdir(exist_ok=True)
    with file.open('w') as fp:
        for sid in (29,30):
            p=np.load(projected/(str(sid)+'.npz'));v=p['vertices'].astype(float);t=p['cells'];d=parts(asset,sid);w=d[1].astype(float);donors=d[2]
            np.testing.assert_allclose(w.sum(1),1,rtol=0,atol=8.e-8)
            if min(w.ravel())<0 or len(w)!=len(v):raise ValueError('Source carrier and donor roster differ')
            w/=w.sum(1)[:,None]
            vm=density*p['consistent_shape_integrals'];allocated=vm@w[t]
            nodal=np.zeros((len(v),len(donors)))
            for k in range(4):np.add.at(nodal,t[:,k],allocated[:,k])
            if nodal.min() < 0:raise ValueError(('Negative donor mass',nodal.min()))
            total=nodal.sum(1)
            np.testing.assert_allclose(total.sum(),p['cell_volumes'].sum()*density,rtol=1.e-12)
            loc=[local(v,poses[b]) for b in donors]
            for j in np.flatnonzero(total>0):
                donor=nodal[j].argmax()
                print('node',offset+j,int(donors[donor]),total[j],*loc[donor][j],file=fp)
                for k,part in enumerate(nodal[j]):
                    if part>0:print('source',offset+j,int(donors[k]),part,*loc[k][j],file=fp)
            exact=np.einsum('tij,tik,tjl->kl',vm,v[t],v[t])
            lump=v.T@(total[:,None]*v);diff=lump-exact
            summary[sid]=dict(nodes=int(np.count_nonzero(total)),zero_support_nodes=int(np.count_nonzero(total==0)),minimum_positive_nodal_mass_kg=float(min(total[total>0])),total_kg=total.sum(),donor_kg={int(b):nodal[:,k].sum() for k,b in enumerate(donors)},exact_first_kg_m=(total@v).tolist(),lumped_minus_consistent_second_moment_kg_m2=diff.tolist())
            offset+=len(v)
    (out/'result.json').write_text(json.dumps(dict(ledger_sha256=sha(file),receipt_sha256=sha(receipt),source_sha256=sha(asset),material_sha256=sha(material),sides=summary),indent=2)+'\n')
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))

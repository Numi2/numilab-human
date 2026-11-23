"""Attribute cortical overlap and displaced tissue mass to each original native material cell."""
from pathlib import Path
import json,sys
import numpy as np
from audit_registered_skeleton import digest
from partition_carpal_material import solid
from build_carpal_cavity import FACES


def run(mesh,plc,conflicts,broad,out):
    z=np.load(plc/'cavity.npz');w=np.load(conflicts/'witnesses.npz');result=json.loads((broad/'result.json').read_bytes())
    x=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=(1,2,3))
    if digest(mesh/'mesh.node')!=result['source_material']:raise ValueError('Source mass mesh mismatch')
    rows=np.loadtxt(mesh/'mesh.ele',skiprows=1,dtype='i4');mass=np.fromfile(mesh/'cell-mass.f64',dtype='<f8')
    cells=z['source_tetrahedra'];r=rows[cells];t=r[:,1:5]-1
    center=x[t].mean((0,1));local=(x-center)*1000;points=local[t];lo=points.min(1);hi=points.max(1)
    used={};stats={}
    for sid in (8,9,10,11,12,13):
        y,inv=np.unique(w['cortex'+str(sid)],axis=0,return_inverse=True);y=(y-center)*1000
        f=inv[w['faces'+str(sid)]]
        if np.einsum('ij,ij->',y[f[:,0]],np.cross(y[f[:,1]],y[f[:,2]]))<0:f=f[:,[0,2,1]]
        bone=solid(y,f);ids=np.flatnonzero(((lo<=y.max(0))&(hi>=y.min(0))).all(1));total=0.
        for j in ids:
            tet=solid(points[j],FACES);hit=tet^bone
            if hit.status()!=tet.status():raise ValueError('Indeterminate material partition')
            amount=hit.volume()
            if amount<=0:continue
            volume=tet.volume()
            if amount>volume+max(1e-11,volume*1e-9):raise ValueError('Invalid source fraction')
            slot=used.setdefault(j,[volume,0.]);slot[1]+=min(amount,volume)
            total+=amount
        stats[sid]=total;print('bone',sid,'mm3',total,flush=True)
    parent=np.array(sorted(used));ids=cells[parent];full,cut=np.array([used[k] for k in parent]).T
    dst=np.maximum(full-cut,0.);m=mass[ids]*np.minimum(cut/full,1.);out.mkdir()
    np.savez_compressed(out/'cut-source-regions.npz',source_tet=ids,material=rows[ids,5],original_volume_mm3=full,interior_volume_mm3=cut,
                        exterior_volume_mm3=dst,displaced_mass_kg=m,center_m=center)
    summary={}
    for k in (15,16):
        val=float(cut[rows[ids,5]==k].sum());expected=sum(r[str(k)]['removed_overlap_mm3'] for r in result['sides'].values())
        if abs(val-expected)>max(1e-6,abs(expected)*1e-8):raise ValueError('Parent ledger differs from the independent whole-material cut')
        summary[k]=dict(intersected_parents=int((rows[ids,5]==k).sum()),overlap_mm3=val,displaced_mass_kg=float(m[rows[ids,5]==k].sum()))
    (out/'result.json').write_text(json.dumps(dict(parent_mass=digest(mesh/'cell-mass.f64'),original=digest(mesh/'mesh.ele'),cortical_overlap=stats,
       materials=summary,parents=len(ids),whole_cells_displaced=int((dst<full*1.e-10).sum())),indent=2)+'\n')
    print('source parents',len(parent),'displaced kg',m.sum(),flush=True)


if __name__=='__main__':run(*map(Path,sys.argv[1:]))

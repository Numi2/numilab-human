"""Assemble the qualified two-material cavity into the full native TetGen importer."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
from register_material_surfaces import sha


def run(mesh,domain,overlap,emitted,proof,out):
    verdict=json.loads(proof.read_text());route=json.loads((emitted/'result.json').read_text())
    if sha(domain)!=verdict['candidate_sha256'] or verdict['missing_joining_faces'] or any(b['invalid'] or b['cortex_intersection_mm3']!=0 for b in verdict['materials'].values()):
        raise ValueError('Domain is not certified')
    if not all(route[tag].get('passed') for tag in ('SKIN','FAT')):raise ValueError('Missing mass closure')
    z=np.load(domain);ov=np.load(overlap/'overlap.npz');owners=ov['parent_ids']
    x=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=(1,2,3));fmt=np.loadtxt(mesh/'mesh.ele',skiprows=1,dtype='i4')
    t=fmt[:,1:5]-1;role=fmt[:,5];keep=np.ones(len(t),dtype=bool);keep[owners]=False
    masses=np.fromfile(mesh/'cell-mass.f64',dtype='<f8');child_mass=np.zeros(len(z['t']))
    for tag in ('SKIN','FAT'):
        d=np.load(emitted/(tag+'-mapping.npz'));dest=d['recipient'];old=dest<len(t)
        masses[dest[old]]=d['receiver_mass_kg'][old];child_mass[dest[~old]-len(t)]=d['receiver_mass_kg'][~old]
    if np.any(child_mass<=0):raise ValueError('Replacement contains massless cells')
    # Keep existing source identifiers when the emitted coordinate is identical.
    prior=np.unique(t[owners]);source={x[k].astype('f4').tobytes():int(k) for k in prior}
    if len(source)!=len(prior):raise ValueError('Ambiguous coordinate on old material cavity')
    extended=x.tolist();mapping=np.full(len(z['x']),-1,dtype=int)
    for k in np.unique(z['t']):
        p=z['x'][k];idx=source.get(p.tobytes())
        if idx is None:
            idx=len(extended);extended.append(p.astype(float).tolist())
        mapping[k]=idx
    new=mapping[z['t']]
    joined=np.vstack((t[keep],new));active,indices=np.unique(joined,return_inverse=True);joined=indices.reshape(-1,4)
    xyz=np.asarray(extended)[active]
    original_to_new=np.full(len(x),-1,dtype=int);old=active<len(x);original_to_new[active[old]]=np.where(old)[0]
    # Attach the geometric label of each child's largest same-material volume intersection.
    # Retain *all* other ancestry and actual mass fractions in the transfer receipts.
    order=np.lexsort((ov['parent'],-ov['volume_mm3'],ov['child']))
    ci,first=np.unique(ov['child'][order],return_index=True);best=ov['parent'][order[first]]
    if not np.array_equal(ci,np.arange(len(new))) or not np.array_equal(role[best],z['role']):raise ValueError('Geometric label unresolved')
    labels=np.loadtxt(mesh/'current-child.labels.csv',delimiter=',',dtype='i8',skiprows=1)
    if not np.array_equal(labels[:,0],np.arange(len(labels))):raise ValueError('Invalid source labels')
    out.mkdir();nr=len(xyz);nt=len(joined);materials=np.r_[role[keep],z['role']]
    np.savetxt(out/'mesh.node',np.column_stack((np.arange(1,nr+1),xyz)),fmt=['%d','%.17g','%.17g','%.17g'],header=f'{nr} 3 0 0',comments='')
    np.savetxt(out/'mesh.ele',np.column_stack((np.arange(1,nt+1),joined+1,materials)),fmt='%d',header=f'{nt} 4 1',comments='')
    lab=np.vstack((labels[keep],labels[best]));lab[:,0]=np.arange(nt)
    np.savetxt(out/'current-child.labels.csv',lab,fmt='%d',delimiter=',',header='child_tet_row,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id',comments='')
    m=np.r_[masses[keep],child_mass];nodal=np.zeros(nr);np.add.at(nodal,joined.ravel(),np.repeat(.25*m,4))
    m.astype('<f8').tofile(out/'cell-mass.f64');nodal.astype('<f8').tofile(out/'node-mass.f64')
    np.savez_compressed(out/'material-source-indices.npz',original_to_emitted=original_to_new,old_cavity_tets=owners,child_geometric_parent=best,child_begin=np.count_nonzero(keep),emitted_from_old=active[old],removed_original_nodes=np.flatnonzero(original_to_new<0))
    native=json.loads((mesh/'native-import-build.json').read_text());cmd=native['launch'].copy();cmd[1]=str(out.resolve())
    if sha(Path(cmd[0]))!=native['binary']:raise ValueError('Changed native importer')
    proc=subprocess.run(cmd,capture_output=True,text=True);(out/'native-import.json').write_text(proc.stdout);(out/'native-import.stderr').write_text(proc.stderr)
    result=dict(command=cmd,binary=native['binary'],exit=proc.returncode,source_receipt=sha(mesh/'receipt.json'),domain_sha256=sha(domain),removed_original_nodes=int(np.sum(original_to_new<0)),added_nodes=int(np.sum(active>=len(x))),removed_cells=len(owners),added_cells=len(new),native_report_sha256=sha(out/'native-import.json'),output_hashes={k:sha(out/k) for k in ('mesh.node','mesh.ele','current-child.labels.csv','cell-mass.f64','node-mass.f64','material-source-indices.npz')})
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    if proc.returncode:raise ValueError(proc.stderr or proc.stdout)
    print(proc.stdout)
if __name__=='__main__':run(*(Path(k) for k in sys.argv[1:]))

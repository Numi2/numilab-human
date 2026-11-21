"""Match oriented detailed skin faces across two real native launch inputs."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys,numpy as np
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def mesh(path,decode):
    c=json.loads(path.read_text());a=c['argv'];f=Path(a[a.index('--skin-payload')+1]).resolve();r=decode(f.read_bytes())
    return c,f,r
def reconcile(work,old,new,out):
    sys.path.insert(0,str(work/'resting-clearance-and-atrium-002/publication/scripts'))
    from rebind_human_skin_reference import decode
    c0,p0,a=mesh(old,decode);c1,p1,b=mesh(new,decode)
    if a['source_sha256']!=b['source_sha256'] or a['nv']!=b['nv'] or not np.array_equal(a['binds'],b['binds']):raise ValueError('Skin vertex or body identity must be rebuilt')
    def identity(face):
        i=int(np.argmin(face));return tuple(map(int,np.roll(face,-i)))
    first={identity(f):i for i,f in enumerate(a['inds'])}
    if len(first)!=len(a['inds']):raise ValueError('Duplicate oriented source faces')
    mapping=np.array([first.get(identity(f),-1) for f in b['inds']],dtype='<i4')
    matched=mapping>=0
    changed_indices=np.flatnonzero((a['inds']!=b['inds']).any(1))
    moved=np.linalg.norm(a['verts'][:,:3].astype(float)-b['verts'][:,:3],axis=1)
    changedweights=np.any(a['weights']!=b['weights'],axis=1)
    if len(set(mapping[matched]))!=sum(matched):raise ValueError('Ambiguous oriented ancestry')
    touched=np.any((moved>0)[b['inds']],axis=1)|np.any(changedweights[b['inds']],axis=1)
    unchanged=matched&~touched
    if np.any(unchanged):
        ro=a['inds'][mapping[unchanged]];rn=b['inds'][unchanged]
        if not np.array_equal(np.sort(ro,axis=1),np.sort(rn,axis=1)):raise ValueError('Bad face transfer')
    out.mkdir(exist_ok=True);artifact=out/'oriented-face-correspondence.npz'
    np.savez_compressed(artifact,display_to_old_oriented_face=mapping,unchanged_source_geometry_and_weights=unchanged,new_or_retriangulated_face=np.flatnonzero(~matched),changed_binding_vertices=np.flatnonzero(changedweights),moved_vertices=np.flatnonzero(moved>0))
    report={'former_skin_sha256':sha(p0),'current_skin_sha256':sha(p1),'shared_skin_source_sha256':a['source_sha256'].hex(),'registered_bodies':a['nb'],'vertex_count':a['nv'],'source_faces':len(a['inds']),'current_faces':len(b['inds']),
        'changed_index_rows':len(changed_indices),'matched_oriented_source_faces':int(matched.sum()),'unmatched_or_retriangulated_faces':int((~matched).sum()),'unchanged_geometry_and_weights_faces':int(unchanged.sum()),
        'moved_positional_vertices':int((moved>0).sum()),'max_movement_m':float(moved.max()),'changed_weight_vertices':int(changedweights.sum()),'metadata_sha256':sha(artifact),
        'former_full_owner_enabled': c0['environment'].get('NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM'),'current_full_owner_enabled':c1['environment'].get('NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM'),
        'scope':'Exact detailed-source topology and binding correspondence. Material node ancestry, TetGen volumes, mass and skin-to-muscle clearance require their own physical checks.'}
    (out/'reconciliation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':reconcile(*map(Path,sys.argv[1:]))

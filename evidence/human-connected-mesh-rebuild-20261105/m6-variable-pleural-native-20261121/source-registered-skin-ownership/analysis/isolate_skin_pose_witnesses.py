"""Isolate donor contributions on known self-crossing faces at one native pose."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from count_accepted_cortical_crossings import dyadic
from map_skin_witness_owners import digest
def run(work,case,step):
    sys.path[:0]=[str(work/'resting-surface-clearance-001/src'),str(work/'resting-clearance-and-atrium-005'),str(work/'resting-clearance-and-atrium-002/publication/scripts')]
    from fast_skin_intersection_audit import exact_integer_pairs
    from rebind_human_skin_reference import decode
    label=str(step);receipt=json.loads((case/f'step-{step}.receipt.json').read_text())
    data=np.load(case/f'skin-pose-{step}.npz')
    src=json.loads((case/'launch.json').read_text())['argv']
    skin=Path(src[src.index('--skin-payload')+1]);p=decode(skin.read_bytes())
    tissue=Path(src[src.index('--soft-tissue-payload')+1]);manifest=json.loads(tissue.with_suffix('.manifest.json').read_text())
    names={b['core_body_index']:b['myosim_body'] for s in manifest['source']['surfaces'] for b in s.get('body_bindings',[])}
    report=case/f'exact-skin-surface-step-{step}.json'
    if not report.exists():report=case/'exact-skin-surface.json'
    rep=json.loads(report.read_text())
    if (rep['accepted_pack_sha256']!=receipt['pack_file_sha256'] or data['pack_sha256']!=receipt['pack_file_sha256'] or
        str(data['source_sha256'])!=digest(skin)):raise ValueError('Capture or bindings do not match')
    fs=p['inds'];pairs=np.asarray(rep['self_intersections']['triangle_pairs'],dtype=int)
    ids,faces=np.unique(fs[pairs],return_inverse=True);faces=faces.reshape(len(pairs),2,3)
    obs=data['positions'][ids];source=p['verts'][ids,:3].astype(float);weights=p['weights'][ids].astype(float)
    poses={r['body_index']:r for r in receipt['accepted_body_poses']}
    coords=[]
    for rec in p['binds']:
        b=poses[int(rec['body'])];vals=rec['v']
        xyz=Rotation.from_quat(vals[3:7]).apply(source*vals[7])+vals[:3]
        coords.append(Rotation.from_quat(b['quaternion_xyzw']).apply(xyz)+b['position_m'])
    coords=np.stack(coords,axis=1)
    dom=weights.argmax(1);anchor=coords[np.arange(len(ids)),dom]
    def hit(x):
        selected=np.asarray(x,dtype='<f4')[faces].reshape(-1,6,3)
        out=[]
        for tri,ix in zip(selected,faces,strict=True):
            lattice=tuple(map(tuple,dyadic(tri)))
            a=(lattice[:3],None,None,0,list(ix[0]));b=(lattice[3:],None,None,1,list(ix[1]))
            out.append(bool(exact_integer_pairs([(a,b)],same_surface=True)['count']))
        return np.array(out,dtype=bool)
    orig=hit(obs)
    if not np.all(orig):raise ValueError('Known native witnesses do not reproduce')
    src=hit(source);blend=np.einsum('ij,ijk->ik',weights,coords);err=np.linalg.norm(blend-obs,axis=1)
    eligible=(err[faces]<=2e-6).all(axis=(1,2))
    rows=[]
    for i,b in enumerate(p['binds']['body']):
        delta=weights[:,i,None]*(anchor-coords[:,i]);alt=hit(obs+delta)
        cleared=int(np.count_nonzero(eligible&~alt));norm=np.linalg.norm(delta,axis=1)
        if cleared:
            rows.append({'body_index':int(b),'body':names.get(int(b)),'eligible_original_witnesses_cleared':cleared,'maximum_change_mm':float(norm.max()*1000)})
    rows.sort(key=lambda x:(-x['eligible_original_witnesses_cleared'],x['maximum_change_mm']))
    out={'step':step,'pack_sha256':receipt['pack_file_sha256'],'self_pairs':len(pairs),'self_pairs_also_present_on_source_faces':int(src.sum()),'source_decode_eligible_pairs':int(eligible.sum()),
         'source_mm_bounds_of_witness_vertices':[list(map(float,source.min(0)*1000)),list(map(float,source.max(0)*1000))],
         'all_donors_to_dominant_rigid_eligible_pairs_remaining':int(np.count_nonzero(eligible&hit(obs+(anchor-blend)))),
         'one_donor_at_a_time':rows,'scope':'Counterfactual on existing native witnesses only. New crossings elsewhere are untested and no physical state or source weights have been changed.'}
    (case/f'donor-isolation-{step}.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]))

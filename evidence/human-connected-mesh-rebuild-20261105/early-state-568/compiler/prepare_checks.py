from pathlib import Path
import json,hashlib,numpy as np
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');D=W/'early-state-screen-568/proposal';OLD=W/'captured-fat-sliver-495/proposal';C=W/'early-state-compiler-570'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
r=json.loads((D/'report.json').read_text());assert r['typed_surface_faces_unchanged'] and r['native_quality_failures']==0
assert r['reference_deformation_screen']['checked_operations']>0
assert r['reference_deformation_screen']['source']['node_mapping']['source_vs_capture_label_disagreements']==19
v=(OLD/'verify-connected-output.py').read_text()
a='with np.load(h/"regional-compensation-1061-root/memory-performance-123/rejection-recovery-125/connected-conditioning-134/fixed-boundary-volume-proposal.npz") as z:baseline=np.concatenate([ot[z["retained_input_cell_rows"]],z["replacement_tetrahedra"]])'
b='with np.load('+repr(str(OLD/'candidate-volume.npz'))+') as z:baseline=z["volume_tetrahedra"]'
assert v.count(a)==1;v=v.replace(a,b)
(D/'verify-connected-output.py').write_text(v)
context=json.loads((OLD/'context-config.json').read_text());context['changed_roles']=list('OID')
for role in 'OID':
 p=D/f'candidate-{role}-typed-current.npz';assert sha(p)==sha(OLD/f'candidate-{role}-typed-current.npz')
 context['candidate_surfaces'][role]={'path':str(p),'sha256':sha(p)}
(D/'context-config.json').write_text(json.dumps(context,indent=2)+'\n')
(D/'surface-identity-reuse.json').write_text(json.dumps(dict(exact_coordinates_and_faces=True,source_volume=str(OLD/'candidate-volume.npz'),source_volume_sha256=sha(OLD/'candidate-volume.npz'),surface_hashes={r:sha(D/f'candidate-{r}-typed-current.npz') for r in 'OID'},physical_admission=False),indent=2)+'\n')
for name in ['plan.json','plan-middle.json','plan-mass.json']:
 p=json.loads((C/name).read_text())
 for stage in p['stages']:stage['source_sha256']=sha(stage['argv'][1])
 (C/name).write_text(json.dumps(p,indent=2)+'\n')
print('EXISTING_COMPILER_BOUND_TO_CURRENT495_SURFACES')

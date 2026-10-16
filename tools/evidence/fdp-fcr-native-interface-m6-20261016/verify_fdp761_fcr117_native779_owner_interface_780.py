from pathlib import Path
import sys,json,hashlib,time,re
import numpy as np
sys.path.insert(0,'/Users/n/numi-human-anatomy-increment-publish-20261016/src')
from numilab_human import resting_anatomy_verify as v,gate_preparation_boundary as gp
D=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378')
O=D/'fdp761-fcr117-native779-owner-and-interface-proof-780'
A=D/'fpl721-brachio109-native746-flip-source-emission-754/native-run/accepted-geometry/step-0.mrvpack'
B=D/'fdp761-fcr117-source778-native-emission-779/native-run/accepted-geometry/step-0.mrvpack'
S=D/'fdp761-fcr117-existing-shared-parent-source-765c';C=D/'fdp731-fcr117-shared-original-source-fdp-priority-761'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not O.exists()
pins=[Path(__file__),A,A.with_suffix('.receipt.json'),B,B.with_suffix('.receipt.json'),D/'fdp761-fcr117-native777-flip-source-778/report.json',D/'fdp761-fcr117-native777-flip-source-778/resting-anatomy-receipt.json',S/'report.json',S/'resting-anatomy-receipt.json',S/'shared-field-inference.json',Path(v.__file__),Path(gp.__file__)]
pins.extend([D/'fdp761-fcr117-native777-flip-source-778/row-133-source.npz',D/'fdp133-fcr117-actual770-paired-local-retessellation-777/report.json',D/'fdp761-fcr117-native777-flip-source-778/report.json'])
for sid in (117,133):pins.extend([S/f'row-{sid}-source.npz',C/f'row-{sid}/candidate.npz'])
pre=gp.freeze_preparation(O.with_suffix('.preparation.json'),dict(status='PASS_PINNED_ACTUAL_SHARED_PARENT_NATIVE_SOURCE_CHECK',predicate_phase_started=False,scope='Require all unrelated typed geometry and accepted physical state bitexact754, own root identity bound to program fingerprint; source geometry may change only117/133. Verify source-local connectivity mapping, allconstructedcommonpoints exact equal across actualGPU rows and retained authored interface, self0closedFCR1/FDP4. Predetermined1um engineering world-emission error bound, no clinical meaning; source/F32 preparation only, allnear/motion/native16/60 remain pending.'),runner_argv=[sys.executable,*sys.argv],input_pins={str(p):sha(p) for p in pins});gp.verify_frozen_preparation(pre);O.mkdir();start=time.monotonic()
a,ai=v.load_capture(A,A.with_suffix('.receipt.json'));b,bi=v.load_capture(B,B.with_suffix('.receipt.json'));assert a.keys()==b.keys();changes={(51005,117),(51005,133)};bad=[]
for key in sorted(a):
 if key not in changes and not all(np.array_equal(a[key][i],b[key][i]) for i in range(3)):bad.append(key)
r1=json.loads(A.with_suffix('.receipt.json').read_text());r2=json.loads(B.with_suffix('.receipt.json').read_text());fields=['accepted_body_state_sha256','accepted_respiration_state_sha256','accepted_body_poses','accepted_registered_body_poses','accepted_respiratory_motion','accepted_step','accepted_time_s'];state={k:r1[k]==r2[k] for k in fields};identity=[]
for pack,receipt in [(A,r1),(B,r2)]:
 text=(pack.parent.parent/'native.log').read_text();match=re.search(r'resting_body_source_fingerprint=(\d+) coupled_program_fingerprint=(\d+)',text);assert match;expected=int(match.group(2))|1;assert int(receipt['accepted_root_fingerprint_hex'],16)==expected;assert int(receipt['accepted_transaction_fingerprint'])==expected;identity.append(dict(body=int(match.group(1)),program=int(match.group(2)),root=expected))
shared={tuple(row['world_point_m']) for row in json.loads((S/'shared-field-inference.json').read_text())};actualshared={};audits=[]
for sid in (117,133):
 with np.load(S/f'row-{sid}-source.npz',allow_pickle=False) as z:sourcefaces=z['faces'].copy();mapping=z['candidate_to_child_source_slot'].copy()
 if sid==133:
  with np.load(D/'fdp761-fcr117-native777-flip-source-778/row-133-source.npz',allow_pickle=False) as z:sourcefaces=z['faces'].copy()
 with np.load(C/f'row-{sid}/candidate.npz',allow_pickle=False) as z:target=z['capture_positions_m'].copy();targetfaces=z['capture_faces'].copy()
 p,f,n,g=b[(51005,sid)];slots,inverse=np.unique(sourcefaces,return_inverse=True);assert np.array_equal(inverse.reshape(-1,3),f),'actual source connectivity differs';slotmap={int(slot):i for i,slot in enumerate(slots)};active=np.unique(targetfaces);assert all(int(mapping[i]) in slotmap for i in active);desired=np.full(target.shape,np.nan,dtype='<f4');desired[active]=p[np.asarray([slotmap[int(mapping[i])] for i in active])];error=np.linalg.norm(desired[active].astype(float)-target[active].astype(float),axis=1);assert error.max()<=1e-6,'source inverse/actual GPU discrepancy exceeds declared engineering bound'
 common={tuple(point.tolist()):desired[i] for i,point in enumerate(target) if i in set(active.tolist()) and tuple(point.tolist()) in shared};assert set(common)==shared;actualshared[sid]=common
 surface=v.prepare_surface((51005,sid),p,f,normals=n,source_vertex_ids=g);audit=surface.self_audit();write(O/f'actual-{sid}-self.json',audit);audits.append(dict(stable_id=sid,self_count=audit['count'],closed=surface.closed,components=surface.quotient_topology['face_component_count'],world_error_max_m=float(error.max()),referenced_candidate_vertices=len(active),allocated_candidate_vertices=len(target),unreferenced_slots_preserved_source_only=True))
different=[list(key) for key in sorted(shared) if not np.array_equal(actualshared[117][key],actualshared[133][key])]
passed=not bad and all(state.values()) and not different and all(row['self_count']==0 and row['closed'] and row['components']==(4 if row['stable_id']==133 else 1) for row in audits)
report=dict(status='PASS_ACTUAL_SHARED_PARENT_NATIVE_SOURCE_AND_INTERFACE' if passed else 'FAIL_ACTUAL_SHARED_PARENT_NATIVE_SOURCE_OR_INTERFACE',unchanged_typed_surfaces=len(a)-2,changed_unrelated_surfaces=bad,accepted_state_bitexact=state,source_identity_bindings=identity,common_points=len(shared),different_common_points=different,audits=audits,source_capture=ai,changed_capture=bi,elapsed_seconds=time.monotonic()-start,anatomical_admission=False,native_qualification=False);write(O/'report.json',report);print(json.dumps(report),flush=True);assert passed

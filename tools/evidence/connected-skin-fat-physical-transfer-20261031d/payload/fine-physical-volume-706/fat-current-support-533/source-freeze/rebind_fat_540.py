from pathlib import Path
import hashlib,json,time
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=ROOT/'fine-physical-volume-706';OUT=H/'fat-current-support-533/run006'
SRCVOL=H/'fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight/volume-transfer-500um-001/compose-D500-I528-comp0-54138-005/fat-D500-I528-comp0-kernel-fan-composed-volume-proposal.npz'
VOL=H/'fat-canonical-current-target-540-root/fat-current-D500-I528-O530-canonical-volume-proposal.npz';VOLREC=H/'fat-canonical-current-target-540-root/receipt.json'
BASE=H/'fat-current-support-533/run003/fat-current-semantic-support-d899-p1.npz';FULL=H/'fat-current-support-533/run003/fat-current-semantic-support-d899-p1-143.npz';SRCREC=H/'fat-current-support-533/run003/receipt.json';REPLAY=H/'fat-current-support-533/run005/independent-replay.json'
PIN={SRCVOL:'d899b2789c0d4fc91553a5d46b1bb8d7b40e9c818d6301f6b32a3265976e3703',BASE:'8e9cb1094a8cd4d12d3155c45b985bb4eb607ede7a64716850f071520d76dd62',FULL:'4aa03ccf8e3b798ba36cd751d2ea61527e6a04f3f41c316ee291a75fbfb9622c',REPLAY:'1ffe48572b45d1e5bc33fb11c6f041782f6fd65d88eadba9fe0346b091cc11c2'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def main():
 t=time.monotonic()
 if OUT.exists():raise FileExistsError(OUT)
 for p,h in PIN.items():
  if sha(p)!=h:raise ValueError(f'input pin mismatch {p}')
 vr=json.loads(VOLREC.read_text());sr=json.loads(SRCREC.read_text());rr=json.loads(REPLAY.read_text())
 if vr.get('status')!='PASS_CANONICAL_CURRENT_FAT_TARGET_METADATA_AND_PARENT_UNIONS' or vr.get('candidate_sha256')!=sha(VOL):raise ValueError('540 metadata receipt invalid')
 if sr.get('physical_admission') is not False or rr.get('status')!='PASS_INDEPENDENT_CURRENT_FAT_P1_AND_ONE_HOP_REPLAY':raise ValueError('source P1/replay not qualified')
 x=load(SRCVOL);y=load(VOL);base=load(BASE);full=load(FULL)
 keys=('points_world_m','points_world_m_f32','tetrahedra','material_indices')
 checks={}
 for k in keys:
  if k not in x or k not in y:raise ValueError(f'missing geometric dependency {k}')
  checks[k]={'equal':bool(x[k].dtype==y[k].dtype and x[k].shape==y[k].shape and np.array_equal(x[k],y[k])),'shape':list(x[k].shape),'dtype':str(x[k].dtype)}
  if not checks[k]['equal']:raise ValueError(f'current target differs from P1 source geometry in {k}')
 if len(y['tetrahedra'])!=798672 or not np.all(y['material_indices']==16):raise ValueError('current Fat geometry counts/material changed')
 if str(base['geometry_sha256'].item())!=sha(SRCVOL) or str(full['geometry_sha256'].item())!=sha(SRCVOL):raise ValueError('source support geometry pin mismatch')
 # Rebinding only the outer file identity is valid because all geometry-bearing P1 inputs are byte-identical.
 base['geometry_sha256']=np.array(sha(VOL));OUT.mkdir()
 base_out=OUT/'fat-current-semantic-support-540-p1.npz';np.savez_compressed(base_out,**base);newbase=sha(base_out)
 full['geometry_sha256']=np.array(sha(VOL));full['admissibility_source_support_sha256']=np.array(newbase)
 full_out=OUT/'fat-current-semantic-support-540-p1-143.npz';np.savez_compressed(full_out,**full)
 with np.load(base_out,allow_pickle=False)as z:
  if any(not np.array_equal(z[k],v) for k,v in base.items()):raise ValueError('base P1 serialization mismatch')
 with np.load(full_out,allow_pickle=False)as z:
  if any(not np.array_equal(z[k],v) for k,v in full.items()):raise ValueError('143 serialization mismatch')
 # Bind corrected current metadata and role sidecars in the receipt; D/I rebind rows are not P1 dependencies.
 report={'schema':'numi.human.current-fat-p1-143-canonical-540-rebind.v1','status':'PASS_EXACT_GEOMETRY_REBIND_OF_INDEPENDENT_P1_143_SUPPORT','physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN','inputs':{'p1_source_geometry':{'path':str(SRCVOL),'sha256':sha(SRCVOL)},'canonical_current_geometry':{'path':str(VOL),'sha256':sha(VOL)},'canonical_current_geometry_receipt':{'path':str(VOLREC),'sha256':sha(VOLREC)},'d899_p1_support':{'path':str(BASE),'sha256':sha(BASE)},'d899_p1_143_support':{'path':str(FULL),'sha256':sha(FULL)},'d899_support_receipt':{'path':str(SRCREC),'sha256':sha(SRCREC)},'independent_d899_replay':{'path':str(REPLAY),'sha256':sha(REPLAY)}},'geometry_dependency_equality':checks,'metadata_authority':{'typed_D_sha256':'9e20be647417adefd13421603850f29e364e83ce0cfd49b668404b2c5011f04d','typed_I_sha256':'6f59262c2ae70065576166dd84f767afb64352241e76009707f78447afe477d1','typed_O_sha256':'2f98f16f06baaefada41d4577594923ff5e9019a84a5e1c7f6cb44a1171f8901','note':'Canonical 540 metadata is authoritative. Its geometry/material are identical to d899; d899 metadata fields for D/I target row maps are stale and are not used by this support rebind.'},'p1_replay_inherited':{'status':rr['status'],'receipt_sha256':sha(REPLAY),'cells':rr['counts']['cells'],'nodes':rr['counts']['nodes'],'p1_group_slots':rr['counts']['P1_group_slots'],'ties':rr['counts']['tie_cells'],'admissible_region_slots':rr['counts']['admissible_region_slots']},'outputs':{'base_p1':{'path':str(base_out),'sha256':newbase,'size_bytes':base_out.stat().st_size},'p1_plus_143':{'path':str(full_out),'sha256':sha(full_out),'size_bytes':full_out.stat().st_size,'admissibility_source_support_sha256':newbase}},'limitations':['This is a byte-exact geometry rebind of the independently replayed d899 P1/143 arrays; canonical 540 metadata and its typed sidecar pins are separately authoritative.','No mass allocation, native installation or physical admission.'],'elapsed_s':time.monotonic()-t}
 rp=OUT/'receipt.json';rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');report['receipt_sha256']=sha(rp);print(json.dumps(report,sort_keys=True),flush=True)
if __name__=='__main__':main()

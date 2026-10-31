from pathlib import Path
import hashlib,json,time
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
OUT=H/'skin-current-support-532/run004'
SRC=H/'skin-current-support-532/run002/skin-current-semantic-support-531-p1-143.npz'
SRC_RECEIPT=H/'skin-current-support-532/run002/receipt.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir()
 start=time.monotonic()
 with np.load(SRC,allow_pickle=False)as z:a={k:z[k] for k in z.files}
 ext={'cell_admissible_region_offsets','cell_admissible_region_ids','admissibility_source_support_sha256','admissibility_max_source_region_hops'}
 if not ext.issubset(a):raise ValueError('required143 extension missing')
 base={k:v for k,v in a.items() if k not in ext}
 base_path=OUT/'skin-current-semantic-support-531-p1.npz'
 np.savez_compressed(base_path,**base);base_sha=sha(base_path)
 derived=dict(base)
 derived['cell_admissible_region_offsets']=a['cell_admissible_region_offsets']
 derived['cell_admissible_region_ids']=a['cell_admissible_region_ids']
 derived['admissibility_source_support_sha256']=np.array(base_sha)
 derived['admissibility_max_source_region_hops']=np.array(1,np.int32)
 derived_path=OUT/'skin-current-semantic-support-531-p1-143.npz'
 np.savez_compressed(derived_path,**derived)
 with np.load(base_path,allow_pickle=False)as z:
  for k,v in base.items():
   if not np.array_equal(z[k],v):raise ValueError(f'base-P1 roundtrip mismatch {k}')
 with np.load(derived_path,allow_pickle=False)as z:
  if str(z['admissibility_source_support_sha256'].item())!=base_sha:raise ValueError('143 source hash mismatch')
  if not np.array_equal(z['cell_admissible_region_ids'],a['cell_admissible_region_ids']):raise ValueError('region IDs changed')
  if not np.array_equal(z['cell_admissible_region_offsets'],a['cell_admissible_region_offsets']):raise ValueError('region offsets changed')
 source_report=json.loads(SRC_RECEIPT.read_text())
 report={'schema':'numi.human.current-skin-p1-and-143-support-artifacts.v1','status':'PASS_SOURCE_ONLY_CURRENT_SKIN_P1_AND_AUTHENTICATED_ONE_HOP_SUPPORT','physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN','inputs':{'combined_p1_143_candidate':{'path':str(SRC),'sha256':sha(SRC)},'producer_receipt':{'path':str(SRC_RECEIPT),'sha256':sha(SRC_RECEIPT)},'source_producer':source_report.get('inputs',{})},'outputs':{'base_p1':{'path':str(base_path),'sha256':base_sha,'size_bytes':base_path.stat().st_size},'p1_plus_authenticated_143':{'path':str(derived_path),'sha256':sha(derived_path),'size_bytes':derived_path.stat().st_size,'admissibility_source_support_sha256':base_sha,'max_source_region_hops':1}},'counts':source_report['counts'],'limitations':source_report['limitations']+['This repack separates the base P1 support hash from the 143 one-hop extension; no new geometry or ancestry is created.'],'elapsed_s':time.monotonic()-start}
 rpath=OUT/'receipt.json';rpath.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':report['status'],'base_p1':report['outputs']['base_p1'],'derived':report['outputs']['p1_plus_authenticated_143'],'receipt_path':str(rpath),'receipt_sha256':sha(rpath),'elapsed_s':report['elapsed_s']},sort_keys=True),flush=True)
if __name__=='__main__':main()

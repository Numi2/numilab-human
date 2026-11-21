import json,sys,shutil,hashlib
from pathlib import Path
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def prepare(source,asset,run,dst):
 cfg=json.loads((source/'config.json').read_text());state=json.loads((asset/'registration.json').read_text())
 bone=Path(next(arg for arg in cfg['argv'] if arg.endswith('.nhbones')));reg=asset/'digit-cortical-features.bin'
 if cfg['pins'][str(bone)]!=state['bone_sha256'] or sha(reg)!=state['payload_sha256']:raise ValueError('Source registration identity changed')
 cfg['pins'][str(reg)]=sha(reg);cfg['pins'][str(asset/'registration.json')]=sha(asset/'registration.json')
 surface=asset/'proximal-surface.bin'
 if surface.exists():
  meta=asset/'proximal-surface.json';m=json.loads(meta.read_text())
  if m['bone_sha256']!=state['bone_sha256'] or sha(surface)!=m['sha256']:raise ValueError('Cortical surface provenance differs')
  cfg['pins'][str(surface)]=sha(surface);cfg['pins'][str(meta)]=sha(meta)
 cfg['environment']['NUMI_HUMAN_DIGIT_CORTICAL_FEATURES']=str(reg);cfg['environment']['NUMI_HUMAN_DIGIT_CONTACT_TRACE']=str(run)
 dst.mkdir(exist_ok=True);shutil.copyfile(source/'run_supported.py',dst/'run_supported.py')
 (dst/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))

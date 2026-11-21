"""Run an existing supported physical scene for 16 or 60 s and request only its actual terminal pack."""
import json,subprocess,sys
from pathlib import Path
from material_context import sha
def configure(cfg,out,steps):
    if steps not in (8000,30000):raise ValueError('Unsupported qualification interval')
    conf=json.loads(cfg.read_text())
    if conf['headless'] or conf['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1':raise ValueError('Unexpected scene')
    if conf['environment'].get('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS')!='8000':raise ValueError('Unexpected source capture')
    conf['environment']['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=str(steps)
    conf['pins'][str(cfg.resolve())]=sha(cfg)
    out.write_text(json.dumps(conf,indent=2)+'\n')
if __name__=='__main__':configure(Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]))

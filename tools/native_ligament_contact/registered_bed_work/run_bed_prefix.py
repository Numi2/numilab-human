"""Run a short native release on each side of the first material-domain crossing."""
import argparse, json
from pathlib import Path
from run_bed_work import digest,run

def prefix(source,owner,out,steps):
    cfg=json.loads((source/'launch.json').read_text())
    if not 1<=steps<=32:raise ValueError("use a single native segment")
    parent=out.with_name(out.name+'-source');parent.mkdir()
    cfg['environment']['NUMI_HUMAN_RESOLVED_BED_WITNESS']=str(parent/'witness.json')
    cfg['parent_launch']=digest(source/'launch.json')
    (parent/'launch.json').write_text(json.dumps(cfg,indent=2,allow_nan=False)+'\n')
    run(parent,owner,out,steps,enabled=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','owner','out'):p.add_argument(k,type=lambda s:Path(s).resolve())
    p.add_argument('steps',type=int)
    prefix(**vars(p.parse_args()))

"""Carry the source repair into the previously verified supported native launch."""
import json,os,shutil,sys
from pathlib import Path
from material_context import sha
from bind_posterior_receipt import run as bind
def run(baseline,candidate,output,runner):
    output.mkdir(exist_ok=True);cfg=json.loads(baseline.read_text());a=cfg['argv']
    at=a.index('--soft-tissue-payload')+1;old=Path(a[at])
    lineage=json.loads((candidate/'source.json').read_text())
    if sha(old)!=lineage['parent_sha256']:raise ValueError('Unsupported NHTISSUE roster')
    for name in [old.name,old.with_suffix('.manifest.json').name,'source.json']:
        dest=output/name
        if not dest.exists():os.link(candidate/name,dest)
    cfg['hashes']={str(baseline):sha(baseline)}
    count=a.index('--muscle-step-count')+1
    if a[count]!='30000' or cfg['headless']:raise ValueError('Expected previous 60 second native control')
    a[count]='8000';a[at]=str((output/old.name).resolve())
    cfg['environment']['NUMI_HUMAN_ACCEPTED_BODY_MOTION_AUDIT']='1'
    cfg['environment']['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']='8000'
    # The native source pack is built in this trial, so no incompatible cache is opened.
    if 'NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE' in cfg['environment']:raise ValueError('Unexpected warm-cache configuration')
    cfg['pins'][a[at]]=sha(Path(a[at]));cfg['pins'][str(baseline)]=sha(baseline)
    f=output/'config.json';f.write_text(json.dumps(cfg,indent=2)+'\n')
    bind(output)
    cfg=json.loads(f.read_text());cfg['pins'].update(cfg.pop('hashes'));f.write_text(json.dumps(cfg,indent=2)+'\n')
    text=runner.read_text()
    at="cfg['pins'].pop(env.pop('NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE'))\nenv.pop('NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256')"
    if text.count(at)!=1:raise ValueError('Native runner changed')
    text=text.replace(at,"if 'NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE' in env:\n    cfg['pins'].pop(env.pop('NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE'))\n    env.pop('NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256')")
    text=text.replace("'resting-coupled.csv','resting-local-bed-sections.csv'","'resting-body-motion-partition.csv','resting-coupled.csv','resting-local-bed-sections.csv'")
    (output/'run_supported.py').write_text(text)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))

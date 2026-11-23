import argparse,json
from pathlib import Path
from run_bed_work import run, digest as sha
p=argparse.ArgumentParser()
p.add_argument('parent',type=Path);p.add_argument('steps',type=int);p.add_argument('out',type=Path)
p.add_argument('--owner',default='owner');a=p.parse_args();root=Path(__file__).resolve().parent
out=a.out.resolve(); source=out.with_name(out.name+'-source');source.mkdir()
cfg=json.loads((a.parent/'launch.json').read_text()); foundation=(root/a.owner).parent/'foundation.json';cfg['pins'][str(foundation)]=sha(foundation)
cfg['environment']['NUMI_HUMAN_RESOLVED_BED_WITNESS']=str(source/'witness.json')
(source/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
run(source,root/a.owner,out,a.steps,enabled=False,foundation=foundation)

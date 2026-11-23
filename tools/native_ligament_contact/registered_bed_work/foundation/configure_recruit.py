from pathlib import Path
import json,sys
from material_context import sha
base,out=map(Path,sys.argv[1:]);out.mkdir()
cfg=json.loads((base/'launch.json').read_text());cfg['environment']['NUMI_HUMAN_RECRUIT_AT_FOUNDATION_LOAD']='1'
cfg['pins'][str(out.parent/'opening-muscle-load.json')]=sha(out.parent/'opening-muscle-load.json')
(out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')

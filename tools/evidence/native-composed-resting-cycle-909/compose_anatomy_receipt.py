from pathlib import Path
import argparse,sys,json
p=argparse.ArgumentParser()
p.add_argument('--human-source',type=Path,required=True)
p.add_argument('--base-receipt',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
sys.path.insert(0,str(a.human_source/'src'))
from numilab_human.resting_anatomy import compose_skin_binding_candidate
d=Path('/Users/n/numi-human-resting-evidence-20261005/common-atlas-skin-composition-907')
r=compose_skin_binding_candidate(a.base_receipt,d/'bodyparts3d-myosim-skinned-shell.nhskin',d/'common-atlas-skin-geometry-registration.manifest.json',a.output)
print(json.dumps(r,indent=2))

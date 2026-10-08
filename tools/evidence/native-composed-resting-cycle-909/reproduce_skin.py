from pathlib import Path
import argparse,hashlib,json,sys
parser=argparse.ArgumentParser()
parser.add_argument('--human-source',type=Path,required=True,help='Human source containing commit f8b2e3b or its unchanged composition owner')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.human_source/'src'))
from numilab_human.common_atlas_skin_geometry_registration import compose_disjoint_skin_position_corrections
e=Path('/Users/n/numi-human-resting-evidence-20261005')
base=e/'common-atlas-skin-registration-003/bodyparts3d-myosim-skinned-shell.nhskin'
body=e/'common-atlas-skin-registration-018/candidate/bodyparts3d-myosim-skinned-shell.nhskin'
eye=e/'ocular-skin-eye-registration-906-feasible-iterate-geometry-audit/bodyparts3d-myosim-skinned-shell.eye-registration-candidate.nhskin'
manifest=json.loads(base.with_name('common-atlas-skin-geometry-registration.manifest.json').read_text())
raw,report=compose_disjoint_skin_position_corrections(base.read_bytes(),[body.read_bytes(),eye.read_bytes()],global_source_matrix=manifest['common_atlas_binding_runtime_rest_validation']['global_source_mm_to_world_m'])
assert hashlib.sha256(raw).hexdigest()=='898990d49a3f1dfa0cbe85b10765bf62fa0cc3c834a3371f5503c83facfacba2'
assert not args.output.exists()
args.output.write_bytes(raw)
print(json.dumps({'output':str(args.output),'sha256':hashlib.sha256(raw).hexdigest()}))

"""Regress the mismatched registration pose and qualify the newly observed initial pose."""
import json,runpy,sys
from pathlib import Path
import numpy as np
from material_context import sha

def verify(p,src):
    def read(n):return json.loads((p/n).read_text())
    fn=runpy.run_path(str(src))['release_root_error']
    old=read('registered-initial-bodies.json')['all_bodies'];new=read('native-initial-bodies.json')['all_bodies']
    if [s['body_index'] for s in old]!=list(range(157)) or [s['body_index'] for s in new]!=list(range(157)):raise ValueError('Incomplete poses')
    source=fn(old,p/'stdout.log');actual=fn(new,p/'stdout.log')
    if not .0117<source<.0118 or not 0<=actual<=2e-6:raise ValueError('Pose regression')
    x=np.array([s['current']['position_world_m'] for s in old]);y=np.array([s['current']['position_world_m'] for s in new]);delta=y-x
    return dict(source_registration_root_error_m=source,executed_initial_root_error_m=actual,
                median_body_translation_m=np.median(delta,axis=0).tolist(),maximum_body_translation_m=float(np.linalg.norm(delta,axis=1).max()),
                bodies=len(new),source_registration=sha(p/'registered-initial-bodies.json'),executed_initial=sha(p/'native-initial-bodies.json'),
                launch=sha(p/'launch.json'),solver_stdout=sha(p/'stdout.log'),shared_validator=sha(src))
if __name__=='__main__':print(json.dumps(verify(*(Path(v) for v in sys.argv[1:])),indent=2))

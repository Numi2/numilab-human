"""Make a matched one-millisecond control with the same native registration and owners."""
import json,sys
from pathlib import Path
from material_context import sha
base,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=False)
d=json.loads((base/'config.json').read_text());args=d['argv'];i=args.index('--muscle-step-seconds')+1
if args[i]!='0.002':raise ValueError('Expected the two millisecond reference')
args[i]='0.001'
d['qualified_step_comparison']={'original':sha(base/'config.json'),'step_seconds':.001,'new_duration_is_steps_times_native_dt':True}
(out/'config.json').write_text(json.dumps(d,indent=2)+'\n')
s=(base/'run_supported.py').read_text();key="!='0.002': raise ValueError('Mismatched native step')"
if s.count(key)!=1:raise ValueError('Original driver step guard')
(out/'run_supported.py').write_text(s.replace(key,"!='0.001': raise ValueError('Mismatched native step')"))

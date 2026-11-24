"""Compare native motion for the same registered tissues after a source-preserving cavity change."""
import csv,json,sys
from pathlib import Path
import numpy as np
from check_full_motion import run as check
from build_bed_audit import sha
def qualify(case,stage,ancestor,control,mesh,out):
    result=out.parent/'coupled-motion.json';check(case,stage,result);prior=json.loads(ancestor.read_bytes());now=json.loads(result.read_bytes());ov=np.load(mesh/'overlay.npz');mapping=ov['old_to_new']
    roles={}
    for role,b in now['roles'].items():
        originals=np.asarray(prior['roles'][role]['failing_cells'],dtype='i4');bad=set(b['failing_cells']);survive={int(mapping[x])for x in originals if mapping[x]>=0}
        roles[role]=dict(prior_failed=len(originals),native_failed=len(bad),removed_original_rows=[int(x)for x in originals if mapping[x]<0],prior_failures_cleared=sorted(survive-bad),new_failing_rows=sorted(bad-survive),inverted=b['inverted'])
    def row(d):
        with (d/'native-traces/resting-coupled.csv').open()as f:return list(csv.DictReader(f))[-1]
    a,b=row(control),row(case);fields=('time_s','pleural_pa','lv_ml','PaO2_mmhg','PaCO2_mmhg','peak_penetration_m','blood_error_ml','root_assistance_n','root_assistance_nm')
    compare={k:[float(z[k]) for z in (a,b)] for k in fields if k in a}
    if now['accepted_step']!=prior['accepted_step'] or not np.isfinite(list(compare.values())).all():raise ValueError('Unmatched physics observation')
    record=dict(uid=sha(case/'launch.json'),executable=sha(Path(json.loads((case/'launch.json').read_bytes())['argv'][0])),execution=now['execution'],physical_time=now['accepted_s'],roles=roles,surface_crossings=now['exact_material_surface_pair_count'],compared=compare,transfer=sha(mesh/'transfer.json'),min_interval=json.loads((mesh/'transfer.json').read_bytes())['minimum_native_interval_margin'])
    out.write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
if __name__=='__main__':qualify(*(Path(v).resolve()for v in sys.argv[1:]))

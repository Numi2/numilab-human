"""Distinguish solver acceptance, the release-relative terminal gate and rollback."""
import json,re,sys
from pathlib import Path
from material_context import sha

def check(case):
    def read(p):return json.loads(p.read_text())
    a=case/'coupled/three'
    text=(a/'stderr.log').read_text()
    w=re.search(r'^resting_fem_terminal_reject .*?global_tet=(\d+) .*? J=([^ ]+) rest_det_m3=([^ ]+) current_det_m3=([^ ]+)',text,re.M)
    if not w:raise ValueError('Missing independent terminal witness')
    j,initial,end=map(float,w.group(2,3,4))
    if abs(end/initial-j)>2e-6:raise ValueError('Inconsistent captured signed volumes')
    td=read(case.parent/'contact-pockets/material-pockets.json')
    f=[c for c in td['joined_interface_rejections'] if c['material']==16][0]
    pre=next(row for comp in f['failed_components'] for row in comp['witnesses'] if row['source_row']==1032163)
    if pre['reference_m3']<=0 or w[1]!='1035269':raise ValueError('Different physical witness')
    ratio=initial/(6*pre['reference_m3']);material=end/(6*pre['reference_m3'])
    control=read(case.parent/'fold-repair/compare.json')
    record=dict(source=sha(a/'stderr.log'),native_cell=int(w[1]),source_cell=1032163,physical_reference_m3=pre['reference_m3'],
                loaded_to_material_volume_ratio=ratio,trial_to_material_volume_ratio=material,trial_to_loaded_volume_ratio=j,
                earlier_mechanical_threshold=0.0000125,dense_prefix_geometry=control['after']['endpoint'])
    verified=read(case/'comparison.json')
    for idx,key in [(-2,'certificate_only'),(-1,'direction_and_certificate')]:
        r=verified[idx];out=dict(r)
        if not r['physical_prefix_loads_equal_to_control'] or not r['physical_prefix_trace_equal_to_control']:
            out['identical_prefix']=False
        if 'terminal_rejection' in r:raise ValueError('Native clock still advanced through a terminal-invalid state')
        if r['failed'] and not (r['rollback_common'] and r['rollback_respiratory']):raise ValueError('Incomplete accepted-state rollback')
        record[key]=out
    cert=record['certificate_only']
    assert cert['failed'] and cert['accepted_steps']==2
    if not (cert['physical_prefix_loads_equal_to_control'] and cert['physical_prefix_trace_equal_to_control']):raise ValueError('Certificate-only replay changed accepted physics')
    fail=read(case/'release/three-1/failure.json')
    assert fail['matter_failing_index']==int(w[1]) and fail['matter_diagnostics'][3]==-913
    if abs(fail['matter_diagnostics'][0]-j)>1e-5:raise ValueError('GPU and terminal geometry disagree')
    return record
if __name__=='__main__':print(json.dumps(check(Path(sys.argv[1]).resolve()),indent=2,allow_nan=False))

"""Check the physical prefix and rollback of the extended native release."""
import csv,json,re,sys
from pathlib import Path
from material_context import sha

def read(p):return json.loads(p.read_text())

def check(base):
    p=base/'six';old=base/'three';j=read(p/'launch.json');control=read(old/'launch.json')
    for k in ('NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE','DYLD_LIBRARY_PATH','NUMI_MATTER_METALLIB_OVERRIDE'):
        if j['environment'][k]!=control['environment'][k]:raise ValueError('Unmatched fitted release or physics')
    out=read(p/'execution.json');force=lambda r:[l for l in (r/'stdout.log').read_text().splitlines() if l.startswith('accepted_fem_body_load_capture=accepted ')]
    common=force(old);full=force(p)
    trace=lambda r:(r/'native-traces/resting-coupled.csv').read_text().splitlines()
    n=len(full)
    if common[:n]!=full[:len(common)] or trace(old)[:n+1]!=trace(p)[:min(n,len(common))+1]:
        raise ValueError('Observer changed the accepted physical prefix')
    data=dict(accepted=n,prefix_identical=min(n,len(common)),launch=sha(p/'launch.json'),pose=sha(p/'native-initial-bodies.json'),wall_s=out['wall_s'])
    if out['exit']:
        f=read(p/'failure.json')
        ok=f['candidate_common_coordinates']==f['accepted_common_coordinates'] and f['respiratory_candidate_status']==f['respiratory_accepted_status']
        if not ok or f['segment_start']!=n:raise ValueError('Unmatched rejected state')
        data.update(rejection_code=f['matter_code'],object=f['matter_object_index'],index=f['matter_failing_index'],diagnostics=f['matter_diagnostics'],rollback=True)
    else:
        r=read(p/'terminal.receipt.json')
        if r['accepted_step']!=6 or n!=6 or r['physical_endpoint']!='accepted':raise ValueError('Missing requested endpoint')
    clocks=list(csv.DictReader((p/'native-traces/resting-coupled.csv').open()))
    data['accepted_s']=float(clocks[-1]['time_s']) if clocks else 0.
    data['end_to_end_factor']=data['accepted_s']/out['wall_s']
    return data
if __name__=='__main__':print(json.dumps(check(Path(sys.argv[1]).resolve()),indent=2))

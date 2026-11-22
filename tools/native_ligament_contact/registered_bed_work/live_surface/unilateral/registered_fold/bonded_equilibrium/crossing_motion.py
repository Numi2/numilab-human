"""Match exact physical triangle-pair witnesses at two native accepted endpoints."""
from pathlib import Path
import json
import sys
from material_context import sha

def read(p):return json.loads(p.read_text())

def witnessed(run):
    e=run/'envelope';stats=read(e/'physical-banks.json')
    frame=read(run/'terminal.receipt.json');state=read(run/'verified-endpoint.json')
    if frame['physical_endpoint']!='accepted' or state['frame_sha256']!=frame['pack_file_sha256'] or stats['pack_sha256']!=state['frame_sha256']:
        raise ValueError('Endpoint identities disagree')
    ocular=read(run/'native-traces/resting-ocular-sclera-window.json')
    display={r['derived_skin_identity']['stable_id'] for r in ocular['windows']+ocular['posterior_lens_backings']}
    out=set()
    for l in (e/'target-witnesses.jsonl').read_text().splitlines():
        r=json.loads(l);i=r['stable_id'];sem=r['semantic']
        if sem==51010 and (i in display or 381<=i<=397):continue
        if r['count']!=len(r['triangle_pairs']):raise ValueError('Incomplete exact witnesses')
        out.update((sem,i,r['primitive'],a,b) for a,b in r['triangle_pairs'])
    if len(out)!=stats['physical_nonocular_pairs'] or stats['checked_banks']!=878:raise ValueError('Incomplete bank coverage')
    return frame,out

def compare(first,second):
    a,left=witnessed(first);b,right=witnessed(second)
    if (a['base_pack_file_sha256']!=b['base_pack_file_sha256'] or b['accepted_step']!=a['accepted_step']+1
        or a['accepted_time_s']>=b['accepted_time_s']):raise ValueError('Unmatched sources or time interval')
    one=read(first/'launch.json');two=read(second/'launch.json')
    k='NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE'
    if one['argv'][0]!=two['argv'][0] or one['environment'][k]!=two['environment'][k]:
        raise ValueError('Different pose or native owner')
    f=lambda r:[l for l in (r/'stdout.log').read_text().splitlines() if l.startswith('accepted_fem_body_load_capture=accepted ')]
    if f(first)!=f(second)[:a['accepted_step']]:raise ValueError('Unmatched replay loads')
    traces=lambda r:(r/'native-traces/resting-coupled.csv').read_text().splitlines()
    if traces(first)!=traces(second)[:len(traces(first))]:raise ValueError('Unmatched physiological history')
    return dict(first_step=a['accepted_step'],last_step=b['accepted_step'],interval_s=b['accepted_time_s']-a['accepted_time_s'],
        first_sha256=a['pack_file_sha256'],last_sha256=b['pack_file_sha256'],banks=878,start=len(left),end=len(right),
        persisted=len(left&right),cleared=len(left-right),new=len(right-left),new_witnesses=sorted(right-left)[:8],
        cleared_witnesses=sorted(left-right)[:8],common_physiology_and_skeletal_load_prefix=True)
if __name__=='__main__':print(json.dumps(compare(*(Path(p).resolve() for p in sys.argv[1:])),indent=2))

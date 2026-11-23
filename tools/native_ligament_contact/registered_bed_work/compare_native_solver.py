"""Verify accepted mechanics and physiology on an exact cached-operator replay."""
import argparse,json,re
from pathlib import Path
from registered_forearms import read,sha
def check(control,case,simd_sweep=False):
    a,b=[read(p/'launch.json') for p in (control,case)]
    changed={k:(a['environment'].get(k),b['environment'].get(k)) for k in a['environment'].keys()|b['environment'].keys() if a['environment'].get(k)!=b['environment'].get(k)}
    flags=('NUMI_HUMAN_STAND_FIRST_SIMD_CONTACT_SWEEP',) if simd_sweep else ('NUMI_HUMAN_STAND_CACHE_CONTACT_JACOBIAN','NUMI_HUMAN_STAND_ZERO_CONTACT_RESPONSE_FASTPATH')
    for k in flags:
        if changed.pop(k,None)!=('0','1'):raise ValueError('Different solver experiment')
    changed.pop('NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT',None)
    if changed or a['pins']!=b['pins'] or a['argv'][:4]!=b['argv'][:4] or a['argv'][5:]!=b['argv'][5:]:raise ValueError('Different matched conditions '+str(changed))
    endpoints=[read(p/'terminal.receipt.json') for p in (control,case)]
    keys=('accepted_step','accepted_time_s','accepted_body_state_sha256','accepted_respiration_state_sha256','pack_file_sha256','base_pack_file_sha256','captured_vertex_buffer_sha256')
    equal={k:endpoints[0][k]==endpoints[1][k] for k in keys}
    def cardiac(receipt):
        payload={k:{p:v for p,v in d.items() if p!='path'} for k,d in receipt['accepted_cardiac_geometry_buffers'].items()}
        if not payload or any(len(v['sha256'])!=64 or v['bytes']<=0 for v in payload.values()):raise ValueError('No certified cardiac buffers')
        return payload
    equal['accepted_cardiac_geometry_buffers']=cardiac(endpoints[0])==cardiac(endpoints[1])
    if (control/'step-3999.receipt.json').exists():
        mid=[read(d/'step-3999.receipt.json') for d in (control,case)]
        equal['middle_geometry_and_cardiac_state']=all(mid[0][k]==mid[1][k] for k in keys) and cardiac(mid[0])==cardiac(mid[1])
    traces={n:sha(control/n)==sha(case/n) for n in ('resting-coupled.csv','resting-com-support-impulses.csv','resting-surface-audit.csv','resting-support-selected-skin-witness.csv','resting-local-bed-sections.csv','resting-com-momentum-diagnostic.csv')}
    timing=[]
    for p,r in zip((control,case),endpoints):
        ex=read(p/'execution.json')
        if ex['exit_code']!=0 or r['physical_endpoint']!='accepted':raise ValueError('Unaccepted trajectory')
        out=(p/'stdout.log').read_text();parts={}
        for k in ('resting_integrated_throughput','resting_integrated_body=completed'):
            line,=re.findall('^'+k+' .*$',out,re.M);kv=dict(re.findall(r'(\w+)=([^ ]+)',line))
            parts[k]=dict(wall_s=float(kv['wall_s']),factor=float(kv['real_time_factor']))
        parts['cold']=dict(wall_s=ex['cold_wall_s'],factor=r['accepted_time_s']/ex['cold_wall_s'])
        timing.append(parts)
    return dict(control=sha(control/'launch.json'),trial=sha(case/'launch.json'),seconds=endpoints[-1]['accepted_time_s'],steps=endpoints[-1]['accepted_step'],endpoints_equal=equal,traces_equal=traces,exact=all(equal.values()) and all(traces.values()),timing=timing)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('control','case'):p.add_argument(k,type=Path)
    p.add_argument('--simd-sweep',action='store_true')
    r=check(**vars(p.parse_args()));print(json.dumps(r,indent=2))
    raise SystemExit(0 if r['exact'] else 1)

"""Resolve repeated stencils from typed native records and relate contact assembly to the Newton residual."""
from pathlib import Path
from collections import Counter,defaultdict
import json,sys

def read_events(path,name):
    return [json.loads(line.split('=',1)[1]) for line in (path/'stderr.log').open() if line.startswith(name+'=')]

def features(root,case):
    events=read_events(root/case,'fem_deformable_limit_trace')
    row=next(e for e in events if e['stage']=='after_deformable_limit')
    assert not any(row['truncated']) and row['recorded_count']==row['active_counts'][0]
    groups=defaultdict(list)
    for r in row['contacts']:
        a,b,k,flags=r['identity'];n=r['nodes'][:4]
        if a==b or flags!=1 or len(set(n))!=4:continue
        if k==0:key=(a,b,k,n[0],*sorted(n[1:]))
        elif k==1:key=(a,b,k,*sorted(n[:2]),*sorted(n[2:]))
        else:continue
        groups[key].append(r)
    details={}
    for kind in (0,1):
        dup=[v for k,v in groups.items()if k[2]==kind and len(v)>1]
        details[str(kind)]=dict(alias_classes=len(dup),participating_records=sum(map(len,dup)),largest=max(map(len,dup),default=0))
    witnesses=[{k:r[k] for k in ('nodes','identity','source','weights','contact_slot','barrier_parameters','contact_point_and_separation')} for r in row['contacts']if 1176 in r['nodes']]
    return dict(active=row['active_counts'][0],aliases=details,at_internal_oblique_edge=witnesses)
def summary(root):
    events={}
    for case in ('rhs-99','edge-owned-128','birth-160'):
        report=read_events(root/case,'fem_contact_rhs_witness')
        events[case]=[{k:r[k] for k in ('root','iteration','pre_contact_xyz_l2','post_contact_xyz_l2','contact_xyz_l2')}for r in report]
    return dict(before=features(root,'features-start-99'),after=features(root,'features-99'),assembly=events)
if __name__=='__main__':
    p=Path(sys.argv[1]);(p/'feature-evidence.json').write_text(json.dumps(summary(p),indent=2)+'\n')

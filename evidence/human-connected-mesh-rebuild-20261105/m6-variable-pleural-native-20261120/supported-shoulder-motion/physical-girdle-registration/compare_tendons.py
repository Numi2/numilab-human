"""Compare individual physical tendon endpoints, material quadrature modes, and source routes."""
import json,sys
from pathlib import Path
def compare(baseline,candidate,out):
    old,new=[json.loads(p.read_text()) for p in (baseline,candidate)]
    affected={29,34,79,84};rows=[]
    assert len(old['endpoints'])==len(new['endpoints'])
    for a,b in zip(old['endpoints'],new['endpoints']):
        for k in ('body_index','endpoint','muscle','muscle_index','route_node_index','source_local_point_m','source_site_index'):
            assert a[k]==b[k],k
        if a!=b:
            rows.append(dict(body=a['body_index'],muscle=a['muscle'],end=a['endpoint'],old=a['attachment_mode'],new=b['attachment_mode'],
                old_reason=a['admission_reason'],new_reason=b['admission_reason'],resolved_point_unchanged=a['resolved_local_point_m']==b['resolved_local_point_m']))
    report=dict(changes=rows,baseline=old['coverage'],candidate=new['coverage'],outside_shoulder_changes=sum(r['body'] not in affected for r in rows))
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('changed',len(rows),'outside',report['outside_shoulder_changes'])
    for r in rows:
        if r['old']!=r['new']: print(r['muscle'],r['end'],r['old'],r['new'])
if __name__=='__main__':compare(*map(Path,sys.argv[1:]))

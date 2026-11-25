"""Test the emitted force route at the accepted endpoint, including its distributed attachments."""
import json
import runpy
import sys
from pathlib import Path
import numpy as np
from audit_guide_corridor import geometry
from fit_registered_apl_corridor import metrics
from fit_physical_guides import world_linear
from install_physical_routes import check_intervals
from compile_connected_joint import digest
from fit_route_binding import asset,read


def verify(reader, case, registration, out):
    cfg=read(case/'launch.json')
    muscle=Path(cfg['argv'][2])
    tendon=asset(cfg,'--tendon-payload')
    source=read(registration)
    for path,key in [(muscle,'updated_muscle'),(tendon,'updated_tendon')]:
        if digest(path)!=source[key] or cfg['pins'][str(path)]!=source[key]:
            raise ValueError('Unmatched force owner')
    rc,=(case/'native-traces/accepted-geometry').glob('*.receipt.json')
    poses,banks,pack=geometry(runpy.run_path(str(reader)),case,rc,False,False)
    rec=read(rc)
    if rec['physical_endpoint']!='accepted' or rec['surface_audit_endpoint']!='passed':
        raise ValueError('Rejected endpoint')
    rows=[]
    for sid,mi in [(147,259),(148,322)]:
        p,_,sites=world_linear(cfg,poses,sid,mi)
        m=metrics(dict(banks=banks),p)
        check_intervals(sid,m)
        rows.append(dict(side=sid,actuator=mi,sites=sites,world_points_mm=p.tolist(),length_mm=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),
                         metrics=m))
    out.write_text(json.dumps(dict(launch=digest(case/'launch.json'),pack=pack,step=rec['accepted_step'],seconds=rec['accepted_time_s'],
                                   registration=digest(registration),available_cortical_banks=len(banks)-1,rows=rows),indent=2)+'\n')
    print([(r['actuator'],r['length_mm'],r['metrics']['skin']['invalid_mm']) for r in rows])

if __name__=='__main__':verify(*(Path(p).resolve() for p in sys.argv[1:]))

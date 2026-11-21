"""Measure donor frame agreement for both native shoulder entheses and wrap centers."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from shoulder_inputs import digest
def run(sites,source,fits,out):
    fs={q['sid']:q for q in json.loads(fits.read_text())}
    rs={r['side']:r for r in json.loads(source.read_text())['routes'] if r['name'].startswith('s_glen')}
    rows=[]
    for row in json.loads(sites.read_text())['rows']:
        s=rs['r' if row['side'] else 'l']
        offsets=[]
        for ep in row['attachments']:
            site=next(q for q in s['sites'] if ep['source_site']==q['site'])
            offsets.append(float(np.linalg.norm(np.asarray(ep['local_m'])-site['point_local_m'])))
        h=row['attachments'][1];rigid=fs[h['stable']];sphere=np.array(s['source_wrap']['center_local_m'])
        fit=Rotation.from_quat(rigid['rotation_xyzw']).inv().apply(sphere-rigid['translation_m'])/rigid['scale']
        rows.append(dict(side=row['side'],scapular_enthesis_displacement_m=offsets[0],humeral_enthesis_displacement_m=offsets[1],
          humeral_wrap_center_displacement_m=float(np.linalg.norm(sphere-fit)),donor_default_path_m=s['default_source_path_m'],
          donor_default_wrap_points=s['default_source_wrap_points'],donor_stiffness=s['source_stiffness']))
    out.write_text(json.dumps(dict(bones_sha256=json.loads(sites.read_text())['candidate_sha256'],sites_sha256=digest(sites),rows=rows),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))

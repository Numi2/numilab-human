from pathlib import Path
import json,time,hashlib,traceback
from numilab_human import organ_family_geometry as g
r=Path(__file__).resolve().parent;repo=g.ROOT;c=json.loads((repo/'config/source-organ-family-composite.v2.json').read_text())
a=[repo/'Sources',repo/'Build/myosim-fullbody',repo/'Build/knee-parity-registration-20260929/candidate.v6.registration.json',repo/'Build/lung-source-coverage-20260929/payload/bodyparts3d-myosim-torso-anatomy.nhanatomy',repo/'Build/organ-family-coverage-20260929/payload.final/source-organ-family-anatomy.nhanatomy',r/'payload.verified'/g.PAYLOAD_NAME]
prefix=repo/'Build/lung-envelope-20260929/payload/thorax-lung-envelope.nhanatomy'
out=r/'audits.final';out.mkdir(exist_ok=True)
files=['src/numilab_human/organ_family_geometry.py','src/numilab_human/lung_envelope.py','src/numilab_human/torso_anatomy_audit.py','config/source-organ-family-composite.v2.json']
(out/'executed-source.json').write_text(json.dumps({f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
rows=[]
for name,pose,mask,focus,openhead in json.loads((r/'profiles.json').read_text()):
 v=r/'final-native'/name/'views';t=time.monotonic()
 try:
  report=g.audit(*a,next(v.glob('*.mrvpack')),next(v.glob('*.torso-anatomy-poses.json')),None if pose is None else tuple(tuple(x) for x in pose),mask,config=c,prefix_base_payload=prefix)
  (out/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
  row={'profile':name,'mask':mask,'focus':focus,'passed':report['passed'],'maximum_added_native_source_error_m':report['maximum_added_native_source_error_m'],'wall_seconds':time.monotonic()-t}
  print(json.dumps(row),flush=True);rows.append(row)
  assert report['passed']
 except Exception:
  (out/'failure.txt').write_text(traceback.format_exc());raise
(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')

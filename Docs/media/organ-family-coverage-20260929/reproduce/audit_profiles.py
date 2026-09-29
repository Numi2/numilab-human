from pathlib import Path
import json,time,hashlib,traceback
from numilab_human import organ_family_geometry as g
root=Path(__file__).resolve().parent;repo=g.ROOT
args=[repo/'Sources',repo/'Build/myosim-fullbody',repo/'Build/knee-parity-registration-20260929/candidate.v6.registration.json',repo/'Build/lung-source-coverage-20260929/payload/bodyparts3d-myosim-torso-anatomy.nhanatomy',repo/'Build/lung-envelope-20260929/payload/thorax-lung-envelope.nhanatomy',root/'payload.final'/g.PAYLOAD_NAME]
sources=['src/numilab_human/organ_family_geometry.py','src/numilab_human/lung_envelope.py','src/numilab_human/torso_anatomy_audit.py','config/source-organ-family-composite.v1.json']
out=root/'audits.verified';out.mkdir(exist_ok=True)
(out/'executed-source.json').write_text(json.dumps({p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
rows=[]
for name,pose in [('raw-source-rest',None),('projected-neutral',()),('torso-flexion',((7,-.4),(8,.1),(9,.2)))]:
 for mask in [63,256,512,1023]:
  dest=root/'final-native'/f'{name}-mask{mask}'/'views';start=time.monotonic()
  a=[*args,next(dest.glob('*.mrvpack')),next(dest.glob('*.torso-anatomy-poses.json')),pose,mask]
  try:
   report=g.audit(*a);(out/f'{name}-mask{mask}.json').write_text(json.dumps(report,indent=2)+'\n')
   row={'pose':name,'mask':mask,'passed':report['passed'],'maximum_added_native_source_error_m':report['maximum_added_native_source_error_m'],'wall_seconds':time.monotonic()-start}
   rows.append(row);print(json.dumps(row),flush=True)
   if not report['passed']:raise RuntimeError('source audit failed')
  except Exception:
   (out/'failure.txt').write_text(traceback.format_exc());raise
(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')

import json,hashlib,sys
from pathlib import Path
import numpy as np
from fractions import Fraction
P=Path('/Users/n/numi-human-left-pq-publication-1344');sys.path.insert(0,str(P/'src'))
from numilab_human import cardiac_cavity_intersections as ci
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
D=A/'back-forearm-native-014-012-right166/integrated-right-back-audit-001'
out=A/'selected-face-native-parity-1368.json';assert not out.exists()
rows=[];pins={}
for step in [0,7423,8000]:
 p=D/f'step-{step}-row-133.npz';r=D/f'step-{step}.json'
 for f in [p,r]:pins[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 with np.load(p,allow_pickle=False) as z:
  rec=ci._records([tuple(Fraction.from_float(float(x)) for x in v) for v in z['positions']],z['faces'].tolist())
 result=ci._audit_self_pairs_touching_faces(rec,{3304,3323,3599})
 expected=[x for x in json.loads(r.read_text())['exact_geometry']['rows']['133']['exact_self_pairs'] if {3304,3323,3599}.intersection(x)]
 assert result['triangle_pairs']==expected,(step,result,expected)
 rows.append({'step':step,'full_row_incident_pairs':expected,'selected_audit':result})
pins[str(Path(ci.__file__))]=hashlib.sha256(Path(ci.__file__).read_bytes()).hexdigest()
out.write_text(json.dumps({'matched':True,'scope':'Exact predicate regression on actual native row133 captures, not anatomical clearance','rows':rows,'pins':pins},indent=2)+'\n')
print(out)


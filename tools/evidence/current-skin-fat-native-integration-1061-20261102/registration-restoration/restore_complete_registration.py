from pathlib import Path
import json,csv,hashlib,time
D=Path(__file__).resolve().parent;S=D.parent
plan=json.loads((S/'native-import-028/source-preflight.plan.json').read_text());env=plan['stages'][0]['environment'];pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source=Path(env[pre+'COMBINED_NODE_MAP_CSV']);original=Path(env[pre+'NODE_REGISTRATION_CSV'])
assert sha(source)==env[pre+'COMBINED_NODE_MAP_SHA256'];assert sha(original)==env[pre+'NODE_REGISTRATION_SHA256']
header=next(csv.reader(original.open()));rows={r:{} for r in 'OID'};duplicates=0;max_distance=0.;t=time.time()
with source.open() as f:
 for row in csv.DictReader(f):
  role=row['registered_surface_layer']
  if role not in rows:continue
  i=int(row['registered_surface_vertex_row'])
  renamed=dict(row,layer=role,surface_vertex_row=str(i))
  for axis in 'xyz':renamed['parent_residual_'+axis+'_m']=row['parent_current_residual_'+axis+'_m']
  values=tuple(renamed[k] for k in header)
  if i in rows[role]:
   assert rows[role][i]==values,('shared-I registration differs',role,i)
   duplicates+=1
  else:rows[role][i]=values
  max_distance=max(max_distance,float(row['parent_registration_distance_m']))
output=D/'complete-node-parent-registration.csv';assert not output.exists()
with output.open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(header)
 for role in 'OID':
  with open(env[pre+'SURFACE_'+role+'_NODES_CSV']) as surface:n=sum(1 for _ in csv.DictReader(surface))
  assert sorted(rows[role])==list(range(n)),role
  for i in range(n):w.writerow(rows[role][i])
receipt={'source':{'path':str(source),'sha256':sha(source)},'incomplete_sidecar':{'path':str(original),'sha256':sha(original)},'output':{'path':str(output),'sha256':sha(output)},'rows':{r:len(v) for r,v in rows.items()},'exact_shared_I_duplicates':duplicates,'maximum_historical_source_registration_distance_m':max_distance,'geometry_changed':False,'mass_changed':False,'physical_admission':False,'native_steps':0,'elapsed_s':time.time()-t}
(D/'registration-restoration.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

from pathlib import Path
import json,sys,time,hashlib
sys.path.insert(0,'/Users/n/numi-human-free-apex-publication-1159/src')
sys.path.insert(0,'/Users/n/numi-human-performance-source-014/matter/tools')
from numilab_human import cardiac_partition_certificate as cert
from numilab_human import cardiac_cavity_intersections as predicates
import cardiac_geometry_binding as cb
O=Path(__file__).parent
P=Path('/Users/n/numi-human-resting-evidence-20261005/native-lung-free-apex-composition-1159-eightops-attempt1/final/resting-thorax.nhanatomy')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,d):
 with (O/name).open('x') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
inputs={str(f):sha(f) for f in [P,Path(__file__),Path(cert.__file__),Path(predicates.__file__),Path(cb.__file__)]}
assert inputs[str(P)]=='c10dce4609be99fdc569801c2705e23c46120b610c8aa86c631165d38baf4713'
t=time.monotonic()
_,_,records,v,i=cb.read_payload(P)
regions={}; face_rows={}
for sid,name in [(318,'right_atrium'),(319,'right_ventricle')]:
 _,p,f=cb.surface_arrays(records,v,i,sid,expected_layer=9)
 regions[name]=[tuple(tuple(map(float,p[j])) for j in row) for row in f]
 face_rows[name]={tuple(sorted(tri)):k for k,tri in enumerate(regions[name])}
shared=face_rows['right_atrium'].keys() & face_rows['right_ventricle'].keys()
interface=[regions['right_atrium'][face_rows['right_atrium'][key]] for key in sorted(shared)]
write('start.json',{'inputs_sha256':inputs,'shared_triangle_count':len(interface),
 'scope':'Exact disjoint-interior check of current right-heart cavity domains and geometrically shared boundary. Does not select biological valve anatomy, alter geometry, advance simulation, or check all cardiac surface pairs.'})
# Timing wrappers preserve all arguments and returned values.
for name in ['_topology','_moment_vector']:
 original=getattr(cert,name)
 def wrapped(*args,_name=name,_original=original,**kwargs):
  t0=time.monotonic();print('begin',_name,flush=True)
  result=_original(*args,**kwargs)
  print('end',_name,time.monotonic()-t0,flush=True);return result
 setattr(cert,name,wrapped)
original=predicates._audit_pair
def audited(*args,**kwargs):
 t0=time.monotonic();print('begin self exact audit',flush=True)
 result=original(*args,**kwargs)
 print('end self exact audit',time.monotonic()-t0,result['count'],flush=True);return result
predicates._audit_pair=audited
try:
 proof=cert.audit_shared_interface_partition(regions,interface,classify_connected_patches=True)
 report={'status':'pass','scope':'Current source-neutral right-atrium/right-ventricle cavity geometry only; geometric shared interface is not a biological valve model or a clinical validation.','inputs_sha256':inputs,'elapsed_s':time.monotonic()-t,'proof':proof,
 'shared_face_rows':[{name:face_rows[name][key] for name in regions} for key in sorted(shared)],
 'inputs_unchanged':all(sha(Path(p))==h for p,h in inputs.items())}
 write('report.json',report);print('report_sha256',sha(O/'report.json'),flush=True)
except Exception as e:
 write('failure.json',{'status':'failed','error':repr(e),'elapsed_s':time.monotonic()-t,'inputs_sha256':inputs})
 raise

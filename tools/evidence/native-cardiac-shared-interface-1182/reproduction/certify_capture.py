from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
sys.path.insert(0,'/Users/n/numi-human-free-apex-publication-1159/src')
sys.path.insert(0,'/Users/n/numi-human-performance-source-014/matter/tools')
from numilab_human import cardiac_partition_certificate as cert
from numilab_human import cardiac_cavity_intersections as predicates
import accepted_mrvpack_surface_audit as am
import cardiac_geometry_binding as cb
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--capture',type=Path,required=True);ap.add_argument('--step',type=int,required=True)
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 a.output.mkdir(exist_ok=False)
 def write(name,d):
  with (a.output/name).open('x') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
 P=Path('/Users/n/numi-human-resting-evidence-20261005/native-lung-free-apex-composition-1159-eightops-attempt1/final/resting-thorax.nhanatomy')
 source_proof=Path('/Users/n/numi-human-retained-delivery-20261009/cardiac-boundary-components-1182/attempt-004/report.json')
 assert sha(source_proof)=='77e1395ae3bab7e6ef33ef790a30bb617f75f24c7d6507e718c72cc95ead04d8'
 sp=json.loads(source_proof.read_text())
 pack=a.capture/f'step-{a.step}.mrvpack';receipt=pack.with_suffix('.receipt.json')
 paths=[P,source_proof,pack,receipt,Path(__file__),Path(cert.__file__),Path(predicates.__file__),Path(am.__file__),Path(cb.__file__)]
 pins={str(p):sha(p) for p in paths}
 assert pins[str(P)]==sp['inputs_sha256'][str(P)]
 for mod in [cert,predicates,cb]:assert pins[str(Path(mod.__file__))]==sp['inputs_sha256'][str(Path(mod.__file__))]
 write('inputs.json',{'sha256':pins,'step':a.step,'scope':'Read-only accepted right-heart geometry using source-declared shared face rows; no physical stepping.'})
 t=time.monotonic()
 try:
  mapped,stream,offset,surfaces=am.read_pack(pack)
  validation=am.validate_accepted_receipt(pack,receipt,a.step,mapped,offset,surfaces)
  keys=[(51025,318),(51025,319)]
  meshes,source=am.load_source_boundary_meshes(P,pins[str(P)],surfaces,keys)
  regions={}
  for name,key in zip(('right_atrium','right_ventricle'),keys):
   faces=np.asarray(surfaces[key]['faces'],dtype=np.int64)
   source_faces=np.asarray(meshes[key][1],dtype=np.int64)
   if faces.shape!=source_faces.shape:raise ValueError('face count changed')
   shifts=faces-source_faces
   if not np.all(shifts==shifts[0,0]):raise ValueError('source face row correspondence changed')
   _,points,_=am.surface_points(mapped,offset,surfaces[key]['faces'])
   regions[name]=[tuple(points[int(j)] for j in row) for row in faces]
  mapped.close();stream.close()
  interface=[]
  for rows in sp['shared_face_rows']:
   ra=regions['right_atrium'][rows['right_atrium']]
   rv=regions['right_ventricle'][rows['right_ventricle']]
   if cert._oriented_key(ra)!=cert._oriented_key(tuple(reversed(rv))):
    raise ValueError('source shared face copy separated or changed winding')
   interface.append(ra)
  proof=cert.audit_shared_interface_partition(regions,interface,classify_connected_patches=True)
  report={'status':'pass','accepted_receipt':validation,'inputs_sha256':pins,'elapsed_s':time.monotonic()-t,'proof':proof,
  'inputs_unchanged':all(sha(Path(p))==h for p,h in pins.items()),
  'scope':'Exact captured Float32 geometry, RA/RV only, shared interface face rows inherited from source proof. A finite pose check is not continuous-time or whole-heart qualification; reduced valves remain the functional owner.'}
  write('report.json',report);print(json.dumps({'status':'pass','step':a.step,'elapsed_s':report['elapsed_s'],'report_sha256':sha(a.output/'report.json')}),flush=True)
 except Exception as e:
  write('failure.json',{'status':'failed','step':a.step,'error':repr(e),'elapsed_s':time.monotonic()-t,'inputs_sha256':pins})
  raise
if __name__=='__main__':main()

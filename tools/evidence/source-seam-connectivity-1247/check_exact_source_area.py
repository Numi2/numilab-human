#!/usr/bin/env python3
"""Exact noncollinearity check of restored authored source triangles."""
from pathlib import Path
import hashlib,json,sys,zipfile
HERE=Path(__file__).resolve().parent
ROOT=Path('/Users/n/numi-human-source-seam-connectivity-1247')
ARCHIVE=Path('/Users/n/numi-human-source-cache/isa_BP3D_4.0_obj_99.zip')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 report=HERE/'source-selection-comparison.json'
 prior=json.loads(report.read_text())
 for p,h in prior['inputs'].items():
  assert sha(p)==h,p
 sys.path.insert(0,str(ROOT/'src'))
 from numilab_human import model
 rows=[]
 with zipfile.ZipFile(ARCHIVE) as archive:
  for r in prior['surfaces']:
   v,f=model._bodyparts_obj_triangles(archive.read(r['member']),r['member'])
   v,f,_=model._bodyparts_largest_connected_surface_component(v,f,r['member'])
   ratios=[[float(x).as_integer_ratio() for x in point] for point in v]
   denominator=max(d for point in ratios for _,d in point)
   xyz=[[n*(denominator//d) for n,d in point] for point in ratios]
   collinear=[]
   for i,(ia,ib,ic) in enumerate(f):
    a,b,c=xyz[ia],xyz[ib],xyz[ic]
    u=[b[k]-a[k] for k in range(3)];w=[c[k]-a[k] for k in range(3)]
    cross=[u[1]*w[2]-u[2]*w[1],u[2]*w[0]-u[0]*w[2],u[0]*w[1]-u[1]*w[0]]
    if not any(cross):collinear.append(i)
   rows.append({'stable_id':r['stable_id'],'member_id':r['member_id'],'face_count':len(f),'exact_collinear_face_rows':collinear,'source_coordinate_unit':'millimetre','predicate':'exact integers on the common dyadic lattice of parsed source Float64 coordinates'})
 result={'comparison_path':str(report),'comparison_sha256':sha(report),'script_sha256':sha(__file__),'candidate_model_sha256':sha(ROOT/'src/numilab_human/model.py'),'surfaces':rows,'qualification':'Authored source geometry only; not registered Float32 or native accepted-pose geometry.'}
 out=HERE/'source-selection-exact-area.json'
 assert not out.exists()
 out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'report':str(out),'sha256':sha(out),'collinear_by_id':{r['stable_id']:r['exact_collinear_face_rows'] for r in rows}}))
if __name__=='__main__':main()

from pathlib import Path
import json,mmap,sys
import numpy as np
sys.path.insert(0,str(Path.cwd()/'resting-surface-clearance-001'))
import audit
base=Path.cwd()/'regional-bed-layers-001/capture-4s/native/accepted-geometry'
def report(step,filename):
 p=base/f'step-{step}.mrvpack'
 raw=json.loads((Path.cwd()/'resting-surface-clearance-001'/filename).read_text())
 r=json.loads(p.with_suffix('.receipt.json').read_text())
 with p.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
  _,_,v,faces=next(audit.primitives(mm,{},skin_only=True))
 t=v[faces]
 pairs=np.array(raw['self_intersections']['triangle_pairs'],dtype=int).reshape(-1,2)
 result={'source_step':step,'raw_self_pairs':len(pairs)}
 if len(pairs):
  joint=t[pairs]
  on_bed=abs(joint[...,2])<1.e-9
  result['both_triangles_coplanar_bed']=int(on_bed.all(axis=(1,2)).sum())
  result['touching_bed_vertices']=int(on_bed.any(axis=(1,2)).sum())
  result['both_above_one_mm']=int((joint[...,2]>.001).all(axis=(1,2)).sum())
 for sem in ('51004','51005','51011'):
  jobs=[row for row in raw['surfaces'] if row['semantic']==int(sem) and row['count']]
  ids=np.concatenate([np.array(row['triangle_pairs'])[:,0] for row in jobs])
  tri=t[ids]
  result[sem]={'pairs':len(ids),'skin_faces':int(len(set(ids))), 'skin_face_plane_contact':int((abs(tri[...,2])<1e-9).any(axis=1).sum())}
 return result
ret=[report(0,'initial.json'),report(2000,'settled-fixed.json')]
Path('coupled-crossing-repair-001/localization.json').write_text(json.dumps(ret,indent=2)+'\n')
print(json.dumps(ret,indent=2))

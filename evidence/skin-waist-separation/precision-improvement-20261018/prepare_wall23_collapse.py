from pathlib import Path
import sys,json,hashlib,numpy as np
sys.path.insert(0,'/Users/n/numi-human-closed-outer-skin-20261017/src')
from numilab_human.resting_pleura_proxy import _parse_payload
from numilab_human.resting_respiratory_mesh_quality import collapse_registered_interior_vertex
from numilab_human.resting_intercostal_mesh_arrangement import encode_payload_with_row_replacements
c=Path('/Users/n/numi-human-separation-20261018');p=c/'C/improve-cardiac321-001/resting-thorax.nhanatomy';out=c/'C/improve-wall23-003';out.mkdir(exist_ok=False)
raw=p.read_bytes();rows=_parse_payload(raw)[-1];r=rows[23];v=r['vertices6'];f=r['faces'];attempts=[]
for keep,drop in [(26655,26654),(26654,26655)]:
 try:
  owners={}
  for idx in (keep,drop):
   point=tuple(map(float,v[idx,:3]));owners[point]=[sid for sid,row in rows.items() if np.any(np.all(row['vertices6'][:,:3]==v[idx,:3],axis=1))]
  assert all(o==[23] for o in owners.values()),owners
  prepared={23:(v.copy(),f.copy(),{i:[i] for i in range(len(f))},{})}
  op=collapse_registered_interior_vertex(prepared,v[keep,:3],v[drop,:3],maximum_source_displacement_m=1e-6,minimum_altitude_m=2.5e-7,maximum_volume_change_m3=1e-12,volume_bounded_surface_ids=(23,),complete_point_owners=owners,allow_equal_owner_sets=True)
  nv,nf,parents,detail=prepared[23];nf=nf.copy();nf[nf>=drop]+=1
  assert np.array_equal(nv[:,:3],np.delete(v[:,:3],drop,axis=0))
  replacement={**r,'vertices6':v.copy(),'faces':nf};payload,abi=encode_payload_with_row_replacements(raw,{23:replacement});path=out/'resting-thorax.nhanatomy';path.write_bytes(payload)
  changed=[int(i) for i in range(len(f)) if not parents[i] or not np.array_equal(f[i],nf[parents[i][0]])]
  ancestry={str(i):parents[i] for i in changed};(out/'source-face-ancestry.json').write_text(json.dumps(ancestry,indent=2)+'\n')
  report=dict(base=str(p),base_sha256=hashlib.sha256(raw).hexdigest(),candidate=str(path),candidate_sha256=hashlib.sha256(payload).hexdigest(),stable_id=23,concept_id='FMA13884',operation=op,abi=abi,keep_source_vertex=keep,drop_source_vertex=drop,original_vertex_allocations_attributes_unchanged=True,affected_source_faces=changed,source_face_ancestry=str(out/'source-face-ancestry.json'),failed_attempts=attempts,native_admission=False)
  (out/'preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));break
 except ValueError as e:attempts.append(dict(keep=keep,drop=drop,error=str(e)));print(attempts[-1],flush=True)
else:
 (out/'failed-preparation.json').write_text(json.dumps(attempts,indent=2)+'\n');raise RuntimeError(attempts)

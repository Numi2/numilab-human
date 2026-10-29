#!/usr/bin/env python3
from pathlib import Path
from fractions import Fraction
import importlib.util,hashlib,json,sys,time
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
sys.path.insert(0,str(P.parent/'authoring-deps'))
import manifold3d as mf
OUT=Path(__file__).resolve().parent/'output'
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def mesh(z):return mf.Manifold(mf.Mesh64(vert_properties=np.ascontiguousarray(z['points_world_m_f64']),tri_verts=np.ascontiguousarray(z['faces'],dtype=np.uint64)))
def main():
 t=time.monotonic();OUT.mkdir(exist_ok=False)
 base=P/'joint-closed-material-interface-652/component-diagnostic-653/run-006'
 bd=json.loads((base/'output/report.json').read_text())
 for p,h in bd['input_pins'].items():
  if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('source pin changed: '+p)
 L=Path(next(k for k in bd['input_pins'] if k.endswith('/remesh.py')))
 spec=importlib.util.spec_from_file_location('local656loader',L);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);owner=mod.load_exact_owner()
 sdir=P/'paired-fine-headwall-rim-635/run-013/output';fdir=P/'paired-fine-headwall-rim-635/run-010/output'
 S=load(sdir/'fine-skin-closed-boundary.npz');F=load(fdir/'fine-fat-closed-boundary.npz')
 sr=json.loads((sdir/'report.json').read_text());fr=json.loads((fdir/'report.json').read_text())
 def source(p):
  with np.load(p,allow_pickle=False) as z:return z['points_world_m'],z['faces']
 op,of=source(sr['fine_surfaces']['O']['path']);ip,iff=source(sr['fine_surfaces']['I']['path']);dp,df=source(fr['fine_surfaces']['D']['path'])
 x=np.concatenate((op,ip,dp));no=len(op);ni=len(ip)
 # Canonical uncompressed source faces, with shared I global identities.
 sf=np.vstack((of,iff[:,[0,2,1]]+no,S['oriented_rim_faces']))
 ff=np.vstack((iff+no,df[:,[0,2,1]]+no+ni,F['oriented_rim_faces']+no))
 sm=mesh(S);fm=mesh(F);inter=sm^fm
 parts=inter.decompose();j=max(range(len(parts)),key=lambda i:parts[i].volume())
 part=parts[j];pb=part.bounding_box();lo=np.asarray(pb[:3]);hi=np.asarray(pb[3:])
 pm=part.to_mesh64();px=np.asarray(pm.vert_properties)[:,:3];pf=np.asarray(pm.tri_verts,dtype=np.int64)
 np.savez_compressed(OUT/'dominant-boolean-piece.npz',points_world_m_f64=px,faces=pf)
 st=x[sf];ft=x[ff]
 keep=lambda tri:np.all(tri.max(axis=1)>=lo,axis=1)&np.all(tri.min(axis=1)<=hi,axis=1)
 si=np.flatnonzero(keep(st));fi=np.flatnonzero(keep(ft))
 a,_=owner.float32_surface_records(x,sf[si],source_seam_groups=());b,_=owner.float32_surface_records(x,ff[fi],source_seam_groups=())
 # Report original closed material row IDs, not crop-local IDs.
 a=[(r[0],r[1],r[2],int(si[n]),r[4]) for n,r in enumerate(a)]
 b=[(r[0],r[1],r[2],int(fi[n]),r[4]) for n,r in enumerate(b)]
 nO=len(of);nI=len(iff)
 pairs=owner._query_surface_aabb(a,owner._prepare_surface_aabb(b))
 skipped=0;retained=[]
 for ra,rb in pairs:
  if nO<=ra[3]<nO+nI and rb[3]<nI:skipped+=1
  else:retained.append((ra,rb))
 exact=owner._audit_record_pairs(retained,same_surface=True)
 # Every point-location query uses rational Float64 query coordinates against
 # the unchanged exact Float32 source boundary, expressed in the same lattice.
 prepared=[]
 for z in (S,F):
  rec,_=owner.float32_surface_records(z['points_world_m_f64'].astype(np.float32),z['faces'],source_seam_groups=())
  n=len(rec);prepared.append(tuple(owner.prepare_point_location(c) for c in (rec[:n//2],rec[n//2:])))
 samples=[('vertices_mean',px.mean(axis=0))]
 samples += [('face_'+str(k),px[tri].mean(axis=0)) for k,tri in enumerate(pf[:16])]
 checks=[]
 for label,p in samples:
  q=tuple(Fraction.from_float(float(v))*(1<<149) for v in p)
  loc=[owner.point_location_prepared_indexed_chunks(q,r)['location'] for r in prepared]
  checks.append({'sample':label,'point_m':p.tolist(),'skin':loc[0],'fat':loc[1]})
 result={'status':'LOCALIZED_SOURCE_DIAGNOSTIC','dominant_boolean_component':j,'dominant_boolean_volume_m3':float(part.volume()),'boolean_piece_vertices':len(px),'boolean_piece_faces':len(pf),'bbox_m':[lo.tolist(),hi.tolist()],'skin_source_faces_in_bbox':len(si),'fat_source_faces_in_bbox':len(fi),'intended_shared_I_pairs_skipped':skipped,'nonshared_exact_crossings':exact,'source_row_witness_layers':[{'skin_row':a,'fat_row':b,'skin_role':'O' if a<nO else 'I' if a<nO+nI else 'rim','fat_role':'I' if b<nI else 'D' if b<nI+len(df) else 'rim'} for a,b in exact['triangle_pairs']],'source_point_locations':checks,'physical_admission':False,'runtime_created':0,'accepted_steps':0,'elapsed_s':time.monotonic()-t}
 (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()

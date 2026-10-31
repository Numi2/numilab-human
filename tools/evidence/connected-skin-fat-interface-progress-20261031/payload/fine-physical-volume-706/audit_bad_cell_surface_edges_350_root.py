#!/usr/bin/env python3
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,time
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('quality350',H/'fixed_boundary_volume_flip_198_root.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
out=H/'native-boundary-edge-witness-census-350-root';out.mkdir(exist_ok=False)
paths={'O':H/'joint-native-OI-ears-279-root/O-batch006-dp001/typed-280-003/candidate-O-typed-current.npz',
 'I':H/'joint-native-OI-ears-279-root/I-batch002-run001/typed-280-002/candidate-I-typed-current.npz',
 'D':H/'native-apex-D-singleton-after-paired-293-root/run-001/typed-001/candidate-D-typed-current.npz'}
volumes={'Skin':H/'connected-skin-quality-342-root/skin-run001/cycle-00-condition/conditioned-volume-proposal.npz',
 'Fat':H/'fat-inferred-identity-mask-336-root/fat-run001/fixed-boundary-volume-proposal.npz'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report={'physical_admission':False,'native_steps':0,'source_pins':{str(p):sha(p) for p in paths.values()},'roles':{}}
for role,f in volumes.items():
 with np.load(f) as z:p=z['points_world_m'];t=z['tetrahedra']
 d,q=m.measurements(p,t);bad=~m.passes(d,q)
 lookup={x.tobytes():i for i,x in enumerate(p)}
 edges=[];layers=[]
 for li,layer in enumerate(('O','I','D')):
  if (role=='Skin' and layer=='D') or(role=='Fat' and layer=='O'):continue
  with np.load(paths[layer]) as z:pts=z['points_world_m_f32'];faces=z['faces']
  mp=np.asarray([lookup.get(x.tobytes(),-1) for x in pts],np.int64)
  ee=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1);ee=np.unique(ee,axis=0)
  mapped=mp[ee];mapped=mapped[np.all(mapped>=0,axis=1)]
  edges.append(mapped);layers.append(np.full(len(mapped),li,np.int8))
 e=np.vstack(edges);l=np.concatenate(layers)
 # Match actual surface edges against all bad-tet edges using packed global IDs.
 n=len(p);ekey=e[:,0]*n+e[:,1];o=np.argsort(ekey);se=ekey[o]
 be=np.sort(t[bad][:,[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]].reshape(-1,2),axis=1)
 bk=be[:,0]*n+be[:,1];ix=np.searchsorted(se,bk);valid=ix<len(se);valid[valid]&=se[ix[valid]]==bk[valid]
 hit_edges,counts=np.unique(o[ix[valid]],return_counts=True)
 length=np.linalg.norm(p[e[hit_edges,0]].astype(float)-p[e[hit_edges,1]],axis=1)
 data={}
 for li,layer in enumerate(('O','I','D')):
  sel=l[hit_edges]==li;le=length[sel]
  data[layer]={'actual_surface_edges_touching_bad_cells':int(sel.sum()),'minimum_length_m':float(le.min(initial=np.inf)),
    'short_edge_counts':{str(th):int(np.count_nonzero(le<=th)) for th in (1e-8,1e-7,1e-6,1e-5,1e-4,1e-3)}}
 np.savez_compressed(out/(role.lower()+'-short-surface-edge-witnesses.npz'),edge_point_rows=e[hit_edges],layer=l[hit_edges],length_m=length,bad_tet_incidence=counts,input_sha256=np.asarray(sha(f)))
 report['roles'][role]={'input':str(f),'input_sha256':sha(f),'bad_cells':int(bad.sum()),'by_surface_role':data}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

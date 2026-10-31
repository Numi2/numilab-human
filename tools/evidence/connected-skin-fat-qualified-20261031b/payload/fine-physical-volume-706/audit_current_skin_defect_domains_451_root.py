from pathlib import Path
import numpy as np, json, hashlib, importlib.util, time
H=Path(__file__).resolve().parent
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
start=time.monotonic();T=mod('volume451',H/'transfer_surface_contractions_indexed_379_root.py');m=T.m
current=json.loads((H/'checkpoint.json').read_text())['current_verified_state_20261031'];e=current['Skin_current'];ip=Path(e['path']);assert sha(ip)==e['sha256']
out=H/'skin-defect-domain-451-root';out.mkdir(exist_ok=False)
with np.load(ip) as z:p=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices']
det,q=m.measurements(p,t);good=m.passes(det,q);b=T.boundary(t);bound=np.zeros(len(p),bool);bound[b.ravel()]=True
surf={}
for role in ('O','I'):
 e=current['source_surfaces'][role];path=Path(e['path']);assert sha(path)==e['sha256']
 with np.load(path) as z:sp=z['points_world_m_f32'];sf=z['faces']
 keys={T.key(x) for x in sp[np.unique(sf)]}
 surf[role]=np.asarray([T.key(x) in keys for x in p])
rows=np.flatnonzero(~good);bad=t[rows];xyz=p[bad].astype(float)
length=np.linalg.norm(xyz[:,:,None,:]-xyz[:,None,:,:],axis=3);es=length[:,np.triu_indices(4,1)[0],np.triu_indices(4,1)[1]]
def pct(a):return list(map(float,np.percentile(a,[0,1,25,50,75,99,100]))) if len(a) else []
cats={}
for code in range(16):
 mask=(bound[bad].sum(1)==(code&7)) if code<5 else np.zeros(len(bad),bool)
 if code<5:cats['boundary_node_count_'+str(code)]=int(mask.sum())
for role,mask in surf.items():cats['touch_'+role]=int(np.count_nonzero(mask[bad].any(1)))
cats['touch_O_and_I']=int(np.count_nonzero(surf['O'][bad].any(1)&surf['I'][bad].any(1)))
centroid=xyz.mean(1)
r={'status':'PASS_CURRENT_SKIN_DEFECT_DOMAIN_CENSUS','physical_admission':False,'native_steps':0,'source':str(ip),'source_sha256':sha(ip),'producer_sha256':sha(__file__),'nodes':len(p),'cells':len(t),'bad':len(rows),'absolute_volume_failures':int(np.count_nonzero(det/6<=1e-18)),'normalized_det_failures':int(np.count_nonzero(q<1e-5)),'categories':cats,'quality_percentiles':pct(q[rows]),'volume_m3_percentiles':pct(det[rows]/6),'shortest_edge_m_percentiles':pct(es.min(1)),'longest_edge_m_percentiles':pct(es.max(1)),'bad_centroid_min':centroid.min(0).tolist(),'bad_centroid_max':centroid.max(0).tolist(),'bad_with_interior_node':int(np.count_nonzero((~bound[bad]).any(1))),'elapsed_s':time.monotonic()-start}
np.savez_compressed(out/'defect-witnesses.npz',cell_rows=rows,tetrahedra=bad,det6=det[rows],quality=q[rows],boundary_node_mask=bound[bad],O_node_mask=surf['O'][bad],I_node_mask=surf['I'][bad])
r['witnesses_sha256']=sha(out/'defect-witnesses.npz');(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)

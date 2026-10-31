from pathlib import Path
import numpy as np,json,time,hashlib
H=Path(__file__).resolve().parent
out=H/'native-volume-quality-193-root';out.mkdir(exist_ok=False)
rows=[]
for role in ('Skin','Fat'):
 start=time.monotonic();base=H/'combined-child-handoff-142';rn=role.lower()
 nodes=np.loadtxt(base/(rn+'.node'),skiprows=1,usecols=(0,1,2,3));ids=nodes[:,0].astype(np.int64);x=nodes[:,1:].astype(np.float32).astype(np.float64)
 ele=np.loadtxt(base/(rn+'.ele'),skiprows=1);t=np.searchsorted(ids,ele[:,1:5].astype(np.int64))
 assert np.array_equal(ids[t],ele[:,1:5].astype(np.int64))
 normmin=np.inf;vmin=np.inf;badnorm=[];badvol=[];w=[]
 for a in range(0,len(t),65536):
  z=x[t[a:a+65536]];d=z[:,1:]-z[:,:1];det=np.einsum('ij,ij->i',d[:,0],np.cross(d[:,1],d[:,2]))
  emax=np.zeros(len(z))
  for i in range(4):
   for j in range(i+1,4):emax=np.maximum(emax,np.linalg.norm(z[:,i]-z[:,j],axis=1))
  nd=det/(emax**3);v=det/6
  bn=np.flatnonzero((~np.isfinite(nd))|(nd<1e-5));bv=np.flatnonzero((~np.isfinite(v))|(v<=1e-18))
  badnorm.extend((bn+a).tolist());badvol.extend((bv+a).tolist());normmin=min(normmin,float(nd.min()));vmin=min(vmin,float(v.min()))
  if len(w)<20:
   for k in bn[:20-len(w)]:w.append({'child_role_row':int(k+a),'node_rows':t[k+a].tolist(),'volume_m3':float(v[k]),'normalized_determinant':float(nd[k]),'maximum_edge_m':float(emax[k])})
 np.savez_compressed(out/(rn+'-failing-cells.npz'),normalized_det_rows=np.array(badnorm,dtype=np.uint32),absolute_volume_rows=np.array(badvol,dtype=np.uint32))
 rows.append({'role':role,'children':len(t),'minimum_F32_widened_signed_volume_m3':vmin,'minimum_normalized_det':normmin,'below_unpaired_normalized_det_1e_5':len(badnorm),'at_or_below_legacy_compiler_volume_1e_18':len(badvol),'witnesses':w,'elapsed_s':time.monotonic()-start})
report={'status':'CURRENT_GEOMETRY_GATE_CENSUS_NOT_NATIVE_ADMISSION','reference':'Exact emitted Float32 coordinates from142; transformed registered reference and actual current compiler route must be checked separately. Positivity alone is not the quality gate.','physical_admission':False,'native_steps':0,'roles':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({**report,'roles':[{k:v for k,v in r.items() if k!='witnesses'} for r in rows]},indent=2))

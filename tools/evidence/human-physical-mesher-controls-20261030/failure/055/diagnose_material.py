from pathlib import Path
import json,hashlib,numpy as np,time
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706'
src=H/'tetgen-explicit-no-improvement-053/skin037-run-001'
out=H/'tetgen-material-ownership-diagnostic-055';out.mkdir(exist_ok=False)
started=time.monotonic()
def table(path):
 with path.open() as f:
  head=f.readline().strip()
  return head,np.loadtxt(f,comments='#')
nh,n=table(src/'skin.1.node');eh,e=table(src/'skin.1.ele')
ids=n[:,0].astype(np.int64); assert np.array_equal(ids,np.arange(1,len(ids)+1))
t=e[:,1:5].astype(np.int64)-1;p=n[:,1:4];pf=p.astype(np.float32).astype(np.float64)
def sv(p):
 q=p[t];return np.einsum('ij,ij->i',q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))/6
v=sv(p);vf=sv(pf)
report={'node_header':nh,'element_header':eh,'source_node_sha256':hashlib.sha256((src/'skin.1.node').read_bytes()).hexdigest(),'source_ele_sha256':hashlib.sha256((src/'skin.1.ele').read_bytes()).hexdigest(),'physical_admission':False,'runtime':0,'accepted_steps':0,'material_attributes':{}}
for a in np.unique(e[:,5]):
 rows=np.flatnonzero(e[:,5]==a); cr=pf[t[rows]].mean(1); w=rows[:min(8,len(rows))]
 report['material_attributes'][str(a)]={'cells':len(rows),'total_abs_volume_f64_m3':float(np.abs(v[rows]).sum()),'total_abs_volume_f32_m3':float(np.abs(vf[rows]).sum()),'zeros_f32':int((vf[rows]==0).sum()),'zeros_f64':int((v[rows]==0).sum()),'negative_f32':int((vf[rows]<0).sum()),'centroid_min':cr.min(0).tolist(),'centroid_max':cr.max(0).tolist(),'witnesses':[{'element_row':int(i),'node_ids':(t[i]+1).tolist(),'points_world_m':pf[t[i]].tolist(),'centroid':pf[t[i]].mean(0).tolist(),'signed_volume_f64':float(v[i]),'signed_volume_f32':float(vf[i])}for i in w]}
report['wall_s']=time.monotonic()-started
(out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:({a:{z:w for z,w in d.items() if z!='witnesses'} for a,d in val.items()}if k=='material_attributes'else val)for k,val in report.items()}))


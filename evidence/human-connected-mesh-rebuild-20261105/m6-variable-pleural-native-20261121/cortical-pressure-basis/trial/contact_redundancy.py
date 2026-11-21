"""Measure independent force and moment directions and repeated physical contact rows."""
import sys,csv,json
from pathlib import Path
import numpy as np
from material_context import sha
a,z,out=map(Path,sys.argv[1:]);m=json.loads((a/'manifest.json').read_text());base=m['finite_facets']['start']+16*m['motion_contact_registry']['original_pairs']
pool=np.fromfile(a/'pool-material-points.bin','f4').reshape(-1,7,4);bones=np.fromfile(a/'motion-cortical-owners.bin','u4').reshape(-1,4)
free=np.fromfile(a/'free-node-rows.bin','u4').reshape(-1,4)[:,2]!=0
active={int(r['slot']):int(r['cortical_row']) for r in csv.DictReader((z/'accepted-cortical-registry.csv').open())}
rows=list(csv.DictReader((z/'accepted-pelvic-state.csv').open()));by={}
for r in rows:by.setdefault(r['field'],{})[int(r['index'])]=np.array([r[i] for i in 'xyzw'],'f4')
x=np.stack([by['material_node'][j][:3] for j in range(len(by['material_node']))]).astype(float)
result={'accepted_step':int(rows[0]['accepted_step']),'source_sha256':sha(z/'accepted-pelvic-state.csv'),'parts':{}}
for side in range(2):
    loads=[];ids=[];columns=[];markers=[];nvec=[]
    for k,v in sorted(by['normal_and_force'].items()):
        if v[3]<=0:continue
        if k<base:q=pool[k,3].view('u4');j=int(q[1]==145);bone=int(q[2])
        else:_,bone,j,_=bones[active[(k-base)//16]]
        if j!=side:continue
        p=by['reverse_parent_nodes'][k].astype(int) if k in by['reverse_parent_nodes'] else pool[k,5].view('u4')
        w=by['reverse_shape'].get(k,pool[k,6]).astype(float);n=v[:3].astype(float)
        f=np.zeros_like(x);np.add.at(f,p,w[:,None]*n);arm=w@x[p];rigid=np.zeros((4,6))
        rigid[[1,128,131,145].index(bone)]=np.r_[n,np.cross(arm,n)]
        ids.append(k);columns.append(f.ravel());markers.append(rigid.ravel());loads.append(v[3]);nvec.append(n)
    A=np.stack(columns,axis=1);B=np.vstack((A,np.stack(markers,axis=1)))
    def analyze(h):
        h=h[np.any(h!=0,axis=1)]
        scales=np.max(abs(h),axis=1);s=np.linalg.svd(h/scales[:,None],compute_uv=False)
        signed=h.T.copy();signed[signed==0]=0
        counts=np.unique(signed,axis=0,return_counts=True)[1]
        return {'loaded':len(ids),'unique_exact_columns':len(counts),'max_exact_multiplicity':int(max(counts)),
          'rank_row_scaled':{str(t):int(sum(s>t*s[0])) for t in (1e-10,1e-8,1e-6,1e-4)}}
    zero=np.array(loads);result['parts'][str(side)]={'free_nodes':analyze(A[np.repeat(free,3)]),'all_tissue_nodes_and_bone_wrenches':analyze(B),'positive_pressure_lt_1e-5_N':int(sum(zero<1e-5)),'pressure_quantiles_N':np.quantile(zero,[0,.25,.5,.75,1]).tolist()}
out.write_text(json.dumps(result,indent=2)+'\n')

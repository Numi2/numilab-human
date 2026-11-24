"""Locate source anatomy along the unsupported load normal in the verified scapula coordinate frame."""
from pathlib import Path
import hashlib,json,re,zipfile
import numpy as np
from scipy.spatial.transform import Rotation
from audit_registered_skeleton import source_geometry
from rebind_ligament_material import banks
from trace_loaded_columns import hits
from compile_connected_joint import digest
from prepare_shoulder import read

def run(w):
    case=w/'native-cavity-bound';columns=read(w/'bed-corridor.json');native=read(w/'native-column.json');cfg=read(case/'launch.json')
    archive=w/'official-isa.zip';bone=Path(cfg['argv'][3]);manifest=read(bone.with_suffix('.manifest.json'));entry=next(r for r in manifest['source']['anchors'] if r['member_id']=='FJ3384')
    sha=next(r['sha256']for r in manifest['source']['bodyparts']['archives']if r['hierarchy']=='is_a')
    if digest(archive)!=sha or columns['native']!=digest(case/'launch.json'):raise ValueError('Stale native or atlas input')
    R=read(next((case/'native-traces/accepted-geometry').glob('*.receipt.json')));body=entry['core_body_index'];pose=next(p for p in R['accepted_body_poses']if p['body_index']==body)
    tf,local,packed=banks(bone)[body,10];world=Rotation.from_quat(pose['quaternion_xyzw']).apply(local)+pose['position_m']
    p=np.loadtxt(case/'physical-accepted.node',skiprows=1,usecols=(1,2,3))[columns['node']]
    n=np.array(read(case/'support-check.json')['bed_normal']);dist=columns['samples']['native'][0]['depth_mm']/1000
    x=np.cross([0,1.,0],n);x/=np.linalg.norm(x);y=np.cross(n,x);axes=np.column_stack((x,y,n));probes=np.array([p,p+25e-6*x,p-25e-6*x,p+25e-6*y,p-25e-6*y])
    rows=[];bad=[];considered=0
    with zipfile.ZipFile(archive)as z:
        files=[name for name in z.namelist()if name.endswith('.obj')];obj=next(name for name in files if Path(name).stem=='FJ3384');data=z.read(obj)
        if hashlib.sha256(data).hexdigest()!=entry['member_sha256']:raise ValueError('Different scapula source')
        a,f=source_geometry(data)
        if not np.array_equal(f,tf)or np.max(abs(packed[:,:3]-a*.001))>3e-7:raise ValueError('Scapular source indices changed')
        M,tr=np.split(np.linalg.lstsq(np.column_stack([a,np.ones(len(a))]),world,rcond=None)[0],[3]);tr=tr[0]
        errors=np.linalg.norm(a@M+tr-world,axis=1);ss=np.linalg.svd(M,compute_uv=False)
        if errors.max()>1e-6 or ss.max()/ss.min()>1.0001 or np.linalg.det(M)<=0:raise ValueError('Non-rigid or unmatched source frame')
        req=(np.vstack((probes,probes+n*dist))-tr)@np.linalg.inv(M);lo=req.min(0)-.05;hi=req.max(0)+.05
        for name in files:
            with z.open(name)as inp:head=inp.read(1450).decode('ascii','replace')
            b=re.search(r'^# Bounds\(mm\): \(([^)]+)\)-\(([^)]+)\)',head,re.M)
            if not b:raise ValueError('Missing source bounds '+name)
            l=np.fromstring(b[1],sep=',');h=np.fromstring(b[2],sep=',')
            if np.any(l>hi) or np.any(h<lo):continue
            considered+=1;nm=re.search(r'^# English name\s*:\s*([^\r\n]+)',head,re.M|re.I);label=nm[1].strip() if nm else Path(name).stem;dat=z.read(name);pts,tri=source_geometry(dat)
            mesh=(pts@M+tr)@axes;origin=probes@axes;start=origin.copy();start[:,2]=mesh[:,2].min()-.01
            spans=[]
            for k,q in enumerate(hits(mesh,tri,0,start)):
                ds=sorted(t+start[k,2]-origin[k,2]for t,_ in q)
                ds=[t for i,t in enumerate(ds)if i==0 or t-ds[i-1]>1e-8]
                if len(ds)%2:
                    if any(-1e-6<t<dist+1e-6 for t in ds):bad.append([name,label,k,ds])
                    spans.append([]);continue
                spans.append([[1000*max(0.,u),1000*min(dist,v)] for u,v in zip(ds[::2],ds[1::2])if min(dist,v)-max(0.,u)>1e-8])
            if any(spans):rows.append(dict(id=Path(name).stem,name=label,sha256=hashlib.sha256(dat).hexdigest(),spans_mm=spans))
    out=dict(archive=sha,bone_payload=digest(bone),native=columns['native'],physical_geometry=native['accepted_frame'],source_anchor=entry,frame=dict(linear_m_per_mm=M.tolist(),translation_m=tr.tolist(),max_vertex_residual_m=errors.max(),singular_values=ss.tolist()),members_scanned=len(files),bounds_candidates=considered,rows=rows,ambiguous=bad,limit_mm=dist*1000,scope='Scapula-registered atlas comparison. Other atlas members have not been admitted as actual native muscles, masses or force carriers.')
    (w/'source-support-column.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('registered frame',errors.max(),'source hits',[(r['id'],r['name'],np.round(r['spans_mm'][0],2).tolist()) for r in rows if r['spans_mm'][0]],'ambiguous',len(bad))
if __name__=='__main__':run(Path.cwd())

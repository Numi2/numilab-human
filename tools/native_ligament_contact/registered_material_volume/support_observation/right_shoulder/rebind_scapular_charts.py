"""Rebind the two trapezius scapular charts to the measured scapula and evaluate the native payload."""
import json,struct,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from igl.copyleft import cgal
from compile_connected_joint import digest
from registration_stages import parts
from repair_registered_surface import state
from prepare_shoulder import read
from assemble_material import frames
from measure_assembly_contacts import canonical
from choose_emitted_shoulder import overlaps
from fit_registered_assembly import layer_surface
from trace_loaded_columns import hits

def run(w,common=False):
    case=w/'native-cavity-bound';cfg=read(case/'launch.json');a=cfg['argv'];raw=Path(a[a.index('--soft-tissue-payload')+1])
    src=read(w/'source-support-column.json');c=read(w/'bed-corridor.json');original=raw.read_bytes();blob=bytearray(original)
    magic,abi,nr,nb,nv,ni,_=struct.unpack_from('<8s6I',blob)
    if magic!=b'NHTISS4\x00' or abi!=5:raise ValueError('Wrong tissue asset')
    records=np.frombuffer(original,'<u4',nr*8,64).reshape(nr,8);offset=64+nr*32
    def binding(sid,body):
        start,count=records[records[:,6]==sid][0][:2].astype(int)
        ids=[i for i in range(start,start+count)if struct.unpack_from('<I',blob,offset+i*36)[0]==body]
        if len(ids)!=1:raise ValueError('Unregistered body')
        return offset+ids[0]*36
    target=binding(83,34);template=original[target:target+36]
    poses=frames(case);M=np.array(src['frame']['linear_m_per_mm']).T*1000;t=np.array(src['frame']['translation_m'])
    _,_,ow,_,aff=parts(raw,83);A,tr=aff[ow.index(34)];h=poses[-1][34];R=Rotation.from_quat(h['quaternion_xyzw']).as_matrix()
    if np.max(abs(R@A-M))>1e-6 or np.linalg.norm(R@tr+h['position_m']-t)>2e-6:raise ValueError('Donor template differs from measured source scapula')
    out=w/('scapular-common-charts' if common else 'scapular-charts');out.mkdir(exist_ok=True);dst=out/raw.name;edits={}
    for sid in (153,155):
        k=binding(sid,34);edits[sid]=[]
        if common:
            _,_,donors,_,maps=parts(raw,sid);oldA,oldt=maps[donors.index(34)]
            C=np.linalg.solve(oldA,A);z=np.linalg.solve(oldA,tr-oldt)
            for b,(B,u) in zip(donors,maps):
                j=binding(sid,b);orient=B@C;scale=np.cbrt(np.linalg.det(orient));rotation=orient/scale
                if abs(scale)<=1e-10 or np.max(abs(rotation.T@rotation-np.eye(3)))>1e-6:raise ValueError('Non-similarity change of source chart')
                struct.pack_into('<I8f',blob,j,b,*(B@z+u),*Rotation.from_matrix(rotation).as_quat(),scale);edits[sid].append(j)
        else:edits[sid].append(k)
        blob[k:k+36]=template
    dst.write_bytes(blob)
    if set(np.flatnonzero(np.frombuffer(original,'u1')!=np.frombuffer(blob,'u1')))-{q for group in edits.values()for k in group for q in range(k,k+36)}:raise ValueError('Changed unrelated source data')
    q=np.loadtxt(case/'physical-accepted.node',skiprows=1,usecols=(1,2,3))[c['node']]
    n=np.array(read(case/'support-check.json')['bed_normal']);x=np.cross([0.,1.,0],n);x/=np.linalg.norm(x);ax=np.column_stack((x,np.cross(n,x),n));org=np.array(read(w/'registered/registration.json')['origin'])
    record={};solids={};bounds=[]
    for sid in (153,155):
        v,wgt,owners,f,old=parts(raw,sid);_,_,_,_,new=parts(dst,sid)
        ix,sel,inv=np.unique(v,axis=0,return_index=True,return_inverse=True);f=np.ascontiguousarray(inv[f])
        if np.max(abs(wgt[sel][inv]-wgt))>1e-6:raise ValueError('Conflicting linked source seam')
        obs={};ss=[]
        for key,b in [('control',old),('scapula',new)]:
            tags=[];ds=[]
            for pose in poses:
                X=state(ix,wgt[sel],b,owners,pose).astype('f4').astype(float)
                tags.append(len(cgal.remesh_self_intersections(X,f,detect_only=True)[2]));ds.append(X)
            P=(X-org)*1000;s,meta=canonical(P,f)
            if s is None:raise ValueError('Invalid shoulder muscle')
            ss.append((s,P.min(0),P.max(0)));bounds.extend([P.min(0),P.max(0)])
            hh=hits(X@ax,f,sid,np.array([q@ax]))[0]
            obs[key]=dict(self_pairs=tags,volume_mm3=s.volume(),depth_crossings_mm=[a*1000 for a,b in hh if a<.15])
            if key=='control':before=ds
            else:obs['peak_displacements_m']=[float(np.linalg.norm(a-b,axis=1).max())for a,b in zip(before,ds)]
        record[sid]=obs;solids[sid]=ss
    neighbors=np.load(w/'registered/neighbors.npz');fixed=layer_surface(w.parent,org,np.min(bounds,axis=0),np.max(bounds,axis=0))
    for k in neighbors.files:
        if not k.endswith('-v'):continue
        typ,bank=map(int,k[:-2].split('-'))
        if not typ or (typ==51005 and bank in solids):continue
        X=neighbors[k];lo,hi=X.min(0),X.max(0)
        if not any(np.all(hi>=a)and np.all(lo<=b)for ss in solids.values()for _,a,b in ss):continue
        s,_=canonical(X,neighbors[k[:-1]+'f']);fixed[typ,bank]=s,lo,hi
    for sid,ss in solids.items():
        rows=[]
        for (typ,id),other in fixed.items():
            if not any(np.all(other[2]>=l)and np.all(other[1]<=h)for _,l,h in ss):continue
            if other[0]is None:rows.append(dict(kind=typ,bank=id,unresolved=True));continue
            rows.append(dict(kind=typ,bank=id,control=overlaps(ss[0],other),after=overlaps(ss[1],other)))
        record[sid]['surroundings']=rows
    mutual=[overlaps(solids[153][k],solids[155][k])for k in range(2)]
    report=dict(reference=digest(raw),candidate=digest(dst),atlas=src['archive'],bone=digest(Path(a[3])),pose=digest(case/'launch.json'),changed_binding_offsets=edits,scapula_template_offset=target,metrics=record,mutual_trapezius_mm3=mutual)
    (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for k,v in record.items(): print(k,v['control']['depth_crossings_mm'],v['scapula']['depth_crossings_mm'],v['scapula']['self_pairs'],[(r['bank'],round(r['after']-r['control']))for r in v['surroundings']if r.get('after',0)>r.get('control',0)+.001][:12])
if __name__=='__main__':run(Path.cwd(),'--common-source' in sys.argv)

"""Clip local support segments against the registered volumes and their original mass source."""
import json,sys
from pathlib import Path
import numpy as np
from build_bed_audit import sha

def intervals(p,t,labels,segments):
    # Clip barycentric half spaces of every bounding-box candidate, then union
    # intervals within each material. A tetrahedron is never counted twice.
    search=[];out=[{} for _ in segments]
    lo=np.minimum(segments[:,0],segments[:,1])-1e-10;hi=np.maximum(segments[:,0],segments[:,1])+1e-10
    for start in range(0,len(t),32768):
        a=p[t[start:start+32768]];mn=a.min(axis=1);mx=a.max(axis=1)
        for i,(left,right)in enumerate(segments):
            sel=np.flatnonzero(((mn<=hi[i])&(mx>=lo[i])).all(axis=1))
            if not len(sel):continue
            tet=a[sel];edges=(tet[:,1:]-tet[:,:1]).transpose(0,2,1)
            rhs=np.stack((left-tet[:,0],right-tet[:,0]),axis=2);b=np.linalg.solve(edges,rhs);b=np.concatenate((1-b.sum(axis=1,keepdims=True),b),axis=1)
            u=b[:,:,0];d=b[:,:,1]-u
            enter=np.zeros(len(sel));leave=np.ones(len(sel));ok=np.ones(len(sel),bool)
            for k in range(4):
                parallel=np.abs(d[:,k])<1e-14;ok &= ~parallel | (u[:,k]>=-1e-11)
                cross=np.divide(-u[:,k],d[:,k],out=np.zeros(len(sel)),where=~parallel)
                enter=np.maximum(enter,np.where(d[:,k]>1e-14,cross,0));leave=np.minimum(leave,np.where(d[:,k]<-1e-14,cross,1))
            for idx in np.flatnonzero(ok&(leave>enter)):
                g=out[i].setdefault(int(labels[start+sel[idx]]),[]);g.append((float(enter[idx]),float(leave[idx])))
    answer=[]
    for groups in out:
        totals={}
        for label,spans in groups.items():
            merged=[]
            for a,b in sorted(spans):
                if merged and a<=merged[-1][1]+1e-11:merged[-1][1]=max(merged[-1][1],b)
                else:merged.append([a,b])
            totals[str(label)]=sum(b-a for a,b in merged)
        answer.append(totals)
    return answer

def run(case,vol,rays,near,out):
    cfg=json.loads((case/'launch.json').read_bytes());line=json.loads(rays.read_bytes());closest=json.loads(near.read_bytes())
    pack=json.loads((case/'native-traces/accepted-geometry/step-1.receipt.json').read_bytes())['pack_file_sha256']
    if pack!=closest['pack'] or sha(case/'physical-accepted.node')!=line['origin_frame']:raise ValueError('Mismatched material and muscle geometry')
    raw=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');x=np.loadtxt(case/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
    lines=[];sites=[];n=np.array(line['bed_normal'])
    for col,nearest in zip(line['loaded_outer_nodes'],closest['columns']):
        if col['node']!=nearest['node'] or nearest['inside'] is not False:raise ValueError('Unknown muscle interior')
        O=x[col['node']];I=O+col['inner_skin_m']*n;D=np.array(nearest['deep_fat_point_m']);M=nearest['muscle_point_m']
        lines.extend(([O,I],[I,D],[D,M]));sites.append(col['node'])
    segments=np.array(lines)
    answers={'accepted_FEM':intervals(x,raw[:,:4]-1,raw[:,4],segments)}
    files=[]
    for k in ('NODES','TETS'):
        key='NUMI_H706_FRACTIONAL_SOURCE330_'+k;f=Path(cfg['environment'][key])
        if sha(f)!=cfg['environment'][key+'_SHA256']:raise ValueError('Unmatched mass source')
        files.append(f)
    src=np.loadtxt(files[0],skiprows=1,delimiter=',',usecols=(2,3,4,5,6,7));r=np.loadtxt(files[1],skiprows=1,delimiter=',',dtype='i4',usecols=(1,3,4,5,6))
    for frame,p in [('source330_current',src[:,:3]),('source330_rest',src[:,3:])]:answers[frame]=intervals(p,r[:,1:],r[:,0],segments)
    report=[]
    for i,node in enumerate(sites):
        rows=[]
        for j,name in enumerate(('outer_skin','underlying_fat','deep_fat_to_nearest_muscle')):
            k=i*3+j;length=float(np.linalg.norm(segments[k,1]-segments[k,0]));rows.append(dict(segment=name,length_m=length,covered_fraction={who:result[k]for who,result in answers.items()}))
        report.append(dict(node=node,segments=rows))
    out.write_text(json.dumps(dict(pose=pack,volume=sha(vol/'mesh.ele'),mass_sources=[sha(f)for f in files],sites=report),indent=2)+'\n')
if __name__=='__main__':run(*(Path(x).resolve()for x in sys.argv[1:]))

"""Independently certify all live material columns at the exported native endpoint."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import brentq
from run_bed_work import digest as sha

def inspect(receipt,columns,curves,foundation=None):
    data=json.loads(receipt.read_text());meta=data['skin_source_mapping']
    if data['physical_endpoint']!='accepted' or meta['influence_layout']!='vertex-major':raise ValueError("unexpected native capture")
    def records(key,dtype):
        item=meta[key];p=Path(item['path'])
        if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError("mapping changed: "+key)
        return np.memmap(p,mode='r',dtype=dtype)
    maps=records('vertex_map',np.dtype([('i','<u4',(4,)),('deform','<f4',(4,)),('basis','<f4',(4,))]))
    w=records('influences',np.dtype([('p','<f4',(4,)),('normal','<f4',(4,)),('body','<u4',(4,))]))
    c=np.genfromtxt(columns,delimiter=',',names=True)
    c=c[c['area_m2'].astype('f4')>0];v=c['vertex'].astype(int)
    origin=meta['influence_first_index']
    poses=[data['initial_anatomical_registration']['body_poses'],data['accepted_body_poses']]
    transforms=[]
    for entries in poses:
        rot=np.full((max(e['body_index'] for e in entries)+1,3,3),np.nan);tr=np.full((len(rot),3),np.nan)
        for e in entries:
            b=e['body_index'];rot[b]=Rotation.from_quat(e['quaternion_xyzw']).as_matrix()
            tr[b]=e['position_m']
        transforms.append((rot,tr))
    z=np.empty((2,len(v)))
    for s in range(0,len(v),256):
        rows=v[s:s+256];first,count=maps['i'][rows,:2].astype('i8').T;first-=origin
        starts=np.r_[0,np.cumsum(count)[:-1]]
        r=np.arange(sum(count))+np.repeat(first-starts,count)
        if min(r)<0 or max(r)>=len(w):raise ValueError("invalid binding")
        q=w[r];b=q['body'][:,0];p=q['p'].astype(float)
        if np.max(abs(np.add.reduceat(p[:,3],starts)-1))>1e-5:raise ValueError("skin weights")
        for j,(rot,tr) in enumerate(transforms):
            xyz=np.einsum('ij,ij->i',rot[b,2],p[:,:3])+tr[b,2]
            z[j,s:s+len(rows)]=np.add.reduceat(xyz*p[:,3],starts)
    deviation=float(np.max(abs(c['z_m']-z[0])))
    if not np.isfinite(z).all() or deviation>5e-6:raise ValueError(('opening source replay discrepancy',deviation))
    law=json.loads(curves.read_text())['material']
    props=[(r['mu_pa'],r['log_volume_coefficient_pa']) for r in law['materials']]
    def stress(s,mu,lam):return mu*(1/s-s)-lam*np.log(s)/s
    cap=min(stress(2/3,*pr) for pr in props)
    ss,sf=[brentq(lambda s:stress(s,*pr)-cap,2/3,1.) for pr in props]
    limit=c['skin_m']*(1-ss)+c['fat_m']*(1-sf)
    depth=-z[1]
    series={}
    if foundation is not None:
        bed=json.loads(Path(foundation).read_text())
        if bed['material']!=sha(curves) or not np.isclose(cap,bed['pressure_limit_pa']):raise ValueError('different tissue law')
        pressure=np.linspace(0,cap,1025)
        for corner in bed['pressure_pa'][1:-1]:pressure[np.argmin(abs(pressure-corner))]=corner
        eps=np.array([[1-brentq(lambda s:stress(s,*pr)-pp,2/3,1.) for pr in props] for pp in pressure])
        mattress=bed['foam_height_m']*np.interp(pressure,bed['pressure_pa'],bed['compression_fraction'])
        depth+=bed['free_surface_z_m']; limit+=mattress[-1]
        lo=np.zeros(len(v),int);hi=lo+1024
        def closure(k):return c['skin_m']*eps[k,0]+c['fat_m']*eps[k,1]+mattress[k]
        for _ in range(10):
            m=(lo+hi)//2;below=closure(m)<depth;lo=np.where(below,m,lo);hi=np.where(below,hi,m)
        p=pressure[lo]+(pressure[hi]-pressure[lo])*(np.maximum(depth,0)-closure(lo))/(closure(hi)-closure(lo))
        force=c['area_m2']*p
        series=dict(foundation=sha(foundation),normal_reaction_at_endpoint_N=float(force.sum()),
                    peak_endpoint_pressure_Pa=float(p.max()),loaded_area_m2=float(c['area_m2'][depth>0].sum()),
                    peak_mattress_strain=float(np.interp(p.max(),bed['pressure_pa'],bed['compression_fraction'])),
                    backing_clearance_min_m=float(np.min(z[1]-bed['backing_z_m'])))
    excess=depth-limit
    # Retain the signed values even when they are inside the observed opening
    # replay deviation. Report both the raw and numerically resolved counts.
    hits=np.flatnonzero(excess>0)
    return dict(**series,step=data['accepted_step'],time_s=data['accepted_time_s'],physical_root=data['accepted_root_fingerprint_hex'],
        pose=sha(receipt),columns=sha(columns),map=meta['vertex_map']['sha256'],influences=meta['influences']['sha256'],
        checked_columns=len(v),opening_replay_error_m=deviation,
        count_beyond_material_domain=len(hits),resolved_count=int(sum(excess>deviation)),largest_signed_excess_m=float(max(excess)),
        witnesses=[dict(vertex=int(v[i]),region=int(c['row'][i]),physical_depth_m=float(depth[i]),allowed_depth_m=float(limit[i]))
                   for i in sorted(hits,key=lambda k:-excess[k])[:12]])

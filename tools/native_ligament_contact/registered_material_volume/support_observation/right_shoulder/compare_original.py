"""Compare original BodyParts3D faces with exact emitted donor reconstruction on the current rig."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from audit_registered_skeleton import source_geometry
from prepare_material_interface import solid
from inspect_shoulder_source import nearest
from compile_connected_joint import digest
from trace_registrations import check
from registration_stages import parts


def barycentric(pt,tri):
    edges=tri[:,1:]-tri[:,:1];rel=pt-tri[:,0]
    uv=np.linalg.solve(edges@edges.transpose(0,2,1),edges@rel[:,:,None])[:,:,0]
    return np.column_stack((1-uv.sum(1),uv))


def compare(j,atlas,tool,lib,out):
    out.mkdir(exist_ok=True)
    case=j/'posterior-interface-082/native-pair';cfg=json.loads((case/'launch.json').read_bytes());a=cfg['argv'];payload=Path(a[a.index('--soft-tissue-payload')+1])
    rc=next((case/'native-traces/accepted-geometry').glob('*.receipt.json'));frame=json.loads(rc.read_bytes());poses={r['body_index']:r for r in frame['accepted_body_poses']}
    rec=json.loads((atlas/'recovery.json').read_bytes());rows={};bodies={};soup={}
    for k,row in rec['members'].items():
        sid=int(k);obj=atlas/Path(row['member']).name
        if digest(obj)!=row['member_sha256']:raise ValueError('Modified atlas member')
        raw,f=source_geometry(obj.read_bytes());x,inv=np.unique(raw,axis=0,return_inverse=True);f=inv[f];x*=.001
        v,w,owners,st,aff=parts(payload,sid)
        used=np.unique(st);v,w,st=v[used],w[used],np.searchsorted(used,st)
        pos,dist,face=nearest((np.ascontiguousarray(v),np.ascontiguousarray(st,'i4')),x,lib)
        weights=np.einsum('ij,ijk->ik',barycentric(pos,v[st[face]]),w[st[face]])
        kd,ix=cKDTree(v).query(x);match=kd<4e-7
        weights[match]=w[ix[match]]
        if weights.min()< -2e-6 or np.max(abs(weights.sum(1)-1))>2e-6:raise ValueError('Unregistered weights')
        # Preserve measured exact correspondences. Projection on changed facets is
        # a diagnostic reconstruction, not an assertion of source attachment.
        weights=np.maximum(weights,0);weights/=weights.sum(1)[:,None]
        y=np.zeros_like(x)
        for i,b in enumerate(owners):
            R=Rotation.from_quat(poses[b]['quaternion_xyzw']).as_matrix()
            y+=weights[:,i,None]*(x@(R@aff[i][0]).T+poses[b]['position_m']+R@aff[i][1])
        source_geo,_,meta=solid(x*1000,f,str(sid))
        live_geo,_,lm=solid(y*1000,f,str(sid))
        report=dict(row=row,vertices=len(x),faces=len(f),exact_fraction=match.mean(),projection_max_m=dist.max(),
            geometry_meta=meta,live_geometry_meta=lm,stored=check((x-x.mean(0))*1000,f,tool,out/(k+'-stored')),
            reconstructed=check((y-y.mean(0))*1000,f,tool,out/(k+'-native')))
        soup[k]=[source_geo,live_geo];rows[k]=report
        np.savez_compressed(out/(k+'.npz'),x=x,faces=f,weights=weights,current=y,body_ids=owners,
            source_binding_face=face,source_binding_bary=barycentric(pos,v[st[face]]),exact=match)
        print(k,len(f),round(match.mean(),3),meta['mesh'],report['stored']['witnesses'],report['reconstructed']['witnesses'],flush=True)
    import itertools
    pairs=[]
    for i,k in itertools.combinations(soup,2):
        a,b=soup[i],soup[k];values=[]
        for u,v in zip(a,b):
            values.append(float(abs((u^v).volume())) if str(u.status())==str(v.status())=='Error.NoError' else None)
        pairs.append(dict(ids=[int(i),int(k)],original_mm3=values[0],projected_mm3=values[1]))
    (out/'comparison.json').write_text(json.dumps(dict(archive=rec['archive_sha256'],payload=digest(payload),capture=digest(rc),rows=rows,pairs=pairs),indent=2)+'\n')
if __name__=='__main__':compare(*(Path(a).resolve()for a in sys.argv[1:]))

"""Compile the retained material edge branch against exact emitted body-local vertices."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from certify_material_branch import Bone, coords, optimize, size, unit
from shoulder_inputs import digest, old
from source_pair_audit import inputs
from numilab_human.capsular_material_registration import read_material_banks, emitted_body_vertices

def run(launch,anchors,motion,bone,sites,branches,log,out):
    _,poses,_,_=inputs(launch,anchors,motion)
    actual=next(json.loads(s.partition('=')[2]) for s in log.open() if s.startswith('registered_ligament_actual_pose_json='))
    assert actual['bonePayloadSha256']==digest(bone)
    actual=np.array(actual['bodyPoses'])
    site=json.loads(sites.read_text()); assert site['candidate_sha256']==digest(bone)
    old.audit.ag.BONES=bone
    _,raw,_,_,_,_,_,cat,_,vo,io=old.audit.ag.read_payloads()
    meshes=old.load_local_meshes(raw,cat,vo,io)
    _,_,recs,sv,_=read_material_banks(bone)
    branch=json.loads(branches.read_text())
    output=[]
    for name in site['rows']:
        side=name['side'];attach=name['attachments'];a,b=attach
        sid=a['stable'];rec=recs[sid];v=emitted_body_vertices(sv[rec[1]:rec[1]+rec[2]],rec)
        template=Bone((v,meshes[sid][1]),np.array([0,0,0,0,0,0,1.]),sid)
        def transform(pose):
            q=pose[a['body']];k=pose[b['body']]
            return Rotation.from_quat(q[3:]).inv().apply(k[:3]-q[:3]+Rotation.from_quat(k[3:]).apply(b['local_m']))
        end=transform(actual)
        match=min((-1,-2,-3),key=lambda i:np.linalg.norm(transform(poses[i])-end))
        print('side, nearest pose, delta',side,match,np.linalg.norm(transform(poses[match])-end),flush=True)
        assert np.linalg.norm(transform(poses[match])-end)<1e-7
        seeds=next(row['edges'] for row in branch if row['side']==side and row['step']==match and row['branch']=='continued')
        assert all(r[0]==sid for r in seeds)
        es=optimize({sid:template},[a['local_m'],end],seeds);p=coords({sid:template},[a['local_m'],end],es)
        last=max(i for i,(_,e,t) in enumerate(es) if t<1e-9 or t>1-1e-9)
        it=es[last]; pivot=p[last+1];fixed=size(p[:last+2]);remaining=es[last+1:]
        print('prefix',last+1,'moving',len(remaining),'length',size(p),'pivot',it,flush=True)
        dirs=unit(np.diff(p,axis=0))
        def feature(i):
            _,e,t=es[i]
            if 1e-9<t<1-1e-9:return set(template.ef[e])
            vv=template.edges[e,int(t>.5)]
            return set(np.flatnonzero((template.f==vv).any(axis=1)))
        edges=[]
        for i,(_,e,t) in enumerate(remaining,last+1):
            common=list(map(int,sorted(feature(i)&feature(i-1))))
            print('edge',e,'common',common,'cone',template.normals[template.ef[e]]@(dirs[i]-dirs[i+1]))
            edges.append(dict(edge=e,vertices=template.v[template.edges[e]].tolist(),normal=template.normals[template.ef[e]].tolist(),t=t,common=common))
        output.append(dict(side=side,step=match,body=a['body'],stable=sid,anchor=a['local_m'],target=b['local_m'],firstTriangle=a['triangle'],secondTriangle=b['triangle'],pivot=pivot.tolist(),prefix=fixed,terminal=end.tolist(),reference=size(p),edges=edges,prefixPoints=p[:last+2].tolist(),prefixEdges=es[:last+1],pivotNormals=template.normals[sorted(feature(last))].tolist(),terminalNormal=(Rotation.from_quat(actual[a['body']][3:]).inv()*Rotation.from_quat(actual[b['body']][3:])).apply(Bone((emitted_body_vertices(sv[recs[b['stable']][1]:sum(recs[b['stable']][1:3])],recs[b['stable']]),meshes[b['stable']][1]),np.array([0,0,0,0,0,0,1.]),b['stable']).normals[b['triangle']]).tolist()))
    out.write_text(json.dumps(dict(bones_sha256=digest(bone),sites_sha256=digest(sites),branches_sha256=digest(branches),routes=output),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))

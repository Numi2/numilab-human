"""Prepare one registered right-shoulder assembly from the pinned native state."""
import json
import mmap
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from inspect_shoulder_source import nearest
from prepare_infraspinatus import dets, exterior
from registration_stages import parts
from build_carpal_cavity import boundary

MOVING=(75,79,83,85,87,89)
LABELS={75:'DELT2',79:'DELT3',83:'INFSP',85:'SUBSC',87:'TMAJ',89:'TMIN'}


def read(p):
    return json.loads(p.read_bytes())


def tetrahedralize(x,f,tool,d):
    d.mkdir(exist_ok=True)
    lines=[f'{len(x)} 3 0 0']
    lines.extend(f'{i+1} '+' '.join(f'{v:.17g}' for v in r) for i,r in enumerate(x))
    lines.append(f'{len(f)} 1')
    for i,tri in enumerate(f):
        lines.extend([f'1 0 {i+1}','3 '+' '.join(str(q+1)for q in tri)])
    (d/'source.poly').write_text('\n'.join(lines+['0','0'])+'\n')
    with (d/'tetgen.log').open('w') as log:
        subprocess.run([tool,'-pYq1.4Q','source.poly'],check=True,cwd=d,stdout=log,stderr=subprocess.STDOUT)
    pts=np.loadtxt(d/'source.1.node',skiprows=1,usecols=(1,2,3))
    cells=np.loadtxt(d/'source.1.ele',skiprows=1,usecols=(1,2,3,4),dtype=int)-1
    if not np.allclose(pts[:len(x)],x,atol=2e-12,rtol=0) or np.min(dets(pts,cells))<=0:
        raise ValueError('Unregistered source or nonpositive cage')
    if set(map(tuple,exterior(cells)))!=set(map(tuple,np.sort(f,axis=1))):
        raise ValueError('Changed anatomical boundary')
    return pts,cells


def setup(j,registry,tool,lib,out):
    out.mkdir(exist_ok=True)
    case=j/'posterior-interface-082/native-pair'
    cfg=read(case/'launch.json');a=cfg['argv']
    src=Path(a[a.index('--soft-tissue-payload')+1]);tendon=Path(a[a.index('--tendon-payload')+1])
    manifest=tendon.with_suffix('.manifest.json');endpoints=read(manifest)['endpoints']
    for p in (src,tendon,manifest,Path(a[2])):
        if digest(p)!=cfg['pins'][str(p)]:raise ValueError('Unmatched attachment or material')
    rc=next((case/'native-traces/accepted-geometry').glob('*.receipt.json'));d=read(rc)
    poses={r['body_index']:r for r in d['accepted_body_poses']};pack=Path(d['accepted_pack_path'])
    if digest(pack)!=d['pack_file_sha256']:raise ValueError('Changed accepted capture')
    api=runpy.run_path(str(registry))
    with pack.open('rb')as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as buf:
        meshes={key:(x.copy(),f.copy()) for key,_,x,f in api['primitives'](buf,poses)}
    meta={}
    location=np.vstack([meshes[(51005,k)][0] for k in MOVING])
    origin=location.mean(0)
    lo=location.min(0)-.025;hi=location.max(0)+.025
    for sid in MOVING:
        raw,weights,donors,srcface,aff=parts(src,sid)
        used=np.unique(srcface)
        current,faces=meshes[(51005,sid)]
        if not np.array_equal(np.searchsorted(used,srcface),faces):raise ValueError('Changed facet ordering')
        _,idx,inv=np.unique(raw[used],axis=0,return_index=True,return_inverse=True)
        xf=current[idx];ff=inv[faces]
        if np.max(abs(xf[inv]-current))>2e-7:raise ValueError('Source seam tears apart in pose')
        try:
            pts,cells=tetrahedralize((xf-origin)*1000,ff,tool,out/f'mesh-{sid}')
        except (ValueError,subprocess.CalledProcessError) as err:
            print('unfit native surface',sid,str(err)[-100:],flush=True)
            meta[sid]=dict(failed=True,reason=str(err))
            continue
        ends=[e for e in endpoints if e.get('muscle')==LABELS[sid]]
        if {e['endpoint'] for e in ends}!={'origin','insertion'}:raise ValueError('Incomplete anatomical endpoints')
        keep=set();anchors=[]
        for e in ends:
            body=e['body_index'];pose=poses[body];R=Rotation.from_quat(pose['quaternion_xyzw'])
            surface=e.get('surface') or {}
            reference=np.array([e['resolved_local_point_m']]+list(surface.get('node_local_points_m',[])))
            q=R.apply(reference)+pose['position_m']
            closest,gap,ids=nearest((np.ascontiguousarray(xf),np.ascontiguousarray(ff,'i4')),q,lib)
            support=np.unique(ff[ids]);keep.update(support.tolist())
            anchors.append(dict(role=e['endpoint'],body=body,node=e['source_site_index'],mode=e['attachment_mode'],
                                registered_site_m=q[0].tolist(),registered_points_m=q.tolist(),surface_faces=ids.tolist(),
                                closest_points_m=closest.tolist(),distances_m=gap.tolist(),fixed_skin_of_muscle_nodes=support.tolist()))
        for k in keep:
            if k>=len(xf):raise ValueError('Anchor outside source surface')
        # Blended rest-to-native linear map transports local edits into source coordinates.
        M=np.array([Rotation.from_quat(poses[b]['quaternion_xyzw']).as_matrix()@matrix for b,(matrix,_) in zip(donors,aff)])
        jac=np.einsum('vb,bij->vij',weights[used[idx]],M)
        if np.min(np.linalg.svd(jac,compute_uv=False)[:,-1])<.2:raise ValueError('Noninvertible skeletal registration')
        saved=out/f'bank-{sid}.npz'
        np.savez_compressed(saved,x=pts,t=cells,f=ff,surface_n=len(xf),pins=np.array(sorted(keep)),source_used=used,
                            seam_to_welded=inv,source_to_native=jac,origin=origin)
        meta[sid]=dict(cells=len(cells),surface_nodes=len(xf),faces=len(ff),nodes=len(pts),anchors=anchors,sha256=digest(saved))
    obst={}
    for (kind,stable),(v,f) in meshes.items():
        if kind==51007 or kind==51005 and stable in MOVING:continue
        if np.any(v.max(0)<lo) or np.any(v.min(0)>hi):continue
        obst[f'{kind}-{stable}-v']=(v-origin)*1000
        obst[f'{kind}-{stable}-f']=f
    # Cover physical fat and skin in a padded search box, including every tet
    # whose bounds intersect the box. The local cut is outside the fitting domain.
    vdir=j/'conditioned/witness-material/volume'
    x=np.loadtxt(case/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
    raw=np.loadtxt(vdir/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    t=raw[:,:4]-1
    for role in (15,16):
        selected=[]
        for offset in range(0,len(t),32768):
            ti=t[offset:offset+32768];p=x[ti]
            use=(raw[offset:offset+len(ti),-1]==role)&np.all((p.max(1)>=lo-.002)&(p.min(1)<=hi+.002),axis=1)
            if use.any():selected.append(ti[use])
        cells=np.concatenate(selected);faces=boundary(cells)
        ids,ix=np.unique(faces,return_inverse=True)
        obst[f'0-{role}-v']=(x[ids]-origin)*1000
        obst[f'0-{role}-f']=ix.reshape(-1,3)
    np.savez_compressed(out/'neighbors.npz',**obst)
    r=dict(case=digest(case/'launch.json'),capture=digest(rc),surface=digest(src),tendon=digest(tendon),tendon_manifest=digest(manifest),
           actuator_payload=digest(Path(a[2])),tetgen=digest(tool),interface_library=digest(lib),origin=origin.tolist(),
           obstacles=digest(out/'neighbors.npz'),mesh=digest(vdir/'mesh.ele'),accepted_material=digest(case/'physical-accepted.node'),muscles=meta)
    (out/'registration.json').write_text(json.dumps(r,indent=2)+'\n')
    print([(k,s.get('cells'),[round(e['distances_m'][0]*1000,1) for e in s.get('anchors',[])])for k,s in meta.items()],len(obst)//2,flush=True)

if __name__=='__main__':setup(*(Path(s).resolve()for s in sys.argv[1:]))

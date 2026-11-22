"""Check the opening surface against the physical skin before counting anatomical crossings."""
import csv, hashlib, json, mmap, re, runpy, subprocess, sys
from contextlib import contextmanager
from verify_bonded_native import matched_horizon
from pathlib import Path
import numpy as np
from material_context import sha

@contextmanager
def read_opening(case,expected):
    p=case/'opening.mrvpack'
    if p.exists():
        if sha(p)!=expected:raise ValueError('Opening mesh changed')
        with p.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as buf:yield buf
    else:
        z=case/'opening.mrvpack.zst'
        recovery=json.loads((case/'opening-recovery.json').read_text())
        if sha(z)!=recovery['verified_member_sha256']['opening.mrvpack.zst']:raise ValueError('Different opening archive')
        buf=subprocess.check_output(['zstd','-dq','-c',str(z)])
        if hashlib.sha256(buf).hexdigest()!=expected:raise ValueError('Different decompressed opening')
        yield buf

def check(case,reader,tools,scan=False,observed=None):
    root=case/'three'; launch=json.loads((root/'launch.json').read_text());env=launch['environment']
    identity=json.loads((case/'opening.json').read_text())
    posefile=root/'registered-initial-bodies.json'
    if not posefile.exists():posefile=Path(launch['argv'][4]).parent/posefile.name
    p=json.loads(posefile.read_text())
    if sha(posefile)!=identity['poses'] or sha(root/'launch.json')!=identity['launch']:
        raise ValueError('Opening is not from this release')
    pose_kind='registration'
    if observed is not None:
        observed_launch=json.loads((observed/'launch.json').read_text())
        config=matched_horizon(launch); actual=matched_horizon(observed_launch)
        for cfg in (config,actual):
            for key in ('NM_FEM_NEWTON_TRACE_ROOT','NM_FEM_BACKTRACKING_TRACE_ROOT','NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT','NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT'):
                cfg[3].pop(key,None)
        if config!=actual or env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']!=observed_launch['environment']['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']:
            raise ValueError('Different physical launch')
        ident=json.loads((case/'observed-owner/identities.json').read_text())
        if observed_launch['pins'][observed_launch['argv'][0]]!=ident['binary_sha256'] or sha(case/'observed-owner/numilab_human_myosim_visual_probe.mm')!=ident['source']:
            raise ValueError('Different native pose emitter')
        posefile=observed/'native-initial-bodies.json'
        p=json.loads(posefile.read_text())
        pose_kind='executed'
        module=case/'audit_registered_initial_envelope.py'
        root_error=runpy.run_path(str(module))['release_root_error'](p['all_bodies'],observed/'stdout.log')
        if root_error is None or root_error>2e-6:raise ValueError(('Native pose differs from released root',root_error))
    poses={r['body_index']:{'position_m':r['current']['position_world_m'],'quaternion_xyzw':r['current']['orientation_world_xyzw']} for r in p['all_bodies']}
    if sorted(poses)!=list(range(157)):raise ValueError('Incomplete articulated pose')
    src=Path(env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE'])
    pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    face=Path(env[pre+'SURFACE_O_FACES_CSV']);mp=Path(env[pre+'COMBINED_NODE_MAP_CSV'])
    if sha(src)!=launch['pins'][str(src)] or sha(face)!=env[pre+'SURFACE_O_FACES_SHA256'] or sha(mp)!=env[pre+'COMBINED_NODE_MAP_SHA256']:
        raise ValueError('Different registered surface')
    t=np.loadtxt(face,delimiter=',',skiprows=1,usecols=(1,2,3),dtype=np.int32)
    out=np.full(t.max()+1,-1,dtype=int)
    with mp.open() as f:
        for row in csv.DictReader(f):
            if row['registered_surface_layer']=='O':
                i=int(row['registered_surface_vertex_row'])
                if out[i]!=-1:raise ValueError('Duplicate outer surface vertex')
                out[i]=int(row['joint_node_id'])-1
    if out.min()<0:raise ValueError('Missing surface vertex')
    target=np.loadtxt(src,skiprows=1,usecols=(1,2,3))[out]
    sys.path.insert(0,str(reader.parent/'src')); audit=runpy.run_path(str(reader))
    with read_opening(case,identity['pack']) as raw:
        ((_,_,x,tris),)=audit['primitives'](raw,poses,skin_only=True)
        if not np.array_equal(t,tris):raise ValueError('Different skin topology')
        error=np.linalg.norm(x.astype(float)-target,axis=1)
        stats=dict(physical_skin_sha256=sha(src),pack_sha256=identity['pack'],opening_pose_sha256=sha(posefile),pose_source=pose_kind,
                   physical_outer_vertices=len(out),max_opening_skin_error_m=float(max(error)),rms_opening_skin_error_m=float(np.sqrt(np.mean(error**2))))
        dest=case/(pose_kind+'-parity.json');dest.write_text(json.dumps(stats,indent=2)+'\n')
        if scan:
            if max(error)>2e-6:raise ValueError('The opening asset differs from the released physical outer surface')
            runpy.run_path(str(tools/'audit_registered_initial_envelope.py'))['scan_banks'](
                audit,raw,poses,x,tris,stats,case/(pose_kind+'-envelope'),'initial-envelope.json',check_skin=True)
    return stats
if __name__=='__main__':
    case,reader,work=(Path(x).resolve() for x in sys.argv[1:4])
    tail=sys.argv[4:]
    observed=Path(tail[tail.index('--executed')+1]).resolve() if '--executed' in tail else None
    if '--scan' in tail:(case/(('executed' if observed else 'registration')+'-envelope')).mkdir(exist_ok=True)
    print(check(case,reader,work,'--scan' in tail,observed))

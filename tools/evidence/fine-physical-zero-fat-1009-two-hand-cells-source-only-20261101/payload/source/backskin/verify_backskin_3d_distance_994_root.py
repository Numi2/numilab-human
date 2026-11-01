#!/usr/bin/env python3
"""Independent lattice/Lipschitz Hausdorff replay for the 992 backskin I patch."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

H = Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
PARENT_STATE = H / 'all-quality-984-retained-I-quad-qualified-root/verified-state.json'
CANDIDATE_STATE = H / 'all-quality-992-backskin-cap-candidate-root/candidate-state-proposal.json'
EXPECTED = {
    'parent_state_sha256': 'ff37d39728b34d91887060f8742d54121a4ecb69c76dbc2f89f2b88bb9e0781c',
    'candidate_state_sha256': 'b9a45bb0b1a467839dd6ca0f7abe049453d28ffe0a343d368778d41958bc9733',
    'parent_surface_sha256': '47899d3bfd0ec0b698aa346a6b37241281fabdbd38b258985c2eb1e2823137a1',
    'candidate_surface_sha256': 'e12a028f2c52260b96d7630a128fa7a4635d073cb157b7d17ae6ff4b2b77588d',
}
OLD_FACES = np.array([[57482,60059,59961],[57482,59961,59941]], dtype=np.int64)
NEW_FACES = np.array([[57482,60059,59941],[59961,59941,60059]], dtype=np.int64)
FACE_ROWS = np.array([52337,52342], dtype=np.int64)
CAP_M = 50e-6
SAMPLE_PAD_M = 5e-9
RADIUS_PAD_M = 2e-12
MAX_BATCH_POINTS = 64_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def point_triangle_distance(P: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Euclidean distance from each row of P to a closed 3D triangle."""
    a,b,c = np.asarray(tri, dtype=np.float64)
    ab=b-a; ac=c-a; n=np.cross(ab,ac)
    n2=float(np.dot(n,n))
    if not np.isfinite(n2) or n2 <= 0.0:
        raise ValueError('degenerate target triangle')
    pa=P-a
    signed=pa@n
    proj=P-(signed/n2)[:,None]*n
    v0=ab; v1=ac; v2=proj-a
    d00=float(np.dot(v0,v0)); d01=float(np.dot(v0,v1)); d11=float(np.dot(v1,v1))
    d20=v2@v0; d21=v2@v1
    den=d00*d11-d01*d01
    if den <= 0 or not np.isfinite(den):
        raise ValueError('invalid target triangle metric')
    v=(d11*d20-d01*d21)/den
    w=(d00*d21-d01*d20)/den
    u=1.0-v-w
    inside=(u>=0.0)&(v>=0.0)&(w>=0.0)
    plane=np.abs(signed)/math.sqrt(n2)
    out=np.full(len(P), np.inf, dtype=np.float64)
    out[inside]=plane[inside]
    for x,y in ((a,b),(b,c),(c,a)):
        e=y-x; e2=float(np.dot(e,e))
        if e2 <= 0.0:
            raise ValueError('zero target edge')
        t=np.clip(((P-x)@e)/e2, 0.0, 1.0)
        q=P-(x+t[:,None]*e)
        d=np.sqrt(np.einsum('ij,ij->i',q,q))
        np.minimum(out,d,out=out)
    return out


def _barycentric_chunks(tri: np.ndarray, n: int):
    a,b,c=np.asarray(tri,dtype=np.float64)
    # Each source microtriangle has edge lengths bounded by max(original edge)/n.
    row_count=max(1, MAX_BATCH_POINTS//(n+1))
    for i0 in range(0,n+1,row_count):
        ii_parts=[]; jj_parts=[]
        for i in range(i0,min(n+1,i0+row_count)):
            j=np.arange(0,n-i+1,dtype=np.int32)
            ii_parts.append(np.full(j.shape,i,dtype=np.int32)); jj_parts.append(j)
        ii=np.concatenate(ii_parts).astype(np.float64)/float(n)
        jj=np.concatenate(jj_parts).astype(np.float64)/float(n)
        yield a+ii[:,None]*(b-a)+jj[:,None]*(c-a)


def directed_grid_bound(source_tris: np.ndarray, target_tris: np.ndarray, n: int) -> dict:
    source_tris=np.asarray(source_tris,dtype=np.float64)
    target_tris=np.asarray(target_tris,dtype=np.float64)
    face_rows=[]
    global_sample_max=-1.0
    global_sample_arg=None
    global_radius=0.0
    samples=0
    for fi,tri in enumerate(source_tris):
        lengths=[np.linalg.norm(tri[1]-tri[0]),np.linalg.norm(tri[2]-tri[1]),np.linalg.norm(tri[0]-tri[2])]
        radius=float(max(lengths))/float(n)
        local_max=-1.0; local_arg=None; local_count=0
        for P in _barycentric_chunks(tri,n):
            d=np.full(len(P),np.inf,dtype=np.float64)
            for tgt in target_tris:
                np.minimum(d,point_triangle_distance(P,tgt),out=d)
            k=int(np.argmax(d)); m=float(d[k])
            if m>local_max: local_max=m; local_arg=P[k].copy()
            local_count += len(P)
        upper=float(np.nextafter(local_max+SAMPLE_PAD_M+radius+RADIUS_PAD_M,math.inf))
        face_rows.append({
            'source_triangle_index':fi,'lattice_n':n,'sample_count':local_count,
            'sampled_max_distance_m':local_max,'sampled_max_point_m':local_arg.tolist(),
            'microtriangle_diameter_upper_m':radius+RADIUS_PAD_M,
            'sample_float64_outward_pad_m':SAMPLE_PAD_M,'upper_bound_m':upper,
        })
        samples += local_count
        if local_max>global_sample_max:
            global_sample_max=local_max; global_sample_arg=local_arg
        global_radius=max(global_radius,radius)
    upper=float(np.nextafter(global_sample_max+SAMPLE_PAD_M+global_radius+RADIUS_PAD_M,math.inf))
    return {
        'lattice_n':n,'sample_count':samples,'sampled_max_distance_m':global_sample_max,
        'sampled_max_point_m':global_sample_arg.tolist(),
        'microtriangle_diameter_upper_m':global_radius+RADIUS_PAD_M,
        'sample_float64_outward_pad_m':SAMPLE_PAD_M,'upper_bound_m':upper,
        'source_triangle_bounds':face_rows,
    }


def run_controls() -> dict:
    n=64
    A=np.array([0.,0.,0.]); B=np.array([1e-3,0.,0.]); C=np.array([0.,1e-3,0.])
    shift=np.array([0.,0.,20e-6])
    base=np.array([[A,B,C]])
    parallel=base+shift
    p=directed_grid_bound(base,parallel,n)
    q=directed_grid_bound(parallel,base,n)
    expected=20e-6
    if abs(p['sampled_max_distance_m']-expected)>2e-12 or abs(q['sampled_max_distance_m']-expected)>2e-12:
        raise AssertionError('parallel-plane control failed')
    if p['upper_bound_m'] < expected or q['upper_bound_m'] < expected:
        raise AssertionError('parallel-plane upper bound undercovers known distance')

    a=np.array([0.,0.,0.]); b=np.array([1.,0.,0.]); c=np.array([0.,1.,0.]); d=np.array([0.,0.,1.])
    flat=np.array([[a,b,c]])
    folded=np.array([[a,b,c],[a,b,d]])
    same=directed_grid_bound(flat,folded,n)
    reverse=directed_grid_bound(folded,flat,n)
    if same['sampled_max_distance_m']>2e-14:
        raise AssertionError('folded patch contained face control failed')
    if abs(reverse['sampled_max_distance_m']-1.0)>2e-12:
        raise AssertionError('folded patch reverse directed-distance control failed')
    if not (same['upper_bound_m']<0.05 and reverse['upper_bound_m']>=1.0):
        raise AssertionError('folded patch coverage control failed')
    return {
        'status':'PASS',
        'parallel_planes':{'expected_distance_m':expected,'forward':p,'reverse':q},
        'folded_patch':{'source_flat_to_target_folded':same,'source_folded_to_target_flat':reverse,
                        'known_reverse_distance_m':1.0},
    }


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    controls=run_controls()
    parent_sha=sha256(PARENT_STATE); cand_sha=sha256(CANDIDATE_STATE)
    if parent_sha!=EXPECTED['parent_state_sha256'] or cand_sha!=EXPECTED['candidate_state_sha256']:
        raise SystemExit('state pin mismatch')
    parent=json.loads(PARENT_STATE.read_text()); cand=json.loads(CANDIDATE_STATE.read_text())
    ps=parent['source_surfaces']['I']; cs=cand['source_surfaces']['I']
    if ps['sha256']!=EXPECTED['parent_surface_sha256'] or cs['sha256']!=EXPECTED['candidate_surface_sha256']:
        raise SystemExit('surface state SHA pin mismatch')
    pp=Path(ps['path']); cp=Path(cs['path'])
    if sha256(pp)!=ps['sha256'] or sha256(cp)!=cs['sha256']:
        raise SystemExit('surface archive SHA mismatch')
    with np.load(pp,allow_pickle=False) as z:
        pfaces=z['faces'].copy(); p32=z['points_world_m_f32'].copy(); p64=z['points_world_m'].copy()
    with np.load(cp,allow_pickle=False) as z:
        cfaces=z['faces'].copy(); c32=z['points_world_m_f32'].copy(); c64=z['points_world_m'].copy()
    if p32.dtype!=np.float32 or c32.dtype!=np.float32 or p32.shape!=c32.shape or p64.shape!=c64.shape:
        raise SystemExit('source point domain/dtype mismatch')
    if p32.tobytes()!=c32.tobytes() or p64.tobytes()!=c64.tobytes():
        raise SystemExit('source point coordinates are not bitwise fixed')
    if pfaces.shape!=cfaces.shape:
        raise SystemExit('source face array shape changed')
    if not np.array_equal(pfaces[FACE_ROWS],OLD_FACES):
        raise SystemExit('parent old source face rows mismatch')
    if not np.array_equal(cfaces[FACE_ROWS],NEW_FACES):
        raise SystemExit('candidate new source face rows mismatch')
    changed=np.flatnonzero(np.any(pfaces!=cfaces,axis=1))
    if not np.array_equal(changed,FACE_ROWS):
        raise SystemExit(f'unexpected source face changes: {changed[:20].tolist()}')
    old_tris=p32[OLD_FACES]
    new_tris=c32[NEW_FACES]
    cap=CAP_M
    directions={}
    chosen_n=None
    for n in (512,1024,2048,4096):
        d1=directed_grid_bound(old_tris,new_tris,n)
        d2=directed_grid_bound(new_tris,old_tris,n)
        directions={'old_to_new':d1,'new_to_old':d2}
        if max(d1['upper_bound_m'],d2['upper_bound_m'])<=cap:
            chosen_n=n; break
    passed=chosen_n is not None
    report={
        'schema':'independent_backskin_patch_3d_lattice_lipschitz_replay_v1',
        'status':'PASS' if passed else 'FAIL_CAP_NOT_CERTIFIED',
        'parent_state':{'path':str(PARENT_STATE),'sha256':parent_sha},
        'candidate_state':{'path':str(CANDIDATE_STATE),'sha256':cand_sha},
        'parent_surface':{'path':str(pp),'sha256':sha256(pp)},
        'candidate_surface':{'path':str(cp),'sha256':sha256(cp)},
        'source_point_arrays_bitwise_identical':True,
        'source_face_arrays_changed_rows':changed.tolist(),
        'face_rows':FACE_ROWS.tolist(),'old_faces':OLD_FACES.tolist(),'new_faces':NEW_FACES.tolist(),
        'points_world_m_f32_sha256':hashlib.sha256(p32.tobytes()).hexdigest(),
        'points_world_m_f64_sha256':hashlib.sha256(p64.tobytes()).hexdigest(),
        'method':{
            'description':'Independent uniform barycentric triangular lattice. Distance to the target triangle union is 1-Lipschitz. Every source microtriangle has diameter at most the source triangle longest edge divided by n; upper bound is maximum sampled point-to-target-union distance plus that diameter and explicit outward float64 padding.',
            'distance_function':'orthogonal plane projection when barycentrically inside, otherwise minimum clamped point-to-edge-segment distance; minimum across both target triangles',
            'candidate_lattice_n':chosen_n,'attempted_n':[512,1024,2048,4096][:([512,1024,2048,4096].index(chosen_n)+1 if chosen_n else 4)],
            'maximum_batch_points':MAX_BATCH_POINTS,'sample_float64_outward_pad_m':SAMPLE_PAD_M,'radius_outward_pad_m':RADIUS_PAD_M,
            'cap_m':cap,'directional_results':directions,
        },
        'controls':controls,
    }
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'chosen_n':chosen_n,'directions':{k:{'sampled_max_distance_m':v['sampled_max_distance_m'],'diameter_upper_m':v['microtriangle_diameter_upper_m'],'upper_bound_m':v['upper_bound_m'],'sample_count':v['sample_count']} for k,v in directions.items()},'out':str(out)},sort_keys=True))
    return 0 if passed else 2

if __name__=='__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Rebase the existing Skin target to pinned typed surfaces; no physical admission."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np
H=Path(__file__).resolve().parent
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def key(p):return np.asarray(p,np.float32).tobytes()
def cyc(f):
    a,b,c=map(int,f);return min((a,b,c),(b,c,a),(c,a,b))
def mapped_role(points,rows,role):
    p=np.asarray(role['points_world_m_f32'])
    if p.dtype!=np.float32 or not np.isfinite(p).all():raise ValueError('typed F32 point contract')
    if not np.array_equal(np.asarray(role['points_world_m'],np.float32),p):raise ValueError('typed emission mismatch')
    mp=np.asarray([rows[key(x)] for x in p],np.int64)
    if not np.array_equal(points[mp],p):raise ValueError('coordinate map mismatch')
    faces=np.asarray(role['faces'],np.int64)
    if faces.ndim!=2 or faces.shape[1]!=3 or np.any(faces<0) or np.any(faces>=len(mp)):raise ValueError('typed face indices')
    return mp[faces]
def oriented_target(points,anchor_boundary,oldroles,newroles):
    rows={key(p):i for i,p in enumerate(points)}
    if len(rows)!=len(points):raise ValueError('physical point coordinate alias requires explicit identity')
    anchor={tuple(sorted(map(int,f))):cyc(f) for f in anchor_boundary}
    if len(anchor)!=len(anchor_boundary):raise ValueError('duplicate anchor face')
    oldkeys=set();target=[];signs={}
    for role in ('O','I'):
        old,new=oldroles[role],newroles[role]
        if not np.array_equal(old['points_world_m_f32'],new['points_world_m_f32']):raise ValueError('surface coordinate reconstruction needs explicit transfer')
        of=mapped_role(points,rows,old);nf=mapped_role(points,rows,new)
        keys=[tuple(sorted(map(int,f))) for f in of]
        if oldkeys.intersection(keys) or not all(k in anchor for k in keys):raise ValueError('role anchor not unique/present')
        same=[cyc(f)==anchor[k] for f,k in zip(of,keys)]
        if not all(same) and any(same):raise ValueError('mixed role orientation')
        signs[role]=1 if all(same) else -1
        target.append(nf if all(same) else nf[:,[0,2,1]])
        oldkeys.update(keys)
    walls=np.asarray([f for f in anchor_boundary if tuple(sorted(map(int,f))) not in oldkeys],np.int64).reshape(-1,3)
    desired=np.vstack([*target,walls])
    if len({tuple(sorted(map(int,f))) for f in desired})!=len(desired):raise ValueError('duplicate target boundary')
    return desired,walls,signs
def controls():
    p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
    def role(f):return {'points_world_m':p.astype(np.float64),'points_world_m_f32':p,'faces':np.array(f,np.int64)}
    o=role([[0,2,1]]);i=role([[0,1,3]])
    bb=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],np.int64)
    d,w,s=oriented_target(p,bb,{'O':o,'I':i},{'O':o,'I':i})
    assert len(w)==2 and len(d)==4 and s=={'O':1,'I':1}
    bad=role([[0,2,1]]);bad['points_world_m_f32']=p.copy();bad['points_world_m_f32'][1,0]=np.nextafter(np.float32(1),np.float32(2))
    try:oriented_target(p,bb,{'O':o,'I':i},{'O':bad,'I':i})
    except ValueError:pass
    else:raise AssertionError('moved emitted coordinate admitted')
    wrong=bb.copy();wrong[0]=wrong[0,[0,2,1]]
    d,w,s=oriented_target(p,wrong,{'O':o,'I':i},{'O':o,'I':i})
    assert s['O']==-1 and tuple(d[0])==tuple(wrong[0])
    return {'controls':3,'status':'PASS_IDENTITY_MOVEMENT_REJECTION_AND_ORIENTATION'}
def run(a):
    tic=time.monotonic()
    for path,digest in ((a.input,a.input_sha256),(a.anchor,a.anchor_sha256),(a.O,a.O_sha256),(a.I,a.I_sha256)):
        if sha(path)!=digest:raise ValueError('input pin mismatch: '+str(path))
    oldpaths={'O':H/'native-apex-surface-floor-259-root/OI-batch001/typed-260-002/candidate-O-typed-current.npz',
      'I':H/'native-apex-connected-262-root/I-batch001/typed-264-001/candidate-I-typed-current.npz'}
    oldpins={'O':'6ea304253da218a4811fe0710ca79ba3b51242def2f62106d5ebb500317bac0d',
      'I':'645999a1226b15eb3c3edf1222d246e0df4155dbf2f517f4125146de48be66d6'}
    for r in oldpaths:
        if sha(oldpaths[r])!=oldpins[r]:raise ValueError('anchor role pin')
    v=load(a.input);anchor=load(a.anchor);p=np.asarray(v['points_world_m'])
    if p.dtype!=np.float32 or not np.isfinite(p).all():raise ValueError('physical F32 points')
    if not np.array_equal(p,anchor['points_world_m']):raise ValueError('physical prefix changed')
    desired,walls,signs=oriented_target(p,anchor['desired_boundary_faces'],
      {r:load(oldpaths[r]) for r in ('O','I')},{'O':load(a.O),'I':load(a.I)})
    if len(walls)!=90:raise ValueError('rim wall ownership changed')
    core=module('target284core',H/'remesh_current_surface_cavities_247_root.py')
    if core.closed(desired)!=(True,0):raise ValueError('target not closed')
    t=np.asarray(v['tetrahedra'],np.int64);actual=core.m.boundary(t)
    ak=core.sortedkeys(actual);dk=core.sortedkeys(desired)
    missing=np.asarray([f for f in desired if tuple(sorted(map(int,f))) not in ak],np.int64).reshape(-1,3)
    extra=np.asarray([f for f in actual if tuple(sorted(map(int,f))) not in dk],np.int64).reshape(-1,3)
    dc={tuple(sorted(map(int,f))):cyc(f) for f in desired}
    if any(cyc(f)!=dc[tuple(sorted(map(int,f)))] for f in actual if tuple(sorted(map(int,f))) in dk):raise ValueError('common winding mismatch')
    det,qual=core.m.m.measurements(p,t)
    if np.any(det<=0):raise ValueError('input nonpositive cells')
    v.update(boundary_faces=actual,combinatorial_boundary_faces=actual,desired_boundary_faces=desired,
      desired_missing_faces=missing,current_extra_boundary_faces=extra)
    a.output.mkdir(parents=True,exist_ok=False)
    out=a.output/'skin-current-typed-target-input.npz';np.savez_compressed(out,**v)
    report={'status':'PASS_MATCHING_TYPED_SKIN_TARGET_PREPARATION_ONLY','physical_admission':False,'native_steps':0,
      'source_sha256':sha(Path(__file__)),'input':str(a.input),'input_sha256':sha(a.input),
      'anchor':str(a.anchor),'anchor_sha256':sha(a.anchor),'target_O':str(a.O),'target_O_sha256':sha(a.O),
      'target_I':str(a.I),'target_I_sha256':sha(a.I),'orientation_signs':signs,
      'points_and_cells_unchanged':True,'rim_walls_unchanged':len(walls),'closed_target':True,
      'missing_target_faces':len(missing),'extra_current_faces':len(extra),'input_cells':len(t),
      'native_quality_failures':int(np.count_nonzero(~core.m.m.passes(det,qual))),
      'output':str(out),'output_sha256':sha(out),'controls':controls(),'elapsed_s':time.monotonic()-tic}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--controls-only',action='store_true')
    for n in ('input','anchor','O','I'):ap.add_argument('--'+n,type=Path);ap.add_argument('--'+n+'-sha256')
    ap.add_argument('--output',type=Path)
    a=ap.parse_args()
    if a.controls_only:print(json.dumps(controls()))
    else:run(a)

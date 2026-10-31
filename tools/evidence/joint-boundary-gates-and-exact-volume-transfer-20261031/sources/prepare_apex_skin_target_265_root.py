#!/usr/bin/env python3
"""Prepare the matching connected Skin volume target from typed O/I surfaces.
Preserves every physical point and cell; no solver, force or mass admission.
"""
from pathlib import Path
import numpy as np,json,hashlib,argparse,time,importlib.util,sys
H=Path(__file__).resolve().parent;S=H/'fixed-boundary-quality-topology-196'
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
M=module('cavity247for265',H/'remesh_current_surface_cavities_247_root.py')
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def cyc(f):
 a,b,c=map(int,f);return min((a,b,c),(b,c,a),(c,a,b))
def key(p):return np.asarray(p,np.float32).tobytes()
def run(a):
 tic=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
 source=H/'connected-star-union-fans-250-root/skin-current-batch004/kernel-fan-volume-proposal.npz'
 assert sha(source)=='00e63d02940dfe631cdbc8abc383cf28f58c437a9839f0e022abd4e52b2ddc7c'
 newpaths={'O':H/'native-apex-surface-floor-259-root/OI-batch001/typed-260-002/candidate-O-typed-current.npz','I':H/'native-apex-connected-262-root/I-batch001/typed-264-001/candidate-I-typed-current.npz'}
 newsha={'O':'6ea304253da218a4811fe0710ca79ba3b51242def2f62106d5ebb500317bac0d','I':'645999a1226b15eb3c3edf1222d246e0df4155dbf2f517f4125146de48be66d6'}
 oldpaths={r:S/'current-typed-adapter-024-run001'/f'candidate-{r}-typed-current-024.npz' for r in ('O','I')}
 v=load(source);p=np.asarray(v['points_world_m'],np.float32);t=np.asarray(v['tetrahedra'],np.int64);bb=M.m.boundary(t);rows={key(x):i for i,x in enumerate(p)};assert len(rows)==len(p)
 current={tuple(sorted(map(int,f))):tuple(map(int,f)) for f in bb};assert len(current)==len(bb)
 oldkeys=set();desired=[];inputpins={};signs={}
 for role in ('O','I'):
  assert sha(newpaths[role])==newsha[role];old=load(oldpaths[role]);new=load(newpaths[role])
  assert np.array_equal(old['points_world_m'],new['points_world_m']) and np.array_equal(old['points_world_m_f32'],new['points_world_m_f32'])
  mp=np.asarray([rows[key(x)] for x in old['points_world_m_f32']],np.int64);ff=mp[np.asarray(old['faces'],np.int64)]
  k=[tuple(sorted(map(int,f))) for f in ff];assert not oldkeys.intersection(k)
  assert all(x in current for x in k), 'old typed role differs from physical boundary'
  matches=[cyc(f)==cyc(current[k[i]]) for i,f in enumerate(ff)]
  assert all(matches) or not any(matches), 'role has mixed boundary orientation'
  same=all(matches);signs[role]=1 if same else -1
  newf=mp[np.asarray(new['faces'],np.int64)]
  if not same:newf=newf[:,[0,2,1]]
  desired.append(newf);oldkeys.update(k)
  inputpins[role]={'old_path':str(oldpaths[role]),'old_sha256':sha(oldpaths[role]),'new_path':str(newpaths[role]),'new_sha256':sha(newpaths[role]),'point_count':len(mp),'old_faces':len(ff),'new_faces':len(newf),'orientation_sign':signs[role]}
 walls=np.asarray([f for f in bb if tuple(sorted(map(int,f))) not in oldkeys],np.int64).reshape(-1,3)
 assert len(walls)==90;desired=np.vstack([*desired,walls]);assert M.closed(desired)==(True,0)
 dk=M.sortedkeys(desired);ak=M.sortedkeys(bb);missing=np.asarray([f for f in desired if tuple(sorted(map(int,f))) not in ak],np.int64).reshape(-1,3);extra=np.asarray([f for f in bb if tuple(sorted(map(int,f))) not in dk],np.int64).reshape(-1,3)
 assert len(dk)==len(desired);dd,qq=M.m.m.measurements(p,t);assert np.all(dd>0)
 v.update(points_world_m=p,tetrahedra=t,boundary_faces=bb,combinatorial_boundary_faces=bb,desired_boundary_faces=desired,desired_missing_faces=missing,current_extra_boundary_faces=extra)
 out=a.output/'skin-native-apex-current-target-input.npz';np.savez_compressed(out,**v)
 report={'status':'PASS_MATCHING_TYPED_SKIN_TARGET_PREPARATION_ONLY','physical_admission':False,'native_steps':0,'source_sha256':sha(Path(__file__)),'input':str(source),'input_sha256':sha(source),'typed_roles':inputpins,'points':len(p),'cells':len(t),'original_points_and_cells_unchanged':True,'unchanged_rim_wall_faces':len(walls),'current_boundary_faces':len(bb),'desired_boundary_faces':len(desired),'desired_missing_faces':len(missing),'current_extra_faces':len(extra),'remaining_nonpositive_cells':int(sum(dd<=0)),'remaining_native_quality_failures':int(sum(~M.m.m.passes(dd,qq))),'output':str(out),'output_sha256':sha(out),'elapsed_s':time.monotonic()-tic}
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);run(ap.parse_args())

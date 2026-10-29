#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');B=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union'
D=B/'dnew-vs-i296-inventory-001/typed-D-62000100/run-002/D-closed-csg-only.npz';SOURCE=B/'inputs/D-temporary-closed.npz';I=B/'minimum-thickness-minkowski-328/attempt-004-prebuffer-simplify/candidate-I-after-fat-f64.npz';OUT=P/'deep-core-void-classification-362'
sys.path.insert(0,str(P.parent/'authoring-deps'));import manifold3d as mf
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def mesh(p):
 with np.load(p,allow_pickle=False) as z:
  x=np.array(z['points_world_m_f64'] if 'points_world_m_f64' in z.files else z['points_world_m'],np.float64,order='C');f=np.array(z['faces_f64'] if 'faces_f64' in z.files else z['faces'],np.uint64,order='C')
 m=mf.Manifold(mf.Mesh64(vert_properties=x,tri_verts=f))
 if m.status()!=mf.Error.NoError:raise ValueError(f'invalid import {p}: {m.status()}')
 return m
def positive_copy(m):
 z=m.to_mesh64();return mf.Manifold(mf.Mesh64(vert_properties=np.array(z.vert_properties[:,:3],np.float64,order='C'),tri_verts=np.array(z.tri_verts[:,::-1],np.uint64,order='C')))
def main():
 t=time.monotonic();OUT.mkdir(exist_ok=False);d=mesh(D);s=mesh(SOURCE);i=mesh(I);cs=d.decompose();n=[positive_copy(c) for c in cs if c.volume()<0]
 for c in n:
  if c.status()!=mf.Error.NoError or c.volume()<=0:raise ValueError('negative-shell reorientation failed')
 cavities=mf.Manifold.batch_boolean(n,mf.OpType.Add)
 if cavities.status()!=mf.Error.NoError:raise ValueError('cavity union failed')
 overlap=mf.Manifold.batch_boolean([cavities,s],mf.OpType.Intersect);outsideI=mf.Manifold.batch_boolean([cavities,i],mf.OpType.Subtract)
 if overlap.status()!=mf.Error.NoError or outsideI.status()!=mf.Error.NoError:raise ValueError('cavity domain Boolean failed')
 r={'status':'READ_ONLY_CONSTRUCTION_VOID_DOMAIN_DIAGNOSTIC','inputs':{str(p):sha(p) for p in [D,SOURCE,I]},'D_components':len(cs),'D_negative_shell_count':len(n),'negative_shell_volume_sum_m3':sum(float(c.volume()) for c in n),'negative_shell_union_volume_m3':float(cavities.volume()),'original_D_components':len(s.decompose()),'original_D_genus':int(s.genus()),'original_D_volume_m3':float(s.volume()),'void_overlap_original_D_m3':float(overlap.volume()),'void_outside_I328_m3':float(outsideI.volume()),'maximum_cavity_face_count':max(len(c.to_mesh64().tri_verts) for c in n),'physical_admission':False,'geometry_changed':False,'elapsed_s':time.monotonic()-t,'interpretation':'Read-only Manifold F64 diagnostic. A construction void outside original D has no original D material interior. This does not by itself establish anatomy-specific appropriateness of filling it or provide exact-rational certification.'}
 (OUT/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()

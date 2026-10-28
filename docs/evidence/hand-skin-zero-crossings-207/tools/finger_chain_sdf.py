from pathlib import Path
import sys,numpy as np,json,time
from scipy.ndimage import distance_transform_edt,binary_fill_holes
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'authoring-deps'));import manifold3d as mf,igl
from cycle_fill_resume import close_for_boolean,from_mesh,export,outer
r=R/'envelope-027-finger-chain-sdf';r.mkdir(exist_ok=False);z=np.load(R/'envelope-014-connected-cycles/candidate.raw.npz');P=z['points_world_m_f64'];F=z['source_O_faces'];V,C,_,_=close_for_boolean(P,F[:,[0,2,1]]);body=from_mesh(V,C,np.arange(len(C)));sid=int(np.asarray(body.to_mesh64().run_original_id)[0]);base=np.load(R/'envelope-006-local-lattice/candidate.npz')['points_world_m'];Q=np.concatenate([np.load(R/('envelope-005-all18-stable/offset-'+key+'.npz'))['points_world_m_f64'] for key in ['59-95','60-96','61-97']]);lo=Q.min(axis=0)-.010;hi=Q.max(axis=0)+.010;clip=mf.Manifold.cube(hi-lo).translate(lo);patch=body^clip;mesh=patch.to_mesh64();PV=np.array(mesh.vert_properties)[:,:3];PF=np.array(mesh.tri_verts,dtype=np.int64);t=time.monotonic();h=.0003;glo=lo-.003;ghi=hi+.003;shape=np.ceil((ghi-glo)/h).astype(int)+1;nx,ny,nz=map(int,shape);zz,yy,xx=np.meshgrid(np.arange(nz),np.arange(ny),np.arange(nx),indexing='ij');GV=np.column_stack([xx.ravel(),yy.ravel(),zz.ravel()])*h+glo;del xx,yy,zz
print(json.dumps({'stage':'grid','points':len(GV),'shape':shape.tolist(),'h_mm':h*1000,'faces':len(PF)}),flush=True)
S,_,_,_=igl.signed_distance(GV,PV,PF,igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL);inside=(S<0).reshape(nz,ny,nx);del S
print(json.dumps({'stage':'sdf','elapsed':time.monotonic()-t}),flush=True)
for radius in [.0006,.0012,.0024]:
 dil=distance_transform_edt(~inside,sampling=h)<=radius;closed=distance_transform_edt(dil,sampling=h)>radius;closed=binary_fill_holes(closed);field=distance_transform_edt(~closed,sampling=h)-distance_transform_edt(closed,sampling=h)-.00015
 MV,MF,_=igl.marching_cubes(field.ravel(),GV,nx,ny,nz,0.);del field;voxel=from_mesh(MV,MF);innerlo=Q.min(axis=0)-.006;innerhi=Q.max(axis=0)+.006;inner=mf.Manifold.cube(innerhi-innerlo).translate(innerlo);addition=voxel^inner;parts=(body+addition).decompose();trial=max(parts,key=lambda x:x.volume());report={'radius_mm':radius*1000,'h_mm':h*1000,'expansion_mm':.15,'before_genus':body.genus(),'after_genus':trial.genus(),'voxel_genus':voxel.genus(),'added_volume_mm3':(trial.volume()-body.volume())*1e9,'elapsed_s':time.monotonic()-t};print(json.dumps(report),flush=True);(r/'report.json').write_text(json.dumps(report,indent=2))
 if trial.genus()==0:
  body=trial;np.savez_compressed(r/'addition.npz',points_world_m_f64=np.asarray(addition.to_mesh64().vert_properties)[:,:3],source_O_faces=np.asarray(addition.to_mesh64().tri_verts));break
body,voids=outer(body.simplify(1e-6));report['emission']=export(body,r/'candidate.npz',sid,len(F),base);report['final_genus']=body.genus();(r/'report.json').write_text(json.dumps(report,indent=2));print('done',flush=True)

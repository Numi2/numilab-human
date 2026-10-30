"""Geometry-only TetGen feasibility probe; never physical source admission."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser()
 for name in ('geometry-report','rim-dir','cores-dir','out'):ap.add_argument('--'+name,type=Path,required=True)
 a=ap.parse_args();g=json.loads(a.geometry_report.read_text())
 if g['status']!='PASS_FIXED_COORDINATE_GEOMETRY_DELTA':raise ValueError('independent geometry missing')
 for key in ('changed_vs_complete_current_surfaces','changed_vs_all185_both_domains'):
  if g[key]['count']!=0:raise ValueError('geometry failed')
 ownerpath=H/'native-domain-217-source-004/build_native_domain_217.py'
 if sha(ownerpath)!='46799a2d63fd20bc8a191f319f1f56244610a5e719983a7ce44cc5222659b6bd':raise ValueError('source004 changed')
 sys.path[:0]=[str(ownerpath.parent),str(P.parent/'authoring-deps')]
 spec=importlib.util.spec_from_file_location('feasibility_source004',ownerpath);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 inputs=json.loads((H/'current-D-proof-015/plan-source004-020/manifest.json').read_text())['inputs']
 geometry={}
 for role in ('O','I','D'):
  item=g['roles'][role];path=Path(item['proposal_path'])
  if sha(path)!=item['proposal_sha256']:raise ValueError('geometry changed')
  with np.load(path,allow_pickle=False)as z:geometry[role]={'points':np.asarray(z['points_world_m'],np.float32),'faces':np.asarray(z['faces'],np.int64)}
 rim=json.loads((a.rim_dir/'receipt.json').read_text())
 if rim['status']!='PASS_COLLAR_AND_PROTECTED149_REINDEX_SOURCE_ONLY':raise ValueError('rim failed')
 if rim['geometry_report']['sha256']!=sha(a.geometry_report):raise ValueError('wrong rim geometry')
 collarpath=Path(rim['outputs']['collar']['path'])
 if sha(collarpath)!=rim['outputs']['collar']['sha256']:raise ValueError('collar changed')
 with np.load(collarpath,allow_pickle=False)as z:collar={k:z[k]for k in z.files}
 O,I,D=(geometry[k]for k in ('O','I','D'));no,ni,nd=(len(x['points'])for x in (O,I,D))
 sw,fw,_=m._map_collar_faces(O,I,D,collar)
 of,_=m._orient_surface_to_wall(O['faces'].copy(),sw,'O/skin-wall')
 isk,_=m._orient_surface_to_wall(I['faces']+no,sw,'I/skin-wall')
 ifa,_=m._orient_surface_to_wall(I['faces'].copy(),fw,'I/fat-wall')
 df,_=m._orient_surface_to_wall(D['faces']+ni,fw,'D/fat-wall')
 skin={'points_world_m':np.vstack((O['points'],I['points'])),'faces':np.vstack((of,isk,sw)),
       'face_marker':np.concatenate((np.full(len(of),m.MARKER['skin_outer_O']),np.full(len(isk),m.MARKER['skin_inner_I']),np.full(len(sw),m.MARKER['skin_wall_OI']))).astype(np.int32),
       'registered_O_shell_node_rows':np.arange(no),'registered_I_shell_node_rows':np.arange(no,no+ni)}
 fat={'points_world_m':np.vstack((I['points'],D['points'])),'faces':np.vstack((ifa,df,fw)),
      'face_marker':np.concatenate((np.full(len(ifa),m.MARKER['fat_outer_I']),np.full(len(df),m.MARKER['fat_inner_D']),np.full(len(fw),m.MARKER['fat_wall_ID']))).astype(np.int32),
      'registered_I_shell_node_rows':np.arange(ni),'registered_D_shell_node_rows':np.arange(ni,ni+nd)}
 for label,shell in (('skin',skin),('fat',fat)):
  shell['faces'],shell['shell_signed_volume_m3'],_=m._closed_shell_orientation(shell['points_world_m'],shell['faces'],label)
 corepaths={}
 for role in ('I','D'):
  r=json.loads((a.cores_dir/role/'report.json').read_text());path=Path(r['core_path'])
  if r['status']!='PASS_CONSTRUCTION_CORE_AND_OWNER_HOLE_EXTRACTION_ONLY' or sha(path)!=r['core_sha256']:raise ValueError('core failed')
  corepaths[role]=path
 for key in ('parent_nodes','parent_cells','tetgen_binary'):
  if sha(inputs[key]['path'])!=inputs[key]['sha256']:raise ValueError('immutable source changed '+key)
 seedviews=H/'fixed-boundary-handoff-036/batch035-plan040/manifest.217-parent-views'
 centroids=m._parent_centroids_by_material(seedviews/'parent-nodes.csv.gz',seedviews/'parent-cells.csv.gz')
 a.out.mkdir(exist_ok=False)
 report={'status':'GEOMETRY_ONLY_PROBE_RUNNING','physical_admission':False,'runtime':0,'accepted_steps':0,
         'geometry_report':{'path':str(a.geometry_report),'sha256':sha(a.geometry_report)},'owner':{'path':str(ownerpath),'sha256':sha(ownerpath)},'seed_views':{key:{'path':str(seedviews/name),'sha256':sha(seedviews/name)}for key,name in [('nodes','parent-nodes.csv.gz'),('cells','parent-cells.csv.gz')]},'tetgen':[]}
 for label,shell,role,mat in (('skin',skin,'I',15),('fat',fat,'D',16)):
  regions,seedreceipt=m._region_points(shell,mat,centroids);holes=m._core_holes([corepaths[role]])
  np.savez_compressed(a.out/(label+'-shell.npz'),**shell)
  poly=a.out/(label+'.poly');m.write_poly(poly,shell['points_world_m'],shell['faces'],shell['face_marker'],holes,[(p,mat)for p in regions])
  result=m.run_tetgen(poly,Path(inputs['tetgen_binary']['path']),a.out,label,mat)
  result['poly_sha256']=sha(poly);result['label']=label;report['tetgen'].append(result)
  (a.out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
  print(json.dumps({'label':label,'status':result.get('status'),'wall_s':result.get('wall_s')}),flush=True)
 report['status']='GEOMETRY_ONLY_FEASIBILITY_PROBE_COMPLETE_NOT_ADMISSION'
 (a.out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':report['status'],'results':[x.get('status')for x in report['tetgen']]}),flush=True)
if __name__=='__main__':main()

"""Bounded FHL27/28 native endpoint check using existing exact audit owners."""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, io, json, os, struct, sys, time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
 os.environ[name]='1'
import numpy as np
R=Path('/Users/n/numi-human-retained-delivery-20261009')
OLD=R/'passive-biceps-micro-overlap-1225/native-biceps-10s-attempt001/native-run'
HIST=Path('/Users/n/numi-human-common-skin-multipose-001/src')
AUDIT=R/'passive-muscle-self-audit-1213/audit_muscle_self_1213.py'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def need(ok,msg):
 if not ok:raise RuntimeError(msg)
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--run',type=Path,required=True)
 parser.add_argument('--candidate',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--payload-sha256',required=True)
 args=parser.parse_args()
 out=args.output.resolve();need(not out.exists(),'refuse output overwrite');out.mkdir()
 start=time.monotonic();run=args.run.resolve();cand=args.candidate.resolve()
 tissue=cand/'bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
 manifest=tissue.with_suffix('.manifest.json'); receipt=cand/'resting-anatomy-receipt.json'
 need(sha(tissue)==args.payload_sha256,'candidate payload pin')
 sys.path.insert(0,str(HIST))
 from numilab_human import cardiac_cavity_intersections as ci, common_atlas_skin_clearance as clear
 need(sha(ci.__file__)=='11f042cc4809a27735c02b66a8fb07ebc92ee66597d7e1814a75231e2cf2d2bb','predicate source')
 need(sha(clear.__file__)=='fbfb5dfbb9b3f7e5fa297a61c793c66cb13fe884d5cebc3a7d1a6b9908961ac1','pack loader source')
 spec=importlib.util.spec_from_file_location('retained_self_audit1213',AUDIT)
 audit=importlib.util.module_from_spec(spec);sys.modules[spec.name]=audit;spec.loader.exec_module(audit)
 paths=[Path(__file__),AUDIT,tissue,manifest,receipt,cand/'report.json',run/'run-metadata.json',run/'invocation.json',run.parent/'execution.json',OLD/'run-metadata.json',OLD/'invocation.json',OLD.parent/'execution.json']
 for base in (run,OLD):
  paths += [base/n for n in ('resting-coupled.csv','resting-com-momentum-diagnostic.csv','resting-com-support-impulses.csv') if (base/n).exists()]
  paths += [base/'accepted-geometry'/f'step-{s}{suffix}' for s in (0,5000) for suffix in ('.mrvpack','.receipt.json')]
 paths += [Path(m.__file__) for k,m in tuple(sys.modules.items()) if k.startswith('numilab_human') and getattr(m,'__file__',None)]
 pins={str(p):sha(p) for p in paths}
 execution=json.loads((run.parent/'execution.json').read_text())
 need(execution['returncode']==0 and not execution.get('changed_inputs'),'native execution closed unsuccessfully or changed inputs')
 meta=json.loads((run/'run-metadata.json').read_text());inv=json.loads((run/'invocation.json').read_text())
 need(meta['exit_code']==0 and meta['loaded_metal_runtime']['verified'] and not meta.get('source_files_changed_during_run'),'native runtime or source verification')
 argv=meta['argv'];assets=meta['asset_sha256']
 need(argv==inv['argv'] and assets==inv['asset_sha256'],'metadata/invocation differ')
 for flag,path in [('--soft-tissue-payload',tissue),('--resting-anatomy-receipt',receipt)]:
  need(flag in argv and argv[argv.index(flag)+1]==str(path) and assets[str(path)]==pins[str(path)],'runtime did not load '+flag)
 need(argv[argv.index('--muscle-step-count')+1]=='5000','not the bounded5000step run')
 md=json.loads(manifest.read_text()); rd=json.loads(receipt.read_text())
 need(md['payload']['sha256']==pins[str(tissue)] and rd['provenance']['native_muscle_surfaces']['sha256']==pins[str(tissue)],'asset receipts')
 need(md['source']['fhl_source_seam_correction']['changed_stable_ids']==[27,28],'FHL composition identity')
 raw=tissue.read_bytes(); magic,abi,nr,nb,nv,ni,fp,src=struct.unpack_from('<8s6I32s',raw)
 need(magic==b'NHTISS4\0' and abi==5 and nr==150 and len(raw)==64+32*nr+36*nb+56*nv+4*ni,'native wire layout')
 recs=np.frombuffer(raw,dtype='<u4',offset=64,count=nr*8).reshape(-1,8)
 vo=64+32*nr+36*nb;ix=np.frombuffer(raw,dtype='<u4',offset=vo+56*nv,count=ni)
 positions=np.ndarray((nv,3),dtype='<f4',buffer=raw,offset=vo,strides=(56,4))
 surfaces=[]
 for sid in (27,28):
  rec=next(r for r in recs if r[6]==sid);fb,bc,fv,vc,fi,ic,_,layer=map(int,rec)
  need(layer==1 and ic%3==0,'muscle identity')
  faces=ix[fi:fi+ic].reshape(-1,3).astype(np.int64)-fv;used=np.unique(faces)
  compact=np.searchsorted(used,faces)
  result=audit.audit_space(ci,positions[fv+used].copy(),compact,used,vc,'authored_nhtiss_source')
  surfaces.append({'stable_id':sid,'faces':faces,'used':used,'compact':compact,'vc':vc,'spaces':[result]})
 need(set(map(int,recs[:,7]))=={1,2},'native muscle/tendon layers')
 target_keys={(51005 if int(r[7])==1 else 51006,int(r[6])) for r in recs}
 endpoint_geometry=[]
 for step in (0,5000):
  p=run/'accepted-geometry'/f'step-{step}.mrvpack';pr=json.loads(p.with_suffix('.receipt.json').read_text())
  need(pr['accepted_step']==step and pr['physical_endpoint']=='accepted' and pr['surface_audit_endpoint']=='passed' and pr['pack_file_sha256']==pins[str(p)],'accepted endpoint receipt')
  op=OLD/'accepted-geometry'/p.name
  pts,ss,_=clear._pack_surfaces(p,target_keys)
  oldpts,oldss,_=clear._pack_surfaces(op,target_keys)
  identity=[]
  for key in sorted(set(ss)&set(oldss)):
   if key[0] not in (51005,51006) or (key[0]==51005 and key[1] in (27,28)):continue
   f=ss[key]['faces'];g=oldss[key]['faces']
   same=f.shape==g.shape and np.array_equal(pts[f],oldpts[g])
   identity.append({'surface_key':list(key),'oriented_triangle_coordinates_f32_exact':same})
  need(len(identity)==148,'non-FHL native surface inventory differs')
  endpoint_geometry.append({'step':step,'unchanged_non_fhl_surfaces':identity,'all148_exact':all(r['oriented_triangle_coordinates_f32_exact'] for r in identity)})
  for item in surfaces:
   sid=item['stable_id'];f=ss[(51005,sid)]['faces'];d=f-item['faces']
   need(f.shape==item['faces'].shape and d.size and np.all(d==d.flat[0]),'native FHL face correspondence')
   xyz=pts[int(d.flat[0])+item['used']].copy()
   result=audit.audit_space(ci,xyz,item['compact'],item['used'],item['vc'],f'accepted_world_step_{step}')
   result['accepted_step']=step;result['accepted_time_s']=pr['accepted_time_s']
   item['spaces'].append(result)
 physiology=[]
 for name in ('resting-coupled.csv','resting-com-momentum-diagnostic.csv','resting-com-support-impulses.csv'):
  a=run/name;b=OLD/name
  if not a.exists() or not b.exists():continue
  ar=list(csv.reader(io.StringIO(a.read_text())));br=list(csv.reader(io.StringIO(b.read_text())))
  need(len(ar)>1 and ar[0]==br[0] and all(len(r)==len(ar[0]) for r in ar+br),'trace columns/row widths differ '+name)
  mismatches=[{'row':i,'column':ar[0][j],'candidate':x,'parent':y} for i,(row,oldrow) in enumerate(zip(ar[1:],br[1:]),1) for j,(x,y) in enumerate(zip(row,oldrow)) if x!=y]
  physiology.append({'file':name,'rows':len(ar)-1,'parent_rows':len(br)-1,'columns':len(ar[0]),'raw_bytes_exact':a.read_bytes()==b.read_bytes(),'same_row_count':len(ar)==len(br),'mismatched_cells':len(mismatches),'first_mismatches':mismatches[:20]})
 serial=[{'stable_id':r['stable_id'],'spaces':r['spaces']} for r in surfaces]
 allgood=all(s['topology']['closed_oriented_manifold_candidate'] and s['self_intersection']['unallowed_pair_count']==0 and s['self_intersection']['omitted_degenerate_face_count']==0 for r in serial for s in r['spaces'])
 for p,h in pins.items():need(sha(p)==h,'verification input changed '+p)
 report={'scope':'Only source and accepted0/5000 endpoint FHL self geometry plus unchanged non-FHL muscle-surface geometry and retained10s physical trace comparison. Not full-cycle geometry, skin-target, long-horizon or clinical qualification.','inputs':pins,'inputs_unchanged':True,'native_execution':execution,'all_six_fhl_spaces_closed_zero_self_zero_degenerate':allgood,'surfaces':serial,'endpoint_geometry':endpoint_geometry,'trace_comparisons':physiology,'wall_seconds':time.monotonic()-start}
 p=out/'report.json';p.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'report':str(p),'sha256':sha(p),'all_fhl_gates':allgood,'non_fhl_geometry_exact':[r['all148_exact'] for r in endpoint_geometry],'traces':physiology,'wall_seconds':report['wall_seconds']},indent=2),flush=True)
if __name__=='__main__':main()

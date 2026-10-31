
from pathlib import Path
import hashlib,py_compile,json
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def emit(src,name,replacements):
 p=H/src;s=p.read_text()
 for a,b in replacements:
  assert s.count(a)==1,(src,a,s.count(a))
  s=s.replace(a,b)
 target=H/name;assert not target.exists();target.write_text(s);py_compile.compile(str(target),doraise=True)
 return {'path':str(target),'sha256':sha(target),'parent':str(p),'parent_sha256':sha(p)}
records=[]
records.append(emit('repair_bounded_outer_O_with_witnesses_447_root.py','repair_outer_O_quality_exchange_478_root.py',[
 ("top=load('surface209I352',H/'transfer_surface_contractions_indexed_379_root.py');m=top.m","top=load('surface209I352',H/'transfer_surface_contractions_quality_exchange_461_root.py');m=top.m"),
 ("ap.add_argument('--exclude-witness-report',nargs=2,action='append',default=[])","ap.add_argument('--exclude-witness-report',nargs=2,action='append',default=[])\nap.add_argument('--allow-construction-quality-exchange',action='store_true')"),
 ("top.try_op(v,sp,sf,op)","top.try_op(v,sp,sf,op,allow_quality_exchange=args.allow_construction_quality_exchange)"),
 ("['kept_volume_node','retired_volume_node','removed_cells','bad_before','bad_after']","['kept_volume_node','retired_volume_node','removed_cells','bad_before','bad_after','new_retained_quality_failures','new_retained_quality_failure_rows']"),
 ("'maximum_edge_length_m':args.maximum_edge_m,","'maximum_edge_length_m':args.maximum_edge_m,'construction_quality_exchange':args.allow_construction_quality_exchange,'quality_policy':'STRICT_NATIVE_BAD_COUNT_DESCENT_PER_MATERIAL; native admission unchanged' if args.allow_construction_quality_exchange else 'NO_RETAINED_GOOD_REGRESSION',"),
 ("'indexed_transfer_owner':str(H/'transfer_surface_contractions_indexed_379_root.py'),'indexed_transfer_owner_sha256':sha(H/'transfer_surface_contractions_indexed_379_root.py')","'indexed_transfer_owner':str(H/'transfer_surface_contractions_quality_exchange_461_root.py'),'indexed_transfer_owner_sha256':sha(H/'transfer_surface_contractions_quality_exchange_461_root.py')")
]))
records.append(emit('bind_and_replay_bounded_outer_O_434_root.py','bind_outer_O_quality_exchange_479_root.py',[
 ("  assert not np.any(T.m.passes(od,oq)[~degen]&~T.m.passes(nd,nq))",
 """  newfail=T.m.passes(od,oq)[~degen]&~T.m.passes(nd,nq)
  if r.get('construction_quality_exchange',False):
   assert int(np.count_nonzero(newfail))==entry['new_retained_quality_failures']
   assert rows[~degen][newfail].tolist()==entry['new_retained_quality_failure_rows']
   mats=a['material_indices'][rows]
   for material in np.unique(mats):
    assert np.count_nonzero((~T.m.passes(nd,nq))&(mats[~degen]==material))<=np.count_nonzero((~T.m.passes(od,oq))&(mats==material))
   assert entry['bad_after']<entry['bad_before']
  else:
   assert not np.any(newfail)"""),
 ("'source_mass_mapping':'NOT_ADMITTED_SOURCE330_REBIND_REQUIRED',","'source_mass_mapping':'NOT_ADMITTED_SOURCE330_REBIND_REQUIRED','construction_quality_exchange':r.get('construction_quality_exchange',False),'new_retained_quality_failures':sum(x.get('new_retained_quality_failures',0) for rr in r['roles'] for x in rr['operations']),")
]))
records.append(emit('qualify_bounded_outer_skin_generic_439_root.py','qualify_current_outer_O_480_root.py',[
 ("ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()","ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--current-context-config',type=Path,required=True);ap.add_argument('--current-context-config-sha256',required=True);args=ap.parse_args()"),
 ("""cfg=json.loads((H/'joint-D309-I399-context-398-root/config354.json').read_text())
for role in ('I','D'):cfg['baseline_candidates'][role]=cfg['candidate_surfaces'][role]
dp=H/'fat614-comp1-target-D-refinement-407-004/candidate-D-comp1-current-target-refinement-typed.npz'
assert sha(dp)=='bf0cef3ab74566428cd46f38716233745b0e2dfb21d7d50d8cfe7b2790b82c38'
for key in ('baseline_candidates','candidate_surfaces'):cfg[key]['D']={'path':str(dp),'sha256':sha(dp)}""",
 """assert sha(args.current_context_config)==args.current_context_config_sha256
cfg=json.loads(args.current_context_config.read_text())
assert cfg['candidate_surfaces']['O']['sha256']==r['source_O_sha256']
for role in ('O','I','D'):
 pin=cfg['candidate_surfaces'][role];assert sha(pin['path'])==pin['sha256']
 cfg['baseline_candidates'][role]=dict(pin)"""),
 ("'raw_report':str(root/'run001/report.json'),","'current_context_config':str(args.current_context_config),'current_context_config_sha256':sha(args.current_context_config),'raw_report':str(root/'run001/report.json'),")
]))
records.append(emit('prune_witnessed_outer_skin_441_root.py','prune_current_outer_O_481_root.py',[
 ("ap=argparse.ArgumentParser();ap.add_argument('--failed-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()","ap=argparse.ArgumentParser();ap.add_argument('--failed-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--current-context-config',type=Path,required=True);ap.add_argument('--current-context-config-sha256',required=True);args=ap.parse_args()"),
 ("H/'bind_and_replay_bounded_outer_O_434_root.py'","H/'bind_outer_O_quality_exchange_479_root.py'"),
 ("[H/'qualify_bounded_outer_skin_generic_439_root.py','--root',out]","[H/'qualify_current_outer_O_480_root.py','--root',out,'--current-context-config',args.current_context_config,'--current-context-config-sha256',args.current_context_config_sha256]")
]))
receipt={'status':'PREPARED_VERSIONED_OUTER_O_CONSTRUCTION_QUALITY_EXCHANGE','physical_admission':False,'native_steps':0,'records':records,'producer_sha256':sha(__file__),'validation':'Python compilation only; actual default-parity and exact volume/context qualification mandatory'}
(H/'outer-O-quality-exchange-477-preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))

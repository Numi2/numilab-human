from pathlib import Path
import json,hashlib,shutil,gzip,re,subprocess
r=Path(__file__).resolve().parent;repo=r.parents[1];out=repo/'Docs/media/whole-visceral-coverage-20260929';native=Path('/Users/home/numi-human-standing-20260922')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cp(p,name):
 d=out/name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
def gz(p,name):
 d=out/name;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
def dump(name,obj):(out/name).write_text(json.dumps(obj,indent=2)+'\n')
t=r/'tests.verified';assert (t/'exit.code').read_text().strip()=='0'
text=(t/'stdout').read_text();match=re.search(r'(\d+) passed',text);assert match and 'failed' not in text and 'skipped' not in text
snap=json.loads((t/'executed-source.json').read_text());assert all(sha(repo/p)==h for p,h in snap.items())
assert sha(native/'apps/numilab_human_myosim_visual_probe.mm')==sha(r/'native/probe.executed.mm')
reports=[json.loads(p.read_text()) for p in (r/'audits.verified').glob('*.json') if p.name not in ['summary.json','executed-source.json']]
assert len(reports)==11 and all(p['passed'] for p in reports)
cp(r/'payload.verified'/ 'source-organ-family-anatomy.manifest.json','payload-manifest.json')
cp(r/'source-frame-reference.json','source-frame-reference.json');cp(r/'head_bone_view_selection.json','head-bone-view-selection.json')
for p in (r/'audits.verified').glob('*.json'):cp(p,'audits/'+p.name)
for profile in (r/'final-native').iterdir():
 assert (profile/'exit.code').read_text().strip()=='0'
 for f in ['command.json','stderr','exit.code','wall.seconds']:cp(profile/f,'native/'+profile.name+'/'+f)
 gz(profile/'stdout','native/'+profile.name+'/stdout.gz')
 for p in (profile/'views').glob('*.json'):cp(p,'native/'+profile.name+'/'+p.name)
 for p in (profile/'views').glob('*.png'):cp(p,'native/'+profile.name+'/'+p.name)
for folder in ['native','compose.verified','tests.final','tests.verified','audits.final']:
 for p in (r/folder).iterdir():
  if p.is_file() and p.name not in ['probe.o','probe.o.d','probe.executed.mm','myosim-visual-probe']:
   if p.name.endswith('stdout'):gz(p,folder+'/'+p.name+'.gz')
   else:cp(p,folder+'/'+p.name)
for p in (t/'executed-source').rglob('*'):
 if p.is_file():cp(p,'executed-source/'+str(p.relative_to(t/'executed-source')))
for p in (r/'tests.final/executed-source').rglob('*'):
 if p.is_file():cp(p,'executed-source/history/tests.initial/'+str(p.relative_to(r/'tests.final/executed-source')))
gz(r/'native/probe.executed.mm','executed-source/native-probe.mm.gz')
for p in ['config/source-organ-family-baseline-members.v2.json','config/source-organ-family-template.v2.json']:
 cp(repo/p,'executed-source/'+p)
for f in ['create_config.py','compile_native.py','capture_native.py','capture_final.py','audit_native.py','audit_verified.py','source_frames.py','run_tests.py','run_tests_verified.py','create_figures.py','update_board.py','verify_pdf.py','package_board.py','collect_public.py']:
 cp(r/f,'reproduce/'+f)
cp(repo/'Docs/media/zanatomy-thorax-source-20260929/ATTRIBUTION.md','ATTRIBUTION.md')
m=json.loads((out/'payload-manifest.json').read_text())
profiles=[]
for name,pose,mask,focus,head in json.loads((r/'profiles.json').read_text()):
 d=r/'final-native'/name;views=d/'views';pack=next(views.glob('*.mrvpack'));ps=next(views.glob('*.torso-anatomy-poses.json'))
 frames={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in views.glob('*.png')}
 assert len(frames)==4
 profiles.append({'name':name,'mask':mask,'focus_body_index':focus,'pose':pose,'opened_head_inspection':head,'packet':{'path':str(pack),'bytes':pack.stat().st_size,'sha256':sha(pack)},'poses':{'path':str(ps),'sha256':sha(ps)},'frames':frames})
receipt={'schema':'numi.human.whole-visceral-source-evidence.v1','date':'2026-09-29','human_base_commit':'8bddab585759c33855f987e003fa8a5406744097','native_commit':subprocess.check_output(['git','-C',str(native),'rev-parse','HEAD'],text=True).strip(),
 'inputs':{'payload':m['payload'],'configuration_sha256':sha(repo/'config/source-organ-family-composite.v2.json'),'base_387_payload_sha256':m['base_payload_sha256'],'registration_sha256':m['registration_sha256'],'native_probe_sha256':sha(r/'native/myosim-visual-probe'),'native_executed_source_sha256':sha(r/'native/probe.executed.mm'),'unchanged_runtime_library_sha256':sha(repo/'Build/tendon-surface-binding-20260929/runtime/lib/libmetalrobo.dylib')},
 'coverage':{'source_families':46,'required_unique_source_members':571,'prior_surfaces':387,'total_surfaces':579,'added_surfaces':192,'added_types':{k:sum(x['layer']==k for x in m['surfaces']) for k in sorted({x['layer'] for x in m['surfaces']})},'shared_family_members':m['selection']['shared_family_members']},
 'validation':{'all_eleven_source_audits_passed':True,'full_composite_poses':3,'final_native_pngs':44,'maximum_added_native_source_position_error_m':max(d['maximum_added_native_source_error_m'] for d in reports),'unchanged_387_prefix_byte_identical':True,'historical_ABI1_to_ABI4_native_geometry_sections_byte_identical':True,'distinct_final_checks_passed':int(match.group(1)),'skipped_checks':0,'terminal_test_exit_code':0,'test_wall_seconds':float((t/'wall.seconds').read_text()),'executed_source_sha256':snap,
 'initial_runs':'Repeat composition payload bytes agree; manifests differ only in diagnostic relative/absolute path. Initial prefix audit failed an old two-owner exact-set assumption. Initial test run stopped after nine passes and that same failure. Executed initial sources and logs retained. Corrected full composite requires exact owner coverage; prefix proofs require every original owner. No geometry or tolerance change.'},
 'qualification':{**reports[0]['qualification'],'whole_human_anatomy':False,'intracranial_or_ocular_clearance':False,'independent_cervical_or_ocular_motion':False,'standing_controller':False},
 'source_topology':{'defective_added_members':[x['member_id'] for x in m['surfaces'] if not x['source_topology']['exact_coordinate_quotient']['closed_oriented_manifold_candidate']],'closed_oriented_candidates':177,'self_and_interdomain_intersections':'not_assessed','repair_applied':False},
 'native_profiles':profiles,'retained_build_evidence':str(r),'standing_video':'28 September native run retained; predates the new anatomy, not a standing run of this payload.','boundary':m['boundary']}
receipt['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='receipt.json'}
dump('receipt.json',receipt);assert all(sha(out/p)==v['sha256'] for p,v in receipt['artifacts'].items())
print(json.dumps({'artifacts':len(receipt['artifacts']),'tests':int(match.group(1)),'native_commit':receipt['native_commit']}))

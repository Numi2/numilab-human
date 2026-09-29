from pathlib import Path
import hashlib,json,shutil,gzip,re,subprocess
root=Path(__file__).resolve().parent;repo=root.parents[1];out=repo/'Docs/media/organ-family-coverage-20260929';out.mkdir(exist_ok=True)
native=Path('/Users/home/numi-human-standing-20260922')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p,name):
 dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
def compressed(p,name):
 dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
def dump(name,obj):(out/name).write_text(json.dumps(obj,indent=2)+'\n')
tests=root/'tests.final';assert (tests/'exit.code').read_text().strip()=='1'
test_text=(tests/'stdout').read_text();match=re.search(r'(\d+) passed',test_text);assert match and int(match.group(1))==83 and '4 failed' in test_text and 'skipped' not in test_text
verified=root/'tests.verified';assert (verified/'exit.code').read_text().strip()=='0'
verified_match=re.search(r'(\d+) passed',(verified/'stdout').read_text());assert verified_match and int(verified_match.group(1))==8
abi2=root/'abi2-checks';assert (abi2/'exit.code').read_text().strip()=='0'
abi2_match=re.search(r'(\d+) passed',(abi2/'stdout').read_text());assert abi2_match and int(abi2_match.group(1))==4
total_checks=int(match.group(1))+4+2+int(abi2_match.group(1))
native_sha=sha(native/'apps/numilab_human_myosim_visual_probe.mm');assert native_sha==sha(root/'native/probe.executed.mm')
snap=json.loads((verified/'executed-source.json').read_text());assert all(sha(repo/p)==h for p,h in snap.items())
copy(root/'payload.final/source-organ-family-anatomy.manifest.json','payload-manifest.json')
for mask,view in [(63,'front'),(1023,'oblique')]:
 copy(next((root/f'final-native/projected-neutral-mask{mask}/views').glob('*-'+view+'.png')),
      f'native-neutral-mask{mask}-{view}.png')
for p in sorted((root/'audits.verified').glob('*.json')):copy(p,'audits/'+p.name)
for profile in sorted((root/'final-native').iterdir()):
 for name in ['command.json','stderr','exit.code','wall.seconds']:copy(profile/name,'native/'+profile.name+'/'+name)
 compressed(profile/'stdout','native/'+profile.name+'/stdout.gz')
 for p in (profile/'views').glob('*.json'):copy(p,'native/'+profile.name+'/'+p.name)
for kind in ['native','compose.final','compose.verified','tests.final','tests.retry','tests.verified','abi2-checks']:
 for p in (root/kind).iterdir():
  if p.is_file() and p.suffix not in ['.o','.d'] and p.name not in ['myosim-visual-probe','probe.executed.mm']:
   if p.name.endswith('stdout'):compressed(p,kind+'/'+p.name+'.gz')
   else:copy(p,kind+'/'+p.name)
compressed(root/'native/probe.executed.mm','executed-source/native-probe.mm.gz')
for p in (root/'source-history').glob('*.py'):copy(p,'executed-source/history/'+p.name)
for p,h in snap.items():
 if p.startswith('tests/'):
  compressed(repo/p,'executed-source/'+p+'.gz')
  prior=out/'executed-source'/p
  if prior.exists():
   assert sha(prior)==h
   prior.unlink()
 else:copy(repo/p,'executed-source/'+p)
for p in ['verify_final.py','audit_profiles.py','run_tests.py','run_abi2_checks.py','retry_tests.py','freeze_history.py','create_figures.py','update_board.py','verify_pdf.py','collect_public.py']:
 copy(root/p,'reproduce/'+p)
copy(repo/'Docs/media/zanatomy-thorax-source-20260929/ATTRIBUTION.md','ATTRIBUTION.md')
reports=[json.loads(p.read_text()) for p in (root/'audits.verified').glob('*-mask*.json')];assert len(reports)==12 and all(d['passed'] for d in reports)
manifest=json.loads((out/'payload-manifest.json').read_text())
receipt={'schema':'numi.human.organ-family-source-evidence.v1','date':'2026-09-29',
 'human_base_commit':'35200bf8e60c2af577eeaad7c4faf394a01bb73c',
 'native_commit':subprocess.check_output(['git','-C',str(native),'rev-parse','HEAD'],text=True).strip(),
 'inputs':{'payload':manifest['payload'],'configuration_sha256':sha(repo/'config/source-organ-family-composite.v1.json'),
  'base_310_payload_sha256':manifest['base_payload_sha256'],'registration_sha256':manifest['registration_sha256'],
  'native_binary_sha256':sha(root/'native/myosim-visual-probe'),'native_executed_source_sha256':native_sha,
  'unchanged_runtime_library_sha256':sha(repo/'Build/tendon-surface-binding-20260929/runtime/lib/libmetalrobo.dylib')},
 'coverage':{'declared_families':18,'required_unique_source_members':378,'original_surfaces':310,'total_surfaces':387,
  'shared_family_members':manifest['selection']['shared_family_members'],
  'added_types':{kind:sum(r['layer']==kind for r in manifest['surfaces']) for kind in ['organ_component','vessel','duct','cavity_reference']}},
 'validation':{'all_native_source_audits_passed':True,'poses':3,'profiles_per_pose':4,'final_native_camera_captures':48,
  'maximum_added_native_source_position_error_m':max(d['maximum_added_native_source_error_m'] for d in reports),
  'original_310_payload_geometry_byte_identical':True,'historical_abi1_abi2_abi3_native_geometry_sections_byte_identical':True,
  'final_checks_passed':total_checks,'initial_passed_checks':83,'corrected_failed_checks':4,'new_archive_refusal_checks':2,'additional_abi2_negative_checks':4,'skipped_checks':0,'final_test_exit_code':0,
  'test_runs':[{'directory':'tests.final','passed':83,'failed':4,'exit_code':1},
               {'directory':'tests.retry','passed':5,'failed':1,'exit_code':1},
               {'directory':'tests.verified','passed':8,'failed':0,'exit_code':0},
               {'directory':'abi2-checks','passed':4,'failed':0,'exit_code':0}],
  'failed_harness_cases':'Already-baseline FJ1932 expected among additions; unchanged owner values used as corruptions; obsolete ABI1 CLI flag; first retry used directory without retained packet. Corrected ABI1 input is current-neutral.v2. No source/geometry tolerances widened.',
  'unique_check_count_excludes_duplicate_abi2_and_abi3_positive_rechecks':True,
  'test_wall_seconds':float((tests/'wall.seconds').read_text()),'executed_source_sha256':snap},
 'qualification':{**reports[0]['qualification'],'whole_human_anatomy':False,'standing_controller':False},
 'source_topology':{'added_defective_members':[r['member_id'] for r in manifest['surfaces'] if not r['source_topology']['exact_coordinate_quotient']['closed_oriented_manifold_candidate']],
  'added_closed_oriented_candidates':74,'self_intersections':'not_checked_by_this_increment','geometry_repair_applied':False,
  'existing_cardiac_RA_RV_intersection_evidence':'Docs/media/cardiac-cavities-20260912/independent/intersections.json',
  'existing_cardiac_intersection_evidence_sha256':sha(repo/'Docs/media/cardiac-cavities-20260912/independent/intersections.json'),
  'existing_RA_RV_intersecting_triangle_pairs':42},
 'retained_build_evidence':str(root),'preliminary_evidence':'Initial shared-helper payload/captures superseded by independent compiler/oracle; all preliminary evidence retained separately.',
 'standing_video':'28 September run retained; predates later anatomy repairs, not a new 387-surface standing run.',
 'boundary':manifest['boundary']}
receipt['artifacts']={str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='receipt.json'}
dump('receipt.json',receipt)
assert all(sha(out/p)==d['sha256'] for p,d in receipt['artifacts'].items())
print(json.dumps({'public_artifacts':len(receipt['artifacts']),'tests_passed':total_checks,'native_commit':receipt['native_commit']}))

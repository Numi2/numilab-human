from pathlib import Path
import json,hashlib,shutil,subprocess,zipfile
repo=Path('/Users/home/numilab-human');base=Path('/Users/home/NumiSuiteReports');out=base/'board-20260929-whole-visceral';public=repo/'Docs/media/whole-visceral-coverage-20260929'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((public/'receipt.json').read_text());assert receipt['validation']['terminal_test_exit_code']==0
native=Path('/Users/home/numi-human-standing-20260922')
human_revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
native_revision=subprocess.check_output(['git','-C',str(native),'rev-parse','HEAD'],text=True).strip()
assert native_revision==receipt['native_commit']
evidence=out/'evidence/whole-visceral-coverage-20260929';shutil.copytree(public,evidence)
s=(repo/'Docs/WHOLE_VISCERAL_SOURCE_GEOMETRY_20260929.md').read_text().replace('media/whole-visceral-coverage-20260929/','')
(evidence/'WHOLE_VISCERAL_SOURCE_GEOMETRY_20260929.md').write_text(s)
video=out/'Numi-Human-10-second-standing-simulation.mp4';assert sha(video)=='420e17b43a797313fe77e8950d6b9fced66f9a2fb046a278e6f39219a2b66687'
probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_frames,duration','-show_entries','format=duration','-of','json',str(video)],capture_output=True,text=True,check=True)
v=json.loads(probe.stdout);stream=v['streams'][0];assert stream['duration']=='10.000000' and stream['nb_frames']=='300'
parity=json.loads((out/'pdf-page-parity.json').read_text());assert parity['all_original_pages_pixel_identical']
manifest={'schema':'numi.suite.executive-board-visual-pack.v6','date':'2026-09-29','slides':30,'human_revision':human_revision,'native_revision':native_revision,'original_28_slide_parts_and_media_byte_identical':True,'original_28_pdf_pages_pixel_identical':True,'anatomy_receipt_sha256':sha(public/'receipt.json'),'latest_anatomy_surfaces':579,'declared_source_families':46,'source_geometry_qualified':True,'whole_human_clinical_anatomy_qualified':False,'standing_video':{'sha256':sha(video),'probe':v,'date':'2026-09-28','predates_new_anatomy':True},'files':{str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'manifest.json'}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
zip_path=base/'Numi-Suite-Board-Visuals-whole-visceral-update-20260929.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(out.rglob('*')):
  if p.is_file():z.write(p,str(Path(out.name)/p.relative_to(out)))
with zipfile.ZipFile(zip_path) as z:
 assert z.testzip() is None
 for name,d in manifest['files'].items():assert hashlib.sha256(z.read(str(Path(out.name)/name))).hexdigest()==d['sha256']
report={'zip':str(zip_path),'bytes':zip_path.stat().st_size,'sha256':sha(zip_path),'files':len(manifest['files'])+1,'human_revision':human_revision,'native_revision':native_revision,'slide_count':30}
(repo/'Build/whole-visceral-coverage-20260929/board-package-result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))

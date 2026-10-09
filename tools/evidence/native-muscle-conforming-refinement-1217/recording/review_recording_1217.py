from pathlib import Path
import csv,hashlib,json,subprocess,shutil
E=Path('/Users/n/numi-human-resting-evidence-20261005')
R=Path('/Users/n/numi-human-retained-delivery-20261009')
run=R/'muscle-conforming-refinement-1216/native-refinement-1217-attempt2/native-run'
out=R/'muscle-conforming-refinement-1216/native-refinement-1217-attempt2/recording-review-1217'
inspector=Path('/Users/n/numi-human-resting-final-source-028/matter/tools/inspect_resting_movie.swift')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(inspector)=='2d6704dd5f06bffd0b8aa0072171af805fdb3dd00dcc11c02e0b692513b21649'
assert sha(run/'run-metadata.json')=='4b394d7dc0461e98ea5aad93435c5097eff2c1b0aa3a9dc293c16fcbeaf1c844'
meta=json.loads((run/'run-metadata.json').read_text())
assert meta['exit_code']==0 and meta['source_files_changed_during_run']==[]
assert meta['environment']['NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS']=='2.5'
assert meta['environment']['NUMI_HUMAN_RESTING_INSPECTION_TOUR']=='1'
out.mkdir(exist_ok=False)
sources=[run/'native-viewer.mov',run/'resting-surface-audit.csv',run/'run-metadata.json',inspector,Path(__file__)]
before={str(p):sha(p) for p in sources}
for p in sources[:2]:
 shutil.copyfile(p,out/p.name)
 assert sha(out/p.name)==before[str(p)]
argv=['/usr/bin/swift',str(inspector),str(out),'2.5','7']
result=subprocess.run(argv,text=True,capture_output=True)
(out/'inspect-resting-movie.log').write_text(result.stdout+result.stderr)
after={str(p):sha(p) for p in sources}
assert result.returncode==0 and before==after
with (run/'resting-surface-audit.csv').open() as f: rows=list(csv.DictReader(f))
lines=result.stdout.splitlines()
summary=dict(w.split('=',1) for w in next(l for l in lines if l.startswith('frames=')).split())
assert int(summary['frames'])==len(rows)
snapshots=[dict(w.split('=',1) for w in l.split()) for l in lines if l.startswith('frame=')]
assert {s['frame'] for s in snapshots}=={'initial','middle','final','skin','muscles','skeleton','organs','lungs','heart','vessels'}
report={'schema':'numi.human.native-refinement-1217-recording-review.v1','status':'pass','registered_trial':False,
 'scope':'Closed standalone 40-second native refinement-1217 recording sample/PTS integrity and exact representative frame extraction; captures at steps 0, 9983, and 20000; no long-run, geometry, or physiological qualification.',
 'run':str(run),'source_hashes':before,'source_hashes_after':after,'unchanged':before==after,
 'inspector':{'argv':argv,'returncode':result.returncode},'movie':summary,'surface_rows':len(rows),
 'decoded_scope':'All compressed image samples read; ten selected frames decoded through AVAssetImageGenerator. No claim of decoding every image frame.',
 'snapshots':snapshots,'snapshot_hashes':{p.name:sha(p) for p in out.glob('frame-*.png')}}
(out/'recording-review.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report':str(out/'recording-review.json'),'sha256':sha(out/'recording-review.json'),'movie':summary,'snapshots':snapshots}))

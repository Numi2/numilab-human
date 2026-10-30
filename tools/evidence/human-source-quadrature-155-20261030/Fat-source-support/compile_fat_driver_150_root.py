import json,hashlib,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
argv=json.loads((D/'compile-argv.json').read_text())
headroot=D.parent/'source/apps'
sources={str(p):sha(p) for p in headroot.glob('*.hpp')}
t=time.monotonic()
with open(D/'compile.stdout.txt','wb') as o,open(D/'compile.stderr.txt','wb') as e:
 r=subprocess.run(argv,stdout=o,stderr=e)
deps=[]
for line in (D/'compile.stderr.txt').read_text().splitlines():
 if line.startswith('.'):
  path=Path(line.lstrip('. '))
  if path.is_file() and not str(path).startswith('/Library/Developer/'):
   deps.append({'path':str(path),'sha256':sha(path),'size':path.stat().st_size})
b=Path(argv[-1]);changed=[p for p,h in sources.items() if sha(Path(p))!=h]
receipt={'status':'PASS_SOURCE_ONLY_DRIVER_COMPILE' if r.returncode==0 and not changed else 'FAIL_SOURCE_ONLY_DRIVER_COMPILE','exit_code':r.returncode,'physical_admission':False,'native_steps':0,'argv':argv,'driver_sha256':sha(D/'fat_partial_mass_driver.cpp'),'binary_path':str(b),'binary_sha256':sha(b) if b.exists() else None,'dependency_trace':deps,'headers_changed_during_compile':changed,'elapsed_s':time.monotonic()-t}
(D/'compile-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:receipt[k] for k in ('status','exit_code','binary_sha256','headers_changed_during_compile','elapsed_s')},indent=2))
raise SystemExit(0 if receipt['status'].startswith('PASS') else 1)

"""Guard the already reviewed native FHL10s declaration without rewriting it."""
from pathlib import Path
import datetime, hashlib, json, os, shutil, subprocess, sys, time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def need(ok,msg):
 if not ok:raise RuntimeError(msg)
def main():
 declaration=ROOT/'run-declaration.json'
 expected={'run-declaration.json':'d30e512fa186a70b852fadbc704e1ae7fd25d65f302d37e74bf61be920b5d587',
 'preparation-report.json':'667875ed4de5c6cffc9fd37a11a1effca4231bd38b20106e60a92d60157ada4a',
 'run.py':'fd11ec624fad091ad77f8a8c95e0467ef989cd215a01fe6220fb9158692b933e'}
 for name,h in expected.items():need(sha(ROOT/name)==h,'reviewed launch file changed '+name)
 d=json.loads(declaration.read_text());argv=d['argv']
 need(d['seconds']==10 and argv[argv.index('--seconds')+1]=='10','not bounded10s')
 out=Path(argv[argv.index('--output')+1])
 need(out==ROOT/'native-run','unexpected output path')
 need(all(not p.exists() for p in [out,ROOT/'owner-stdout.log',ROOT/'execution.json',HERE/'started.json',HERE/'result.json']),'refuse replay/overwrite')
 need('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS=0,5000' in argv,'endpoint capture settings')
 prep=json.loads((ROOT/'preparation-report.json').read_text()); native=prep['owner_resolved_native_argv']
 need(native[native.index('--stand-contact-iterations')+1]=='64','contact iteration count')
 need(native[native.index('--muscle-step-count')+1]=='5000','physical horizon')
 need('contoured' not in ' '.join(s for s in argv if '=' in s and not s.startswith('NUMI_BUILD_DIR=')),'unexpected contoured environment')
 bed=Path(native[native.index('--support-contact-payload')+1])
 need(sha(bed)=='bcfece8e5da553b98694b724644234407fa4c38383b1e18d3c24d7caefa28927','flat support identity')
 processes=subprocess.check_output(['/bin/ps','-axo','pid=,comm='],text=True)
 active=[line for line in processes.splitlines() if line.strip().split()[-1].endswith('/numi-human-native') or line.strip().split()[-1]=='numi-human-native']
 need(not active,'another native GPU owner is active')
 free=shutil.disk_usage(ROOT).free
 need(free>950*1024*1024,'insufficient short-run disk reserve')
 (HERE/'exclusive-launch.lock').mkdir()
 start=time.monotonic();before=datetime.datetime.now(datetime.timezone.utc).isoformat()
 record={'start_utc':before,'launcher_sha256':sha(__file__),'reviewed_files':expected,'available_bytes_before':free,'native_processes_before':active,
 'host':subprocess.check_output(['/bin/hostname'],text=True).strip(),'system':subprocess.check_output(['/usr/sbin/sysctl','hw.model','hw.memsize'],text=True),
 'qualification_correction':'The inherited declaration qualification_boundary mentions310s; actual seconds10, native5000 steps, captures0/5000. This is a bounded native smoke, not long-horizon qualification.',
 'concurrent_work':'SingleCPU offline skin fit running. This run is not a standalone performance benchmark.'}
 (HERE/'started.json').write_text(json.dumps(record,indent=2)+'\n')
 result=subprocess.run([sys.executable,str(ROOT/'run.py')],cwd=ROOT)
 execution=json.loads((ROOT/'execution.json').read_text()) if (ROOT/'execution.json').exists() else {}
 final={'start_utc':before,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':time.monotonic()-start,'runner_returncode':result.returncode,'native_returncode':execution.get('returncode'),'execution_sha256':sha(ROOT/'execution.json') if execution else None,'changed_inputs':execution.get('changed_inputs'),'available_bytes_after':shutil.disk_usage(ROOT).free}
 (HERE/'result.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final,indent=2),flush=True)
 need(result.returncode==0 and execution.get('returncode')==0 and not execution.get('changed_inputs'),'native proof failed; retained outputs')
if __name__=='__main__':main()

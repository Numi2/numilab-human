from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def write(p,r):Path(p).write_text(json.dumps(r,indent=2)+'\n')
def run(cmd,log):
 with Path(log).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(str(cmd)+': '+Path(log).read_text()[-2000:])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--state',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--rounds',type=int,default=12);a=ap.parse_args()
 a.output.mkdir(exist_ok=False);state=a.state;progress=[]
 driver=H/'quality-partial-cavity-684-root/batch_interior_steiner_kernel_cone_quality_767_root.py';ver=H/'quality-partial-cavity-684-root/independent_promoter_771_root.py'
 assert pin(driver)['sha256']=='c9d11fbcc2403755f283a9773053dd8e91c4d81dde8f3594c3f29be208a7da75'
 assert pin(ver)['sha256']=='020524d1f30001572e125b5a99f7216f586d24b610af4dd3509b6aa2c8ead69f'
 for i in range(a.rounds):
  s=read(state);before=sum(s[r+'_current']['native_quality_failures'] for r in ['Skin','Fat'])
  if not before:break
  out=a.output/('round%03d'%(i+1));out.mkdir();proposal=out/'proposal'
  run([sys.executable,str(driver),'--state',str(state),'--output',str(proposal),'--seconds','90','--maximum-attempts','10000','--maximum-operations','1000'],out/'producer.log')
  report=read(proposal/'report.json')
  if report['status']=='NO_VALID_BOUNDED_PROGRESS':break
  if not any(r.get('status')=='PASS_INTERIOR_STEINER_KERNEL_CONE_BATCH' for r in report['role_results']):break
  verified=out/'verified-state.json'
  run([sys.executable,str(ver),'--state',str(state),'--journal',str(proposal/'report.json'),'--output',str(verified),'--receipt',str(out/'independent-replay.json')],out/'verifier.log')
  s=read(verified);counts={r:s[r+'_current']['native_quality_failures'] for r in ['Skin','Fat']};assert sum(counts.values())<before
  state=verified;row={'round':i+1,'state':pin(state),**counts};progress.append(row);write(a.output/'latest-verified.json',{'current':row,'progress':progress});print('VERIFIED_ADAPTIVE_CONE',json.dumps(row),flush=True)
 write(a.output/'completion.json',{'status':'ZERO_NATIVE_QUALITY_FAILURES' if sum(read(state)[r+'_current']['native_quality_failures'] for r in ['Skin','Fat'])==0 else 'BOUNDED_METHOD_COMPLETE_NOT_FINISHED','state':pin(state),'progress':progress,'physical_admission':False,'native_steps':0})
if __name__=='__main__':main()

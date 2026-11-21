import hashlib,json,os,resource,shutil,subprocess,sys,time
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,x):path.write_text(json.dumps(x,indent=2)+'\n')

def trial(base,volume,output,rigid,receipt):
    original=base/'launch.json'; cfg=json.loads(original.read_text()); j=json.loads(receipt.read_text())
    old=cfg['argv'][1]; assert cfg['pins'][old]==sha(old)==j['source_sha256'] and sha(rigid)==j['result_sha256']
    env=cfg['environment']
    pack=env.pop('NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE')
    assert cfg['pins'].pop(pack)==env.pop('NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256')
    cfg['argv'][1]=str(rigid); cfg['argv'][4]=str(volume/'native')
    assert '--resting-hip-capsule-reference' in cfg['argv']
    for path,x in cfg['pins'].items():assert sha(path)==x,path
    cfg['pins'][str(rigid)]=sha(rigid)
    save(output/'launch.json',cfg)
    start=time.monotonic(); resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    host={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_'))}
    with (output/'stdout.log').open('wb') as so,(output/'stderr.log').open('wb') as se:
        with subprocess.Popen(cfg['argv'],cwd=cfg['cwd'],env=host|env,stdout=so,stderr=se) as proc:
            why=None
            while proc.poll() is None:
                if shutil.disk_usage('/System/Volumes/Data').free<2**30:why='system disk reserve'
                if time.monotonic()-start>420:why='wall deadline'
                if why:
                    proc.terminate()
                    try:proc.wait(10)
                    except subprocess.TimeoutExpired:proc.kill()
                    break
                time.sleep(.2)
            code=proc.wait()
    save(output/'execution.json',dict(exit_code=code,wall_s=time.monotonic()-start,watchdog=why))
    for path in (volume/'native').glob('*.csv'):
        if path.name in {p.name for p in base.glob('*.csv')}:shutil.copyfile(path,output/path.name)
    print(code)
if __name__=='__main__':trial(*map(Path,sys.argv[1:]))

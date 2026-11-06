from pathlib import Path
import json,time,shutil
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
p=W/'rejected541-compiler-recovery-574/recovery-index.json';start=time.time()
while True:
 r=json.loads(p.read_text()) if p.exists() else []
 if len(r)>=2 and shutil.disk_usage(W).free>=16*2**30:break
 assert time.time()-start<840,'Recovery not ready within bounded preparation budget'
 time.sleep(5)
print('Two verified archive shards and16GiB free; retain12GiB live guard',flush=True)

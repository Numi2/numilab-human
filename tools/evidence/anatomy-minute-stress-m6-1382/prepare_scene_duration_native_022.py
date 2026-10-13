#!/usr/bin/env python3
"""Extend a pinned existing native scene for accepted-pose stress evidence."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(v,m):
    if not v: raise ValueError(m)
def save(p,j): p.write_text(json.dumps(j,indent=2,sort_keys=True)+'\n')
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seconds',type=float,required=True);ap.add_argument('--captures',required=True)
    ap.add_argument('--intervention',type=float,nargs=3)
    a=ap.parse_args();o=a.output.resolve();par=a.parent.resolve()
    req(not o.exists(),'preserve prior output')
    d=json.loads((par/'baseline/run-declaration.json').read_text())
    ex=json.loads((par/'baseline/execution.json').read_text())
    req(ex['returncode']==0 and not ex['changed_inputs'] and not ex['timed_out'],'parent failed')
    pins=dict(d['immutable_assets'])
    for p,h in pins.items():req(sha(p)==h,'parent drift '+p)
    steps=round(a.seconds/.002);caps=list(map(int,a.captures.split(',')))
    req(16<=a.seconds<=310 and abs(steps*.002-a.seconds)<1e-8,'bounded duration')
    req(len(caps)<=8 and len(set(caps))==len(caps) and all(0<=s<=steps and (s in (0,steps) or s%32==31) for s in caps),'bounded captures')
    src=d['argv'];mi=src.index('numilab_human.resting_run');tail=src[mi+1:].copy()
    tail[tail.index('--seconds')+1]=str(a.seconds)
    oldout=Path(tail[tail.index('--output')+1]).resolve()
    tail[tail.index('--output')+1]=str(o/'baseline/native-run')
    if a.intervention:tail+=['--drive-intervention',*map(str,a.intervention)]
    env=dict(x.split('=',1) for x in src[2:mi-2])
    env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=','.join(map(str,caps))
    env['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(o/'baseline/native-run/common-field-failure.json')
    sys.path.insert(0,env['PYTHONPATH'])
    from numilab_human import resting_run as rr
    p=argparse.ArgumentParser();rr.add_arguments(p);native,assets=rr.command(p.parse_args(tail))
    old=d['owner_cli_preview']['native_argv']
    expected=old.copy()
    expected[expected.index('--muscle-step-count')+1]=str(steps)
    expected=[x.replace(str(oldout),str(o/'baseline/native-run')) for x in expected]
    if a.intervention:expected+=['--resting-drive-intervention',*map(str,a.intervention)]
    req(native==expected,'physical invocation differs beyond duration/intervention/output')
    o.mkdir();(o/'baseline').mkdir()
    launch=o/'launch_arm.py';shutil.copy2(par/'launch_arm.py',launch)
    for p in (Path(__file__),par/'baseline/execution.json',par/'baseline/run-declaration.json',launch):pins[str(p.resolve())]=sha(p)
    pins.update(assets)
    d.update(seconds=a.seconds,accepted_steps=steps,capture_steps=caps,
        immutable_assets=pins,argv=['/usr/bin/env','-i',*[k+'='+v for k,v in sorted(env.items())],src[mi-2],'-m','numilab_human.resting_run',*tail],
        launch_adapter={'path':str(launch),'sha256':sha(launch)},
        owner_cli_preview={'native_argv':native,'asset_sha256':assets,'recorded_invocation_environment':rr.invocation_environment(env)},
        qualification='Native accepted-pose anatomy stress evidence with preserved back412 and repaired skin. Forearm crossings unresolved; no complete anatomy or final five-minute qualification.')
    dp=o/'baseline/run-declaration.json';save(dp,d)
    save(o/'launch-guard.json',{'launcher_sha256':sha(launch),'arms':{'baseline':{'run_declaration_sha256':sha(dp),'native_output':str(o/'baseline/native-run')}}})
    save(o/'preparation.json',{'parent':str(par),'seconds':a.seconds,'capture_steps':caps,'intervention':a.intervention,'same_native_binary':True,'no_physical_owner_changes':True,'native_run_pending':True})
    req(all(sha(p)==h for p,h in pins.items()),'input changed')
    print(json.dumps({'output':str(o),'pins':len(pins),'seconds':a.seconds}))
if __name__=='__main__':main()

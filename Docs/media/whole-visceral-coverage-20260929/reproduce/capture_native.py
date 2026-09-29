from pathlib import Path
import json,hashlib,subprocess,os,time,struct
root=Path(__file__).resolve().parent;repo=root.parents[1]
py=repo/'.venv-mujoco312/bin/python'
env={**os.environ,'PYTHONPATH':'src:tests:Sources/myosim/checkout','OPENBLAS_NUM_THREADS':'1','MTL_DEBUG_LAYER':'1'}
def run(cmd,dest):
 dest.mkdir(parents=True,exist_ok=True);(dest/'command.json').write_text(json.dumps({'cwd':str(repo),'argv':cmd},indent=2)+'\n')
 t=time.monotonic();r=subprocess.run(cmd,cwd=repo,env=env,capture_output=True,text=True,timeout=240)
 for f,v in [('stdout',r.stdout),('stderr',r.stderr),('exit.code',str(r.returncode)+'\n'),('wall.seconds',str(time.monotonic()-t)+'\n')]: (dest/f).write_text(v)
 print(dest.relative_to(root),r.returncode,flush=True)
 if r.returncode:print(r.stdout,r.stderr);raise SystemExit(r.returncode)
common=['--configuration',str(repo/'config/source-organ-family-composite.v2.json'),'--sources',str(repo/'Sources'),'--artifact',str(repo/'Build/myosim-fullbody'),'--registration',str(repo/'Build/knee-parity-registration-20260929/candidate.v6.registration.json'),'--base-payload',str(repo/'Build/organ-family-coverage-20260929/payload.final/source-organ-family-anatomy.nhanatomy')]
run([str(py),'-m','numilab_human.organ_family_geometry','compose',*common,'--output',str(root/'payload.verified')],root/'compose.verified')
for f in ['source-organ-family-anatomy.nhanatomy','source-organ-family-anatomy.manifest.json']:
 assert (root/'payload'/f).read_bytes()==(root/'payload.verified'/f).read_bytes()
base=json.loads((repo/'Build/lung-source-coverage-20260929/projected-neutral/command.json').read_text());base[0]=str(root/'native/myosim-visual-probe')
base[base.index('--torso-anatomy-payload')+1]=str(root/'payload.verified/source-organ-family-anatomy.nhanatomy')
bones=Path(base[3]);raw=bones.read_bytes();_,abi,count,*_=struct.unpack_from('<8s5I32s',raw)
anchor=json.loads(bones.with_suffix('.manifest.json').read_text())['source']['anchors']
head_visible=[];omitted=[]
for i in range(count):
 row=struct.unpack_from('<6I8f',raw,60+(60 if abi==3 else 56)*i)
 if row[0]!=23 or anchor[i]['member_id']=='FJ3289':head_visible.append(row[5])
 else:omitted.append({'stable_id':row[5],'member_id':anchor[i]['member_id']})
(root/'head_bone_view_selection.json').write_text(json.dumps({'omitted_cranial_source_meshes':omitted,'visible_stable_ids':head_visible,'boundary':'Skull view selection opens the head for source-region inspection. The bone payload is unchanged; omitted cranial bones are not in these inspection packets. Every anatomy surface remains in every packet.'},indent=2)+'\n')
profiles=[('raw-source-rest',None,63,20,False),('projected-neutral',(),63,20,False),('torso-flexion',((7,-.4),(8,.1),(9,.2)),63,20,False)]
for name,mask,focus,openhead in [('brain',1024,23,True),('ventricular-regions',2048,23,True),('gut-junction',4096,7,False),('ocular-regions',8192,23,True),('ocular-muscles',16384,23,True),('all-regions',32767,20,False),('gut',1,7,False),('pelvic',1,128,False)]:
 profiles.append((name,(),mask,focus,openhead))
(root/'profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
for name,pose,mask,focus,openhead in profiles:
 dest=root/'final-native'/name;cmd=list(base);cmd[4]=str(dest/'views');cmd[cmd.index('--focus-body-index')+1]=str(focus)
 if pose is None:del cmd[cmd.index('--joint-equality-payload'):]
 for q,v in pose or ():cmd+=['--pose-q',str(q),str(v)]
 cmd+=['--torso-anatomy-layer-mask',str(mask)]
 if openhead:
  for sid in head_visible:cmd+=['--visible-bone-stable-id',str(sid)]
 run(cmd,dest)

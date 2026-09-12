import hashlib,json,math,pathlib,struct,subprocess
root=pathlib.Path('/Users/n/human-cardiac-material-frame-preparation-20260912/cpp-evidence/controls')
root.mkdir(parents=True,exist_ok=False)
exe='/Users/n/human-cardiac-material-frame-preparation-20260912/cardiac_material_frame_check'
identity='01'+'00'*31
q=math.sqrt(.5)
cases=[('identity',[1.,0,0],[0.,1,0],[0.,0,0,1],1,identity,True),
 ('quarter-turn',[0.,1,0],[-1.,0,0],[0.,0,q,q],1,identity,True),
 ('raw-nonunit-nonorthogonal',[2.,0,0],[.01,3,0],[0.,0,0,1],1,identity,True),
 ('count-mismatch',[1.,0,0],[0.,1,0],[0.,0,0,1],2,identity,False),
 ('nonunit-quaternion',[1.,0,0],[0.,1,0],[0.,0,0,2],1,identity,False),
 ('nan-quaternion',[1.,0,0],[0.,1,0],[math.nan,0.,0,1],1,identity,False),
 ('noncanonical-sign',[1.,0,0],[0.,1,0],[0.,0,0,-1],1,identity,False),
 ('negative-zero',[1.,0,0],[0.,1,0],[-0.,0,0,1],1,identity,False),
 ('wrong-basis',[1.,0,0],[0.,1,0],[0.,0,q,q],1,identity,False),
 ('parallel-axes',[1.,0,0],[1.,0,0],[0.,0,0,1],1,identity,False),
 ('nan-source',[math.nan,0,0],[0.,1,0],[0.,0,0,1],1,identity,False),
 ('missing-identity',[1.,0,0],[0.,1,0],[0.,0,0,1],1,'00'*32,False),
 ('truncated-quaternion',[1.,0,0],[0.,1,0],[0.,0,0],1,identity,False)]
results=[]
for name,f,s,q,count,pin,valid in cases:
 d=root/name;d.mkdir()
 for fn,values in [('fibres.f64le',f),('sheets.f64le',s),('quaternions.f64le',q)]:
  (d/fn).write_bytes(struct.pack('<'+'d'*len(values),*values))
 argv=[exe,str(d/'fibres.f64le'),str(d/'sheets.f64le'),str(d/'quaternions.f64le'),str(count),pin]
 p=subprocess.run(argv,capture_output=True,text=True)
 (d/'stdout.log').write_text(p.stdout);(d/'stderr.log').write_text(p.stderr)
 assert (p.returncode==0)==valid,(name,p.returncode,p.stderr)
 results.append({'case':name,'expected_valid':valid,'exit_code':p.returncode,'argv':argv})
(root/'results.json').write_text(json.dumps({'status':'pass','cases':results,'accepted':3,'rejected':10,'physical_steps':0},indent=2)+'\n')
print('PASS independent C++ controls: 3 accepted, 10 invalid cases rejected')

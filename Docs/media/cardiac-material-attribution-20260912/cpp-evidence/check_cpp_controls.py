import json,pathlib,struct,subprocess
root=pathlib.Path('/Users/n/human-cardiac-material-attribution-20260912/cpp-evidence/controls')
root.mkdir(parents=True,exist_ok=False)
exe='/Users/n/human-cardiac-material-attribution-20260912/cardiac_material_attribution_check'
points=[0.,0,0,1.,0,0,0,1.,0,0,0,1.]
cases=[]
for label in [1,2,3,4,5,6,7,10,11,17,18,24]:
 cls=0 if label<=2 else 1 if label<=4 or label>=18 else 2 if label==5 else 3 if label==6 else 4 if label<=10 else 4294967295
 cases.append((f'label-{label}',points,[0,1,2,3],[label],[cls],1,True))
cases.extend([
 ('wrong-material',points,[0,1,2,3],[1],[1],1,False),
 ('unresolved-forged-as-valve',points,[0,1,2,3],[11],[4],1,False),
 ('unknown-label',points,[0,1,2,3],[25],[4294967295],1,False),
 ('reversed-tet',points,[1,0,2,3],[1],[0],1,False),
 ('degenerate-tet',points,[0,1,1,3],[1],[0],1,False),
 ('out-of-range-node',points,[0,1,2,4],[1],[0],1,False),
 ('nonfinite-node',[float('nan')]+points[1:],[0,1,2,3],[1],[0],1,False),
 ('truncated-class',points,[0,1,2,3],[1],[],1,False),
 ('count-mismatch',points,[0,1,2,3],[1],[0],2,False)])
results=[]
for name,nodes,tets,labels,classes,count,valid in cases:
 d=root/name;d.mkdir()
 for fn,kind,values in [('nodes.f64le','d',nodes),('tetrahedra.u32le','I',tets),('labels.u32le','I',labels),('classes.u32le','I',classes)]:
  (d/fn).write_bytes(struct.pack('<'+kind*len(values),*values))
 argv=[exe,str(d/'nodes.f64le'),str(d/'tetrahedra.u32le'),str(d/'labels.u32le'),str(d/'classes.u32le'),str(count)]
 proc=subprocess.run(argv,capture_output=True,text=True)
 (d/'stdout.log').write_text(proc.stdout);(d/'stderr.log').write_text(proc.stderr)
 assert (proc.returncode==0)==valid,(name,proc.returncode,proc.stderr)
 if valid and labels==[1]:
  data=json.loads(proc.stdout)
  assert data['lv_geometric_mass']['volume_exact']=={'numerator':'1','denominator':'6'}
  assert data['lv_geometric_mass']['mass_exact']=={'numerator':'175','denominator':'1'}
 results.append({'case':name,'expected_valid':valid,'returncode':proc.returncode,'argv':argv})
(root/'results.json').write_text(json.dumps({'status':'pass','cases':results,'accepted':12,'rejected':9,'physical_steps':0},indent=2)+'\n')
print('PASS independent C++ attribution: 12 accepted,9 rejected')

from pathlib import Path
import json,os,sys,tempfile
root=Path('/Users/n/human-cardiac-wall-anatomy-20260912')
sys.path[:0]=[str(root/'tools'),str(root/'src')]
from qualify_cardiac_wall_source import qualify
original=root/'asset-final';reports=[]
for case in ('changed_config','missing_buffer_record','changed_buffer','promoted_mechanics'):
 with tempfile.TemporaryDirectory(dir=root) as temp:
  temp=Path(temp);asset=temp/'asset';asset.mkdir()
  for p in original.iterdir():
   if p.name!='manifest.json':os.link(p,asset/p.name)
  m=json.loads((original/'manifest.json').read_text())
  if case=='changed_config':m['source_config']['source']['metres_per_source_length_unit']=1
  if case=='missing_buffer_record':del m['buffers']['labels.u32le']
  if case=='changed_buffer':
   p=asset/'labels.u32le';p.unlink();p.write_bytes(b'bad')
  if case=='promoted_mechanics':m['qualification']['anatomical_wall_simulation']=True
  (asset/'manifest.json').write_text(json.dumps(m))
  try:qualify(asset,root/'cardiac-wall-asset-check',temp/'evidence')
  except ValueError as error:reports.append({'case':case,'rejected':True,'error':str(error)})
  else:raise AssertionError(case+' was accepted')
print(json.dumps({'status':'pass','cases':reports,'physical_steps':0},indent=2))

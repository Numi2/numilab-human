from pathlib import Path
import argparse,hashlib,io,json,subprocess,tarfile,zipfile
import numpy as np

def sha(b): return hashlib.sha256(b).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);planp=out/'selection-plan.json';plan=json.loads(planp.read_text());meta=json.loads((out/'github-release-metadata.json').read_text());build=json.loads((out/'archive-build-receipt.json').read_text());idxlocal=json.loads((out/'archive-index.json').read_text())
 asset=meta['assets'][0];tag=meta['tagName'];name=asset['name'];expected=build['archive_sha256'];assert asset['digest']=='sha256:'+expected and asset['size']==build['archive_size_bytes'];assert tag==plan['release_target']['tag'] and name==plan['release_target']['asset_name']
 cmd=['gh','release','download',tag,'--repo','Numi2/numilab-human','--pattern',name,'--output','-'];r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=300)
 if r.returncode: raise RuntimeError('GH download failed '+r.stderr.decode('utf8','replace')[-1000:])
 tar_bytes=r.stdout;remote_sha=sha(tar_bytes)
 if len(tar_bytes)!=asset['size'] or remote_sha!=expected: raise RuntimeError('independent downloaded asset size/SHA mismatch')
 z=subprocess.run(['/opt/homebrew/bin/zstd','-dc'],input=tar_bytes,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=300)
 if z.returncode: raise RuntimeError('zstd stream failed '+z.stderr.decode('utf8','replace')[-1000:])
 tar_bytes=z.stdout;del r,z
 records=[];header_count=0;all_arrays=0;index=None;selection=None;seen=set()
 with tarfile.open(fileobj=io.BytesIO(tar_bytes),mode='r:') as tf:
  for m in tf:
   if not m.isfile() or m.name in seen: raise RuntimeError('unexpected/duplicate archive member '+m.name)
   seen.add(m.name);f=tf.extractfile(m)
   if m.name=='archive-index.json': index=json.load(f);continue
   if m.name=='selection-and-retirement.json': selection=f.read();continue
   if not m.name.startswith('blobs/'): raise RuntimeError('unexpected member '+m.name)
   digest=m.name.split('/',1)[1];blob=f.read()
   if len(blob)!=m.size or sha(blob)!=digest: raise RuntimeError('blob hash/size mismatch '+digest)
   with zipfile.ZipFile(io.BytesIO(blob),'r') as zf:
    bad=zf.testzip()
    if bad is not None: raise RuntimeError('NPZ CRC failure '+bad)
    names=[n for n in zf.namelist() if n.endswith('.npy')]
    if not names: raise RuntimeError('no NPY members '+digest)
    with np.load(io.BytesIO(blob),allow_pickle=False) as arrays:
     ar=[]
     for n in names:
      x=arrays[n]
      if x.dtype.hasobject: raise RuntimeError('object array unsupported '+n)
      raw=x.tobytes(order='C');ar.append({'name':n,'shape':list(x.shape),'dtype':x.dtype.str,'nbytes':x.nbytes,'array_sha256':sha(raw)})
      all_arrays+=1
      del raw,x
   header_count+=len(names)
   records.append({'sha256':digest,'size_bytes':len(blob),'zip_crc':'PASS','npy_arrays_loaded':len(names),'array_records':ar,'ram_exact_restore_sha256':sha(bytes(blob))})
   del blob
 if index is None or selection is None: raise RuntimeError('missing index or selection')
 if sha(selection)!=sha(planp.read_bytes()): raise RuntimeError('embedded selection differs from plan')
 if sha(json.dumps(index,sort_keys=True,separators=(',',':')).encode())!=sha(json.dumps(idxlocal,sort_keys=True,separators=(',',':')).encode()): raise RuntimeError('remote archive index differs from local')
 expected_rows={x['expected_sha256_from_source_receipt']:x for x in plan['files']}
 if set(expected_rows)!={x['sha256'] for x in records}: raise RuntimeError('restored blob set differs from plan')
 if len(records)!=plan['chunk']['file_count'] or sum(x['size_bytes'] for x in records)!=plan['chunk']['original_bytes']: raise RuntimeError('restore totals differ from plan')
 report={'schema':'numi-human.pre824-volume-archive-independent-readback.v1','status':'PASS_GH_DOWNLOAD_ARCHIVE_SHA_ALL_NPZ_CRC_AND_ALL_NPY_ARRAYS_LOADED_EXACT_RAM_RESTORE','tag':tag,'asset_id':asset['id'],'asset_node_id':asset.get('node_id'),'asset_name':name,'asset_bytes':len(tar_bytes),'asset_sha256':remote_sha,'archive_index_canonical_sha256':sha(json.dumps(index,sort_keys=True,separators=(',',':')).encode()),'embedded_plan_sha256':sha(selection),'blob_count':len(records),'original_volume_bytes':sum(x['size_bytes'] for x in records),'npz_crc_all_passed':True,'npy_header_count':header_count,'npy_arrays_loaded':all_arrays,'filesystem_restore_performed':False,'exact_ram_restore_per_file':True,'files':records,'physical_admission':False,'native_steps':0}
 p=out/'independent-verification.json';p.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':report['status'],'asset_sha256':remote_sha,'blob_count':len(records),'original_bytes':report['original_volume_bytes'],'npy_arrays_loaded':all_arrays,'report':str(p),'report_sha256':sha(p.read_bytes())},sort_keys=True))
if __name__=='__main__':main()

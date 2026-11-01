#!/usr/bin/env python3
"""Independently download and fully verify a 958 source-only GH snapshot in RAM."""
import ast, datetime, hashlib, io, json, struct, subprocess, tarfile, zipfile
from pathlib import Path
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
OUT=H/'quality-958-source-snapshot-20261101-plan'; PLAN=OUT/'selection-plan.json'; META=OUT/'github-release-metadata.json'; BUILD=OUT/'archive-build-receipt.json'
TAG='human-quality-source-snapshot-20261101-958-001'; ASSET='quality-958-source-snapshot-20261101-001.tar.zst'; REPO='Numi2/numilab-human'
def sha(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def header(fp,name):
 v=np.lib.format.read_magic(fp)
 if v==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(fp)
 elif v==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(fp)
 elif v==(3,0):
  n=struct.unpack('<I',fp.read(4))[0]
  if n>10000:raise RuntimeError('oversized NPY header')
  d=ast.literal_eval(fp.read(n).decode('utf8').strip());shape,fortran,dtype=tuple(d['shape']),bool(d['fortran_order']),np.dtype(d['descr'])
 else:raise RuntimeError('unsupported NPY version '+repr(v))
 return {'name':name,'version':list(v),'shape':list(shape),'fortran_order':bool(fortran),'dtype':dtype.str,'dtype_descr':repr(dtype.descr),'hasobject':bool(dtype.hasobject)}
def main():
 planbytes=PLAN.read_bytes();plan=json.loads(planbytes);build=json.loads(BUILD.read_text());meta=json.loads(META.read_text())
 if plan.get('snapshot',{}).get('physical_admission') is not False or plan['snapshot'].get('native_steps')!=0 or plan['snapshot'].get('fat_native_quality_failures')!=4 or plan['snapshot'].get('skin_native_quality_failures')!=5 or plan['snapshot'].get('mass_reconciliation_status')!='PENDING_VOLUME_CHANGE_RECONCILIATION':raise RuntimeError('source-only/failure/pending labels changed')
 if meta.get('tagName')!=TAG:raise RuntimeError('GitHub tag mismatch')
 matches=[a for a in meta.get('assets',[]) if a.get('name')==ASSET]
 if len(matches)!=1:raise RuntimeError('expected exactly one GH asset')
 asset=matches[0];expected_sha=build['archive_sha256'];expected_size=build['archive_size_bytes']
 if asset.get('digest')!='sha256:'+expected_sha or asset.get('size')!=expected_size:raise RuntimeError('GH metadata digest/size mismatch')
 cmd=['gh','release','download',TAG,'--repo',REPO,'--pattern',ASSET,'--output','-']
 remote=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=300)
 if remote.returncode:raise RuntimeError('independent GH download failed: '+remote.stderr.decode('utf8','replace')[-1200:])
 archive=remote.stdout
 if len(archive)!=expected_size or sha(archive)!=expected_sha:raise RuntimeError('independent GH archive SHA/size mismatch')
 dec=subprocess.run(['/opt/homebrew/bin/zstd','-dc'],input=archive,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=300)
 if dec.returncode:raise RuntimeError('zstd decode failed: '+dec.stderr.decode('utf8','replace')[-1200:])
 tar_bytes=dec.stdout; seen=set(); index=None; embedded_plan=None; provenance=None; tool_blobs={}; blobs={}
 with tarfile.open(fileobj=io.BytesIO(tar_bytes),mode='r:') as tf:
  for m in tf:
   if not m.isfile() or m.name in seen:raise RuntimeError('unexpected/nonfile/duplicate tar entry '+m.name)
   seen.add(m.name);f=tf.extractfile(m)
   if m.name=='archive-index.json':index=json.load(f)
   elif m.name=='snapshot/selection-plan.json':embedded_plan=f.read()
   elif m.name=='snapshot/snapshot-provenance.json':provenance=f.read()
   elif m.name.startswith('snapshot-tooling/'):tool_blobs[m.name]=f.read()
   elif m.name.startswith('blobs/'):
    b=f.read();d=m.name.split('/',1)[1]
    if len(b)!=m.size or sha(b)!=d:raise RuntimeError('blob SHA/size mismatch '+d)
    blobs[d]=b
   else:raise RuntimeError('unexpected tar member '+m.name)
 if index is None or embedded_plan is None or provenance is None:raise RuntimeError('archive metadata incomplete')
 if sha(embedded_plan)!=sha(planbytes):raise RuntimeError('embedded selection plan differs from pinned local plan')
 if index.get('schema')!='numi-human.quality-source-only-snapshot-archive.v1' or index.get('snapshot',{}).get('physical_admission') is not False or index.get('snapshot',{}).get('native_steps')!=0:raise RuntimeError('archive index scope/status mismatch')
 if sha(provenance)!=index.get('provenance_sha256'):raise RuntimeError('embedded provenance digest mismatch')
 if index.get('selection_plan_sha256')!=sha(planbytes):raise RuntimeError('selection plan hash mismatch')
 for mf in index.get('method_files',[]):
  name=mf['archive_member'];b=tool_blobs.get(name)
  if b is None or len(b)!=mf['size_bytes'] or sha(b)!=mf['sha256']:raise RuntimeError('snapshot tool hash mismatch '+name)
 expected={r['sha256']:r['size_bytes'] for r in plan['files']}
 if set(blobs)!=set(expected):raise RuntimeError('archive unique blob set differs from selected source files')
 if any(len(blobs[d])!=n for d,n in expected.items()):raise RuntimeError('blob size differs from selection plan')
 map_index={r['path']:(r['sha256'],r['size_bytes']) for r in index['files']}
 if map_index!={r['path']:(r['sha256'],r['size_bytes']) for r in plan['files']}:raise RuntimeError('archive index path/hash map mismatch')
 npz_records=[]; headers_total=0; arrays_total=0
 for row in plan['files']:
  if not row['path'].endswith('.npz'):continue
  digest=row['sha256'];blob=blobs[digest]
  with zipfile.ZipFile(io.BytesIO(blob),'r') as zf:
   bad=zf.testzip()
   if bad is not None:raise RuntimeError('NPZ CRC failed '+bad)
   names=[n for n in zf.namelist() if n.endswith('.npy')]
   if not names:raise RuntimeError('NPZ has no NPY members '+row['role'])
   headers=[]
   for name in names:
    with zf.open(name,'r') as f:headers.append(header(f,name))
  # Exact RAM rehydration from independent remote blob; load and hash every array.
  restored=io.BytesIO();restored.write(bytes(blob));restored.seek(0);rb=restored.read()
  if sha(rb)!=digest or len(rb)!=row['size_bytes']:raise RuntimeError('RAM restore hash/size mismatch '+row['role'])
  with zipfile.ZipFile(io.BytesIO(rb),'r') as zf:
   bad=zf.testzip()
   if bad is not None:raise RuntimeError('restored NPZ CRC failed '+bad)
  readbacks=[]
  with np.load(io.BytesIO(rb),allow_pickle=False) as z:
   for name in z.files:
    a=z[name]
    if a.dtype.hasobject:raise RuntimeError('object array rejected in '+row['role']+'/'+name)
    readbacks.append({'name':name,'shape':list(a.shape),'dtype':a.dtype.str,'array_bytes_sha256':sha(a.tobytes(order='C'))})
    arrays_total+=1
    del a
  headers_total+=len(headers)
  npz_records.append({'role':row['role'],'path':row['path'],'sha256':digest,'size_bytes':len(rb),'zip_crc':'PASS','npy_headers':headers,'ram_restore_sha256':sha(rb),'loaded_array_count':len(readbacks),'array_readbacks':readbacks})
 # Confirm each selected live input remained byte-identical; none is retired.
 post=[]
 for row in plan['files']:
  p=Path(row['path'])
  if not p.is_file() or p.is_symlink() or p.stat().st_size!=row['size_bytes'] or sha_file(p)!=row['sha256']:raise RuntimeError('source input changed during archive workflow '+row['role'])
  post.append({'role':row['role'],'path':str(p),'sha256':row['sha256'],'size_bytes':row['size_bytes'],'unchanged':True})
 report={'schema':'numi-human.source-only-snapshot-independent-gh-verification.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z'),'status':'PASS_INDEPENDENT_GH_ARCHIVE_FULL_BLOB_AND_EXACT_NPZ_RAM_RESTORE','repository':REPO,'release_tag':TAG,'release_url':meta.get('url'),'github_asset':{'name':ASSET,'id':asset.get('id'),'node_id':asset.get('node_id'),'digest':asset.get('digest'),'size_bytes':asset.get('size'),'archive_sha256_from_remote_stream':sha(archive),'archive_size_from_remote_stream':len(archive)},'selection_plan_sha256':sha(planbytes),'archive_index_canonical_sha256':sha(json.dumps(index,sort_keys=True,separators=(',',':')).encode()),'provenance_sha256':sha(provenance),'selected_file_count':len(plan['files']),'unique_blob_count':len(blobs),'original_bytes':sum(len(x) for x in blobs.values()),'independent_download_method':'gh release download --output - captured in memory; full archive SHA/size checked','decompression':'zstd -dc in memory; exact tar member set, archive index, embedded plan, provenance, all archived snapshot-tooling files and all content-addressed blobs checked','all_selected_npz_zip_crcs_passed':True,'all_selected_npy_headers_read':headers_total,'all_selected_npz_arrays_loaded':arrays_total,'exact_ram_restoration':'fresh BytesIO copy for every selected NPZ, exact SHA/size, CRC, every NPY array loaded and content-hashed','npz_files':npz_records,'selected_live_inputs_aftercheck':{'all_unchanged':True,'files':post},'snapshot_status':'SOURCE_ONLY_BATCH_PROPOSAL_PENDING_CURRENT_ALL185_AND_NATIVE_GATES','fat_native_quality_failures':4,'skin_native_quality_failures':5,'physical_admission':False,'native_steps':0,'mass_reconciliation_status':'PENDING_VOLUME_CHANGE_RECONCILIATION','full_original_ancestry_closure':False,'no_current_files_retired':True}
 out=OUT/'independent-verification.json';out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':report['status'],'tag':TAG,'asset_id':asset.get('id'),'sha256':sha(archive),'size':len(archive),'blobs':len(blobs),'NPZs':len(npz_records),'npy_headers':headers_total,'arrays_loaded':arrays_total,'receipt':str(out)},sort_keys=True))
if __name__=='__main__':main()

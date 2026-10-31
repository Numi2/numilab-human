#!/usr/bin/env python3
import argparse, hashlib, json, subprocess, tarfile
from pathlib import Path

def hash_stream(src, dsts):
 h=hashlib.sha256(); n=0
 try:
  while True:
   b=src.read(8*1024*1024)
   if not b: break
   h.update(b); n+=len(b)
   for dst in dsts: dst.write(b)
 finally:
  for dst in dsts: dst.flush()
 return h.hexdigest(),n

def main():
 ap=argparse.ArgumentParser(description='Stream-verify quality695 archive; optionally restore selected source paths under a chosen root')
 ap.add_argument('--archive',required=True); ap.add_argument('--restore-root'); ap.add_argument('--restore-path',action='append',default=[])
 a=ap.parse_args(); required=set(a.restore_path)
 if required and not a.restore_root: ap.error('--restore-path requires --restore-root')
 proc=subprocess.Popen(['/opt/homebrew/bin/zstd','-dc',a.archive],stdout=subprocess.PIPE)
 index=None; mapping={}; seen=set(); total=0; restored={}; verified=0
 try:
  with tarfile.open(fileobj=proc.stdout,mode='r|') as tf:
   for member in tf:
    if not member.isfile(): raise RuntimeError('non-file member rejected '+member.name)
    if member.name=='archive-index.json':
     if index is not None: raise RuntimeError('duplicate archive index')
     index=json.load(tf.extractfile(member)); mapping={}
     for row in index['files']: mapping.setdefault(row['sha256'],[]).append(row)
     continue
    if member.name in ('selection-and-retirement.json','checkpoint-snapshot.json'): continue
    if not member.name.startswith('blobs/') or '..' in Path(member.name).parts: raise RuntimeError('unexpected member '+member.name)
    if index is None: raise RuntimeError('archive index must precede blobs')
    h_expected=member.name.split('/',1)[1]
    if h_expected in seen: raise RuntimeError('duplicate blob '+h_expected)
    seen.add(h_expected)
    rows=mapping.get(h_expected)
    if not rows: raise RuntimeError('unindexed blob '+h_expected)
    outs=[]
    if required:
     for row in rows:
      if row['path'] not in required: continue
      rel=Path(row['source_relative_path'])
      if rel.is_absolute() or '..' in rel.parts: raise RuntimeError('unsafe restore path')
      dest=Path(a.restore_root)/rel; dest.parent.mkdir(parents=True,exist_ok=True)
      outs.append((row['path'],dest.open('wb')))
    src=tf.extractfile(member)
    actual,n=hash_stream(src,[x[1] for x in outs])
    for original,dst in outs: dst.close()
    if actual!=h_expected or n!=member.size: raise RuntimeError('blob SHA/size mismatch '+member.name)
    if any(n!=row['size_bytes'] for row in rows): raise RuntimeError('indexed size mismatch '+member.name)
    for original,dest in outs: restored[original]=str(dest)
    total+=n; verified+=1
  if proc.wait()!=0: raise RuntimeError('zstd decoder failed')
  if index is None or verified!=index['unique_blob_count'] or total!=index['unique_blob_bytes'] or seen!={x['sha256'] for x in index['unique_blobs']}: raise RuntimeError('archive/index inventory mismatch')
  if required and set(restored)!=required: raise RuntimeError('selected restore missing '+repr(sorted(required-set(restored))))
  for row in index['files']:
   if row['path'] in required:
    p=Path(restored[row['path']])
    if p.stat().st_size!=row['size_bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']: raise RuntimeError('restored member mismatch '+str(p))
  h=hashlib.sha256()
  with Path(a.archive).open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
  print(json.dumps({'status':'PASS_STREAMED_ARCHIVE_AND_SELECTED_RESTORE','archive_sha256':h.hexdigest(),'verified_blob_count':verified,'verified_blob_bytes':total,'restored_paths':sorted(required),'restore_root':a.restore_root,'physical_admission':False,'native_steps':0}))
 except BaseException:
  if proc.poll() is None: proc.kill(); proc.wait()
  raise
if __name__=='__main__': main()


#!/usr/bin/env python3
"""Stream-verify the release archive and optionally restore only the frozen 678 snapshot.

The verifier hashes every member without writing the full parent DAG. The optional
restore materializes the three 678 typed surfaces, current Skin/Fat NPZs, and
small state/readback files into a new destination using content-addressed blobs.
It never overwrites an existing differing file and never deletes files.
"""
import argparse, hashlib, json, os, subprocess, sys, tarfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()

def digest_stream(f, sink=None):
    h=hashlib.sha256(); n=0
    while True:
        b=f.read(8*1024*1024)
        if not b: break
        h.update(b); n+=len(b)
        if sink is not None: sink.write(b)
    return h.hexdigest(),n

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--archive",required=True,type=Path)
    ap.add_argument("--index",required=True,type=Path)
    ap.add_argument("--restore-current",type=Path)
    a=ap.parse_args()
    index=json.loads(a.index.read_text())
    expected={x["archive_member"]:(x["sha256"],int(x["size_bytes"])) for x in index["unique_blobs"]}
    selected=set()
    if a.restore_current:
        for x in index["files"]:
            kinds=set(x.get("kinds",[]))
            if x.get("state_index")==0 and (any(k.startswith("typed_surface_") for k in kinds) or any(k=="volume_Skin" or k=="volume_Fat" for k in kinds)):
                selected.add(x["sha256"])
            if any(k in kinds for k in ("finalized_Skin_volume","finalized_Fat_volume","final_metadata_state","metadata_finalizer_report","finalizer_readback")):
                selected.add(x["sha256"])
    restore_root=a.restore_current
    blob_dir=None
    if restore_root:
        blob_dir=restore_root/"blobs"
        blob_dir.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen(["zstd","-dc",str(a.archive)],stdout=subprocess.PIPE)
    seen=set(); internal_index_seen=False
    try:
        with tarfile.open(fileobj=proc.stdout,mode="r|") as tf:
            for member in tf:
                name=member.name.removeprefix("./")
                if member.isdir():
                    if name.rstrip("/") != "blobs": raise RuntimeError("unexpected archive directory "+name)
                    continue
                if not member.isfile(): raise RuntimeError("unexpected non-file archive member "+name)
                f=tf.extractfile(member)
                if f is None: raise RuntimeError("unreadable archive member "+name)
                if name=="archive-index.json":
                    got,n=digest_stream(f)
                    if n!=a.index.stat().st_size or got!=sha(a.index):
                        raise RuntimeError("embedded archive index differs from published index")
                    internal_index_seen=True
                    continue
                if name not in expected: raise RuntimeError("unexpected archive member "+name)
                if name in seen: raise RuntimeError("duplicate archive member "+name)
                want_hash,want_size=expected[name]
                tmp=None; out=None
                if want_hash in selected:
                    tmp=blob_dir/(want_hash+".part")
                    final=blob_dir/want_hash
                    if final.exists():
                        if final.stat().st_size!=want_size or sha(final)!=want_hash:
                            raise RuntimeError("existing restore blob differs "+str(final))
                    else:
                        if tmp.exists(): raise RuntimeError("partial restore blob already exists "+str(tmp))
                        out=tmp.open("xb")
                try:
                    got,n=digest_stream(f,out)
                finally:
                    if out is not None: out.close()
                if got!=want_hash or n!=want_size:
                    raise RuntimeError(f"blob hash/size mismatch {name}: {got}/{n}")
                if tmp is not None and out is not None:
                    os.replace(tmp,blob_dir/want_hash)
                seen.add(name)
        rc=proc.wait()
        if rc: raise RuntimeError("zstd exited "+str(rc))
    finally:
        if proc.stdout: proc.stdout.close()
    if not internal_index_seen: raise RuntimeError("archive-index.json absent")
    if seen!=set(expected):
        raise RuntimeError(f"archive members missing: {len(set(expected)-seen)}")
    if restore_root:
        for x in index["files"]:
            if x["sha256"] not in selected or not x.get("source_relative_path"): continue
            dest=restore_root/"restored-tree"/x["source_relative_path"]
            dest.parent.mkdir(parents=True,exist_ok=True)
            blob=blob_dir/x["sha256"]
            if dest.exists():
                if dest.stat().st_size!=x["size_bytes"] or sha(dest)!=x["sha256"]:
                    raise RuntimeError("restore destination conflicts "+str(dest))
            else:
                os.link(blob,dest)
    print(json.dumps({"status":"PASS_STREAMED_ARCHIVE_HASH_AND_RESTORE_CHECK","archive_sha256":sha(a.archive),"archive_size_bytes":a.archive.stat().st_size,"verified_blob_count":len(seen),"verified_blob_bytes":sum(x[1] for x in expected.values()),"restored_current_snapshot_blobs":len(selected),"restore_root":str(restore_root) if restore_root else None,"physical_admission":False,"native_steps":0},indent=2))

if __name__=="__main__": main()

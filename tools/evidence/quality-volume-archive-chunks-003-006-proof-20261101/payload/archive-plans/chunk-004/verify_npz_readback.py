#!/usr/bin/env python3
"""Streaming ZIP CRC and NPY header readback for the independently downloaded chunk archive."""
import ast
import hashlib
import io
import json
import os
import struct
import subprocess
import tarfile
import zipfile
from pathlib import Path

import numpy as np

ARCHIVE = Path("/Users/numi/Work/.quality-volume-archive-chunk-004-independent-download/quality-volume-archive-chunk-004.tar.zst")
OUT = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/quality-volume-archive-chunk-004-plan/npz-readback-report.json")
TEMP = OUT.parent / ".chunk004-npz-readback.tmp"
EXPECTED_ARCHIVE_SHA = "7733a785fa9a7d113bb2085e9ad5e0893119d5e0fc76b8591521d22342bebf30"

def sha_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def header(fp, name):
    ver=np.lib.format.read_magic(fp)
    if ver==(1,0):
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(fp)
    elif ver==(2,0):
        shape,fortran,dtype=np.lib.format.read_array_header_2_0(fp)
    elif ver==(3,0):
        length_raw=fp.read(4)
        if len(length_raw)!=4: raise RuntimeError("truncated v3 NPY header: "+name)
        length=struct.unpack("<I",length_raw)[0]
        if length>10000: raise RuntimeError("oversized v3 NPY header: "+name)
        raw=fp.read(length)
        if len(raw)!=length: raise RuntimeError("truncated v3 NPY header body: "+name)
        obj=ast.literal_eval(raw.decode("utf-8").strip())
        shape=tuple(obj["shape"]); fortran=bool(obj["fortran_order"]); dtype=np.dtype(obj["descr"])
    else:
        raise RuntimeError("unsupported NPY version "+repr(ver)+" in "+name)
    if not isinstance(shape,tuple) or not isinstance(fortran,bool): raise RuntimeError("invalid header fields: "+name)
    return {"name":name,"version":list(ver),"shape":list(shape),"fortran_order":fortran,"dtype":repr(dtype.descr),"dtype_str":dtype.str,"hasobject":bool(dtype.hasobject)}

def main():
    if TEMP.exists() or OUT.exists(): raise RuntimeError("refusing existing readback output/temp")
    actual_archive=sha_file(ARCHIVE)
    if actual_archive!=EXPECTED_ARCHIVE_SHA: raise RuntimeError("independent archive SHA mismatch")
    proc=subprocess.Popen(["/opt/homebrew/bin/zstd","-dc",str(ARCHIVE)],stdout=subprocess.PIPE)
    index=None; mapping={}; seen=set(); verified=0; total=0; headers_total=0; records=[]
    try:
        with tarfile.open(fileobj=proc.stdout,mode="r|") as tf:
            for member in tf:
                if not member.isfile(): raise RuntimeError("non-file tar member: "+member.name)
                if member.name=="archive-index.json":
                    if index is not None: raise RuntimeError("duplicate archive index")
                    index=json.load(tf.extractfile(member))
                    for row in index["files"]: mapping.setdefault(row["sha256"],[]).append(row)
                    continue
                if member.name in ("selection-and-retirement.json","checkpoint-snapshot.json"): continue
                if not member.name.startswith("blobs/") or ".." in Path(member.name).parts: raise RuntimeError("unexpected member: "+member.name)
                if index is None: raise RuntimeError("blob before archive index")
                digest=member.name.split("/",1)[1]
                if digest in seen: raise RuntimeError("duplicate blob "+digest)
                seen.add(digest)
                rows=mapping.get(digest)
                if not rows: raise RuntimeError("unindexed blob "+digest)
                h=hashlib.sha256(); n=0
                with TEMP.open("xb") as out:
                    src=tf.extractfile(member)
                    while True:
                        b=src.read(8*1024*1024)
                        if not b: break
                        out.write(b); h.update(b); n+=len(b)
                if h.hexdigest()!=digest or n!=member.size: raise RuntimeError("blob digest/size mismatch "+digest)
                if any(row["size_bytes"]!=n for row in rows): raise RuntimeError("index size mismatch "+digest)
                file_record={"sha256":digest,"size_bytes":n,"source_paths":[r["source_relative_path"] for r in rows]}
                with zipfile.ZipFile(TEMP,"r") as zf:
                    bad=zf.testzip()
                    if bad is not None: raise RuntimeError("NPZ CRC failure "+digest+":"+bad)
                    members=[name for name in zf.namelist() if name.endswith(".npy")]
                    if not members: raise RuntimeError("NPZ has no NPY arrays "+digest)
                    hh=[]
                    for name in members:
                        with zf.open(name,"r") as f: hh.append(header(f,name))
                    file_record["zip_crc"]="PASS"; file_record["npy_header_count"]=len(hh)
                    file_record["npy_header_summary_sha256"]=hashlib.sha256(json.dumps(hh,sort_keys=True,separators=(",",":")).encode()).hexdigest()
                    file_record["npy_versions"]=sorted({tuple(x["version"]) for x in hh})
                TEMP.unlink()
                records.extend([file_record])
                headers_total+=file_record["npy_header_count"]; total+=n; verified+=1
                print("VERIFIED_BLOB",verified,"/",index["unique_blob_count"],"bytes",n,"headers",file_record["npy_header_count"],flush=True)
        if proc.wait()!=0: raise RuntimeError("zstd decoder exit "+str(proc.returncode))
        if index is None or verified!=index["unique_blob_count"] or total!=index["unique_blob_bytes"] or seen!={x["sha256"] for x in index["unique_blobs"]}: raise RuntimeError("archive/index inventory mismatch")
        report={"schema":"numi-human.quality-volume-npz-readback.v1","status":"PASS_ALL_NPZ_ZIP_CRC_AND_NPY_HEADERS","independent_archive_path":str(ARCHIVE),"archive_sha256":actual_archive,"archive_index_sha256":hashlib.sha256(json.dumps(index,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"npz_count":verified,"uncompressed_blob_bytes":total,"npy_array_headers_read":headers_total,"all_zip_crcs_passed":True,"files":records,"physical_admission":False,"native_steps":0}
        OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"status":report["status"],"npz_count":verified,"headers":headers_total,"bytes":total,"report":str(OUT),"archive_sha256":actual_archive},sort_keys=True))
    except BaseException:
        if proc.poll() is None: proc.kill(); proc.wait()
        if TEMP.exists(): TEMP.unlink()
        raise
if __name__=="__main__": main()

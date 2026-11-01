#!/usr/bin/env python3
"""Complete-payload semantics layered over the frozen 925 NPZ COW writer."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import os, uuid, zipfile

HERE=Path(__file__).resolve().parent
WRITER=HERE.parent/"npz-cow-append-925-root"/"npz_cow_append_925.py"
WRITER_SHA="4799f33d73a37072498565b6ff9c20d181916f0158e253fe5c7e9056ba2c0270"

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def _writer():
    if _sha(WRITER)!=WRITER_SHA: raise RuntimeError("frozen 925 writer hash mismatch")
    spec=importlib.util.spec_from_file_location("npz_cow_append_frozen_925",WRITER)
    if spec is None or spec.loader is None: raise RuntimeError("cannot import frozen writer")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def _noop_clone(w,parent_path,dest_path,payload):
    """Byte-identical COW clone for an empty delta; 925 append mode needs a ZIP member."""
    src=Path(parent_path).resolve(strict=True); dst=Path(dest_path).absolute()
    if not dst.parent.is_dir(): raise FileNotFoundError(dst.parent)
    if dst.exists(): raise FileExistsError(dst)
    if src==dst: raise ValueError("source and destination must differ")
    parent_sha=w._sha256(src); st=src.stat()
    tmp=dst.parent/("."+dst.name+".noop-"+uuid.uuid4().hex)
    try:
        w._clonefile(src,tmp)
        with zipfile.ZipFile(src,"r",allowZip64=True) as z0, zipfile.ZipFile(tmp,"r",allowZip64=True) as z1:
            n0=z0.namelist(); n1=z1.namelist()
            if len(n0)!=len(set(n0)) or len(n1)!=len(set(n1)) or n0!=n1:
                raise zipfile.BadZipFile("no-op clone central-directory mismatch")
            if z0.testzip() is not None or z1.testzip() is not None:
                raise zipfile.BadZipFile("no-op clone CRC failure")
            i0={i.filename:i for i in z0.infolist()}; i1={i.filename:i for i in z1.infolist()}
            for name in n0:
                if (i0[name].header_offset!=i1[name].header_offset or
                    w._raw_member_sha(src,i0[name])!=w._raw_member_sha(tmp,i1[name])):
                    raise AssertionError("no-op clone changed compressed member")
        count=0
        with np.load(src,allow_pickle=False) as a, np.load(tmp,allow_pickle=False) as b:
            if a.files!=b.files or set(a.files)!=set(payload): raise AssertionError("no-op payload key mismatch")
            for key in a.files:
                if a[key].dtype.hasobject or b[key].dtype.hasobject: raise TypeError("object array forbidden")
                if not w._same_array(a[key],payload[key]) or not w._same_array(a[key],b[key]):
                    raise AssertionError("no-op complete payload differs")
                count+=1
        if w._sha256(src)!=parent_sha: raise RuntimeError("parent changed during no-op clone")
        fd=os.open(tmp,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        w._atomic_rename_exclusive(tmp,dst)
        dfd=os.open(dst.parent,os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
        return {"schema":"numi.standard-npz-apfs-clone-append.v1","status":"PASS_VERIFIED_STANDARD_NPZ",
            "source_path":str(src),"source_sha256":parent_sha,"source_size_bytes":int(st.st_size),
            "output_path":str(dst),"output_sha256":w._sha256(dst),"output_size_bytes":int(dst.stat().st_size),
            "parent_unchanged":True,"apfs_clonefile_used":True,"full_copy_fallback":False,
            "source_member_count":count,"output_member_count":count,
            "replaced_keys":[],"new_keys":[],"deleted_keys":[],"readback_arrays_bitwise_verified":count,
            "zip_crc":"PASS","duplicate_central_names":0,
            "unchanged_compressed_members_bitwise_verified":count,
            "unchanged_compressed_members":[{"name":name,"header_offset":int(i1[name].header_offset),
                "compressed_bytes":int(i1[name].compress_size),
                "compressed_sha256":w._raw_member_sha(dst,i1[name])} for name in n1]}
    except Exception:
        try:
            if tmp.exists(): tmp.unlink()
        except Exception: pass
        raise

def _delete_only_clone(w,parent_path,dest_path,payload,deleted_keys):
    """925 cannot publish a deletion-only append (empty delta leaves no new EOCD)."""
    src=Path(parent_path).resolve(strict=True); dst=Path(dest_path).absolute()
    if not dst.parent.is_dir(): raise FileNotFoundError(dst.parent)
    if dst.exists(): raise FileExistsError(dst)
    parent_sha=w._sha256(src); st=src.stat(); tmp=dst.parent/("."+dst.name+".delete-"+uuid.uuid4().hex)
    deleted_names={w._key_name(k) for k in deleted_keys}
    try:
        w._clonefile(src,tmp)
        with zipfile.ZipFile(tmp,"a",allowZip64=True) as zf:
            infos=list(zf.infolist()); names=[i.filename for i in infos]
            if len(names)!=len(set(names)) or any(not n.endswith(".npy") for n in names):
                raise zipfile.BadZipFile("invalid parent NPZ directory")
            if deleted_names-set(names): raise KeyError("delete key absent from parent")
            zf.filelist=[i for i in infos if i.filename not in deleted_names]
            zf.NameToInfo={i.filename:i for i in zf.filelist}
            zf._didModify=True
            zf.fp.flush(); zf.fp.seek(zf.start_dir); zf.fp.truncate()
        with zipfile.ZipFile(src,"r") as a, zipfile.ZipFile(tmp,"r") as b:
            if a.testzip() is not None or b.testzip() is not None: raise zipfile.BadZipFile("CRC failure")
            am={i.filename:i for i in a.infolist()}; bm={i.filename:i for i in b.infolist()}
            if set(bm)!=(set(am)-deleted_names): raise AssertionError("deleted member set mismatch")
            for name,bi in bm.items():
                ai=am[name]
                if ai.header_offset!=bi.header_offset or w._raw_member_sha(src,ai)!=w._raw_member_sha(tmp,bi):
                    raise AssertionError("retained compressed member changed")
        with np.load(tmp,allow_pickle=False) as out:
            if set(out.files)!=set(payload): raise AssertionError("deletion readback keys differ")
            for key,value in payload.items():
                if out[key].dtype.hasobject or not w._same_array(value,out[key]):
                    raise AssertionError(f"deletion readback mismatch for {key}")
        if w._sha256(src)!=parent_sha: raise RuntimeError("parent changed during deletion append")
        fd=os.open(tmp,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        w._atomic_rename_exclusive(tmp,dst)
        dfd=os.open(dst.parent,os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
        return {"schema":"numi.standard-npz-apfs-clone-append.v1","status":"PASS_VERIFIED_STANDARD_NPZ",
            "source_path":str(src),"source_sha256":parent_sha,"source_size_bytes":int(st.st_size),
            "output_path":str(dst),"output_sha256":w._sha256(dst),"output_size_bytes":int(dst.stat().st_size),
            "parent_unchanged":True,"apfs_clonefile_used":True,"full_copy_fallback":False,
            "source_member_count":len(am),"output_member_count":len(bm),"replaced_keys":[],
            "new_keys":[],"deleted_keys":sorted(deleted_keys),"readback_arrays_bitwise_verified":len(payload),
            "zip_crc":"PASS","duplicate_central_names":0,
            "unchanged_compressed_members_bitwise_verified":len(bm),"unchanged_compressed_members":[]}
    except Exception:
        try:
            if tmp.exists(): tmp.unlink()
        except Exception: pass
        raise

def write_npz_clone_complete(parent_path,dest_path,payload):
    """Treat payload as the complete desired array set, diffing before ZIP append."""
    w=_writer()
    desired=dict(payload)
    desired_names={}
    for key,value in desired.items():
        name=w._key_name(key)
        arr=np.asarray(value)
        if arr.dtype.hasobject: raise TypeError(f"object array forbidden for {key!r}")
        desired_names[name]=arr
    replacements={}
    unchanged=[]
    with np.load(parent_path,allow_pickle=False) as parent:
        parent_keys=list(parent.files)
        if len(parent_keys)!=len(set(parent_keys)): raise ValueError("duplicate NPZ keys in parent")
        missing=set(parent_keys)-set(desired)
        added=set(desired)-set(parent_keys)
        for key in parent_keys:
            if key not in desired: continue
            if w._same_array(parent[key],desired[key]): unchanged.append(key)
            else: replacements[key]=desired[key]
    deleted=sorted(missing)
    added=sorted(added)
    for key in added: replacements[key]=desired[key]
    if not replacements and not deleted:
        report=_noop_clone(w,parent_path,dest_path,desired)
    elif not replacements:
        report=_delete_only_clone(w,parent_path,dest_path,desired,deleted)
    else:
        report=w.write_npz_clone(parent_path,dest_path,replacements,delete_keys=deleted)
    report["schema"]="numi.standard-npz-apfs-clone-append-complete-payload.v1"
    report["complete_payload_contract"]=True
    report["input_payload_array_count"]=len(desired)
    report["unchanged_payload_keys"]=sorted(unchanged)
    report["changed_existing_keys"]=sorted(replacements.keys()-set(added))
    report["new_keys"]=added
    report["deleted_keys"]=deleted
    return report


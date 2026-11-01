#!/usr/bin/env python3
"""Write a standard NPZ using an APFS clone and ZIP append-only entries.

Unchanged compressed member payloads stay at their original offsets. Replaced/new
arrays are appended as ordinary NPY ZIP members and deleted names are omitted from
the new central directory. The source is never modified; clone failure is fatal.
"""
from __future__ import annotations
import ctypes
import errno
import hashlib
import io
import os
from pathlib import Path
import struct
import uuid
import zipfile
import numpy as np

_RENAME_EXCL = 0x00000004

def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def _key_name(key):
    if not isinstance(key,str) or not key or key.endswith(".npy") or "/" in key or "\\" in key:
        raise ValueError(f"invalid NPZ key: {key!r}")
    return key+".npy"

def _array(value,key):
    a=np.asarray(value)
    if a.dtype.hasobject: raise TypeError(f"object array forbidden for key {key!r}")
    return a

def _raw_member_sha(path,info):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        f.seek(info.header_offset)
        hdr=f.read(30)
        if len(hdr)!=30 or struct.unpack_from("<I",hdr)[0]!=0x04034b50:
            raise zipfile.BadZipFile(f"bad local header for {info.filename}")
        nlen,xlen=struct.unpack_from("<HH",hdr,26)
        f.seek(info.header_offset+30+nlen+xlen)
        left=int(info.compress_size)
        while left:
            b=f.read(min(1<<20,left))
            if not b: raise zipfile.BadZipFile(f"truncated compressed data for {info.filename}")
            h.update(b); left-=len(b)
    return h.hexdigest()

def _same_array(a,b):
    return (a.dtype==b.dtype and a.shape==b.shape and
            a.flags.f_contiguous==b.flags.f_contiguous and
            a.tobytes(order="A")==b.tobytes(order="A"))

def _clonefile(src,dst):
    if os.uname().sysname!="Darwin": raise OSError("APFS clonefile requires macOS")
    lib=ctypes.CDLL("/usr/lib/libSystem.B.dylib",use_errno=True)
    fn=getattr(lib,"clonefile",None)
    if fn is None: raise OSError("libSystem clonefile is unavailable; full-copy fallback is disabled")
    fn.argtypes=(ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int); fn.restype=ctypes.c_int
    rc=fn(os.fsencode(src),os.fsencode(dst),0)
    if rc!=0:
        e=ctypes.get_errno() or errno.EIO
        raise OSError(e,os.strerror(e),str(src),str(dst))

def _atomic_rename_exclusive(src,dst):
    lib=ctypes.CDLL("/usr/lib/libSystem.B.dylib",use_errno=True)
    fn=getattr(lib,"renamex_np",None)
    if fn is not None:
        fn.argtypes=(ctypes.c_char_p,ctypes.c_char_p,ctypes.c_uint); fn.restype=ctypes.c_int
        if fn(os.fsencode(src),os.fsencode(dst),_RENAME_EXCL)!=0:
            e=ctypes.get_errno() or errno.EIO
            raise OSError(e,os.strerror(e),str(src),str(dst))
    else:
        if Path(dst).exists(): raise FileExistsError(dst)
        os.rename(src,dst)

def write_npz_clone(parent_path,dest_path,payload,*,delete_keys=()):
    """Create dest_path from parent_path with standard NPZ semantics.

    payload maps array keys to replacement/new arrays; delete_keys are omitted from
    output. Destination must be new and on the same APFS volume. All arrays are
    checked with allow_pickle=False.
    """
    src=Path(parent_path).resolve(strict=True); dst=Path(dest_path).absolute()
    if not src.is_file(): raise ValueError("parent must be a regular file")
    if not dst.parent.is_dir(): raise FileNotFoundError(f"destination directory missing: {dst.parent}")
    if dst.exists(): raise FileExistsError(f"destination already exists: {dst}")
    if src==dst: raise ValueError("source and destination must differ")
    changes={_key_name(k):_array(v,k) for k,v in dict(payload).items()}
    deleted={_key_name(k) for k in delete_keys}
    if set(changes)&deleted: raise ValueError("a key cannot be both replaced and deleted")
    parent_sha=_sha256(src); parent_stat=src.stat()
    temp=dst.parent/("."+dst.name+".cowtmp-"+uuid.uuid4().hex)
    try:
        _clonefile(src,temp)
        if temp.stat().st_dev!=parent_stat.st_dev:
            raise OSError("APFS clone landed on a different device")
        if _sha256(src)!=parent_sha: raise RuntimeError("parent changed during clone")
        with zipfile.ZipFile(temp,"a",allowZip64=True) as zf:
            infos=list(zf.infolist()); names=[i.filename for i in infos]
            if len(names)!=len(set(names)): raise zipfile.BadZipFile("duplicate central-directory member names in parent")
            if any(not n.endswith(".npy") for n in names): raise zipfile.BadZipFile("parent is not a plain NPZ archive")
            with np.load(src,allow_pickle=False) as parent:
                if set(parent.files)!={n[:-4] for n in names}: raise zipfile.BadZipFile("NPZ key/index mismatch")
                for k in parent.files:
                    if parent[k].dtype.hasobject: raise TypeError(f"object array forbidden in source key {k!r}")
            unknown_delete=deleted-set(names)
            if unknown_delete: raise KeyError(f"delete keys absent from parent: {sorted(unknown_delete)}")
            removed=set(changes)|deleted
            zf.filelist=[i for i in infos if i.filename not in removed]
            zf.NameToInfo={i.filename:i for i in zf.filelist}
            counts={}
            for i in infos: counts[i.compress_type]=counts.get(i.compress_type,0)+1
            ctype=max(counts,key=counts.get) if counts else zipfile.ZIP_STORED
            if ctype not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):
                raise NotImplementedError(f"unsupported NPZ compression type {ctype}")
            for name,a in changes.items():
                bio=io.BytesIO()
                np.lib.format.write_array(bio,a,allow_pickle=False)
                zf.writestr(name,bio.getvalue(),compress_type=ctype,
                    compresslevel=6 if ctype==zipfile.ZIP_DEFLATED else None)
            # Remove stale central-directory/EOCD tail before writing the new one.
            zf.fp.flush(); zf.fp.seek(zf.start_dir); zf.fp.truncate()
        fd=os.open(temp,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        if _sha256(src)!=parent_sha: raise RuntimeError("parent changed during append")
        with zipfile.ZipFile(temp,"r",allowZip64=True) as zf:
            names2=zf.namelist()
            if len(names2)!=len(set(names2)): raise zipfile.BadZipFile("duplicate names after append")
            if any(not n.endswith(".npy") for n in names2): raise zipfile.BadZipFile("non-NPY member after append")
            bad=zf.testzip()
            if bad is not None: raise zipfile.BadZipFile(f"CRC failure in {bad}")
            out_map={i.filename:i for i in zf.infolist()}
        expected_names=(set(names)-removed)|set(changes)
        if set(out_map)!=expected_names: raise AssertionError("output member set differs from requested edits")
        old_map={i.filename:i for i in infos}; unchanged=set(names)-removed; preserved=[]
        for name in sorted(unchanged):
            a=old_map[name]; b=out_map[name]
            da=_raw_member_sha(src,a); db=_raw_member_sha(temp,b)
            if (a.header_offset!=b.header_offset or a.compress_size!=b.compress_size or
                a.compress_type!=b.compress_type or a.CRC!=b.CRC or da!=db):
                raise AssertionError(f"unchanged compressed member changed: {name}")
            preserved.append({"name":name,"header_offset":int(b.header_offset),
                "compressed_bytes":int(b.compress_size),"compressed_sha256":db})
        compared=0
        with np.load(src,allow_pickle=False) as parent, np.load(temp,allow_pickle=False) as output:
            if set(output.files)!={n[:-4] for n in expected_names}: raise AssertionError("readback keys differ")
            for name in sorted(expected_names):
                key=name[:-4]; want=changes[name] if name in changes else parent[key]; got=output[key]
                if want.dtype.hasobject or got.dtype.hasobject: raise TypeError(f"object array in key {key}")
                if not _same_array(want,got): raise AssertionError(f"dtype/shape/order/bit-byte mismatch for {key}")
                compared+=1
        if _sha256(src)!=parent_sha: raise RuntimeError("parent changed during verification")
        _atomic_rename_exclusive(temp,dst)
        dfd=os.open(dst.parent,os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
        return {"schema":"numi.standard-npz-apfs-clone-append.v1","status":"PASS_VERIFIED_STANDARD_NPZ",
            "source_path":str(src),"source_sha256":parent_sha,"source_size_bytes":int(parent_stat.st_size),
            "output_path":str(dst),"output_sha256":_sha256(dst),"output_size_bytes":int(dst.stat().st_size),
            "parent_unchanged":True,"apfs_clonefile_used":True,"full_copy_fallback":False,
            "source_member_count":len(names),"output_member_count":len(expected_names),
            "replaced_keys":sorted(n[:-4] for n in changes if n in set(names)),
            "new_keys":sorted(n[:-4] for n in changes if n not in set(names)),
            "deleted_keys":sorted(n[:-4] for n in deleted),"readback_arrays_bitwise_verified":compared,
            "zip_crc":"PASS","duplicate_central_names":0,
            "unchanged_compressed_members_bitwise_verified":len(preserved),
            "unchanged_compressed_members":preserved}
    except Exception:
        try:
            if temp.exists(): temp.unlink()
        except Exception: pass
        raise


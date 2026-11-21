"""Validate the material edit scope before using the existing per-owner NHTENDON3 recook."""
import json,struct,sys
from pathlib import Path
import numpy as np
from shoulder_inputs import digest

def validate(parent,new):
    a=parent.read_bytes();b=new.read_bytes()
    assert len(a)==len(b)
    d=json.loads(new.with_suffix('.manifest.json').read_text())
    assert d['payload']['sha256']==digest(new) and d['source']['connected_girdle_parent']['bone_sha256']==digest(parent)
    count,vcount,icount=struct.unpack_from('<3I',a,12);start=60+60*count;end=start+24*vcount
    assert len(b)==end+4*icount and a[:start]==b[:start] and a[end:]==b[end:]
    owners={};allowed=np.zeros(vcount,bool)
    for i in range(count):
        body,first,n,_,_,sid=struct.unpack_from('<6I',a,60+60*i);assert sid==i+1
        if sid in (8,9):allowed[first:first+n]=True;owners[body]=sid
    aa=np.frombuffer(a[start:end],np.uint8).reshape(-1,24);bb=np.frombuffer(b[start:end],np.uint8).reshape(-1,24)
    assert np.array_equal(aa[~allowed],bb[~allowed]) and set(owners.values())=={8,9}
    return set(owners.values()),owners

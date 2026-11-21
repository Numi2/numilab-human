from contextlib import contextmanager
import hashlib,subprocess
import numpy as np
from material_context import sha
@contextmanager
def stream(path):
    if path.suffix!='.zst':
        with path.open('rb') as f:yield f
        return
    with subprocess.Popen(['zstd','-qdc',path],stdout=subprocess.PIPE) as p:
        yield p.stdout
        if p.wait()!=0:raise ValueError('Compressed source unreadable')
def keep_delta(base,modified,out):
    total=0;parent=hashlib.sha256()
    with stream(base) as a,modified.open('rb') as b,out.open('wb') as dst,subprocess.Popen(['zstd','-q','-6','-c'],stdin=subprocess.PIPE,stdout=dst) as z:
        while x:=a.read(2**20):
            y=b.read(len(x))
            if len(y)!=len(x):raise ValueError('Pack sizes differ')
            parent.update(x);total+=len(y)
            z.stdin.write((np.frombuffer(x,'u1')^np.frombuffer(y,'u1')).tobytes())
        if b.read(1):raise ValueError('Candidate pack has excess bytes')
        z.stdin.close()
        if z.wait()!=0:raise ValueError('Compression failure')
    recovered=hashlib.sha256()
    with stream(base) as a,stream(out) as b:
        while x:=a.read(2**20):
            y=b.read(len(x))
            if len(y)!=len(x):raise ValueError('Short recovery delta')
            recovered.update((np.frombuffer(x,'u1')^np.frombuffer(y,'u1')).tobytes())
        if b.read(1):raise ValueError('Long recovery delta')
    if recovered.hexdigest()!=sha(modified):raise ValueError('Recovery identity mismatch')
    return dict(parent=str(base),parent_decompressed_sha256=parent.hexdigest(),file=modified.name,restored_sha256=recovered.hexdigest(),bytes=total,delta=out.name,delta_sha256=sha(out))

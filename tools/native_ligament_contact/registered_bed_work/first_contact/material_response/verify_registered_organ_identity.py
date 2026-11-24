"""Compare the admitted physical organ's source geometry and typed identity across anatomy revisions."""
from pathlib import Path
import sys,struct,hashlib,json
import numpy as np
from compile_connected_joint import digest
ALLOWED={'1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241','3b0acb6146358f3302f2b85e2214bafa9f2fae7b0e51af70fd0838d6609da4a4'}
def record(path):
    raw=path.read_bytes();sig,abi,N,V,I,fp,source=struct.unpack_from('<8s5I32s',raw);rs=np.frombuffer(raw,'<u4',N*8,60).reshape(-1,8);assert len(raw)==60+N*32+V*24+I*4
    found=rs[rs[:,5]==463];assert len(found)==1;b,fv,n,fi,ic,s,l,res=map(int,found[0]);assert b==20 and n==223 and ic==1326
    v=raw[60+32*N+fv*24:60+32*N+(fv+n)*24];t=np.frombuffer(raw,'<u4',ic,60+32*N+V*24+fi*4);assert t.min()>=fv and t.max()<fv+n
    return struct.pack('<5I',s,b,l,n,ic)+v+(t-fv).astype('<u4').tobytes(),dict(stable_id=s,body=b,layer=l,vertices=n,oriented_triangles=ic//3,payload_hash=hashlib.sha256(raw).hexdigest(),source_payload_path=str(path),registration_fingerprint=fp,first_vertex=fv,first_index=fi)
def run(src,current,out):
    a,A=record(src);b,B=record(current)
    if A['payload_hash'] not in ALLOWED or a!=b:raise ValueError('Source organ record differs')
    out.write_text(json.dumps(dict(reference=A,current=B,organ_record_sha256=hashlib.sha256(a).hexdigest(),compared_bytes=len(a)),indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))

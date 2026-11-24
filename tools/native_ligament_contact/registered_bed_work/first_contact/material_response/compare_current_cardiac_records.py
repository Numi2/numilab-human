"""Check every physical and passive vertex range used by the cardiac field against its source payload."""
from pathlib import Path
import struct,json,sys
import numpy as np
from compile_connected_joint import digest

def load_records(path):
    data=path.read_bytes();_,abi,n,v,i,_,_=struct.unpack_from('<8s5I32s',data)
    if abi!=5 or len(data)!=60+32*n+24*v+4*i:raise ValueError('Invalid anatomy')
    records=np.frombuffer(data,'<u4',n*8,60).reshape(-1,8);out={}
    for b,fv,nv,fi,ni,s,l,res in records.astype(int):
        if s in out:raise ValueError('Duplicate anatomical identity')
        xyz=np.frombuffer(data,'<f4',6*nv,60+32*n+24*fv).reshape(-1,6)
        triangles=np.frombuffer(data,'<u4',ni,60+32*n+24*v+4*fi).astype(int)-fv
        if ni%3 or not nv or triangles.min()<0 or triangles.max()>=nv:raise ValueError('Invalid anatomy triangle')
        out[s]=(b,l,res,xyz,triangles)
    return out
def run(job,base,output):
    read=lambda p:json.loads(p.read_bytes());a=read(base/'launch.json')['argv'];p=Path(a[a.index('--resting-anatomy-receipt')+1]);c=read(p)['provenance']['cardiac_geometry_binding']['common_field'];source=Path(read(job/'registered-organ-viewer/records.json')['source_path']);current=Path(a[a.index('--torso-anatomy-payload')+1])
    if digest(source)!=c['anatomy_payload_sha256']:raise ValueError('Historical field source differs')
    old,new=map(load_records,[source,current]);records=[];sidecars={}
    for key in ['map','polynomials','domain_boxes']:
        spec=c[key];f=source.parent/spec['path']
        if not f.is_file() or digest(f)!=spec['sha256']:raise ValueError('Cardiac data asset mismatch: '+key)
        sidecars[key]={'path':str(f),'name':spec['path'],'sha256':digest(f)}
    ranges=c['vertex_ranges']+c['passive_attachment_ranges']
    for i,v in enumerate(ranges):
        sid=v['stable_id'];x=old[sid];y=new[sid]
        if x[:3]!=y[:3] or v['vertex_count']!=len(x[3]) or v['vertex_count']!=len(y[3]):raise ValueError('Cardiac identity or vertex count changed at '+str(sid))
        p0=x[3];p1=y[3];diff=np.any(p0.view('<u4')!=p1.view('<u4'),axis=1);same_tri=np.array_equal(x[4],y[4])
        records.append(dict(id=int(sid),physical=i<len(c['vertex_ranges']),vertices=len(p0),changed_vertices=int(diff.sum()),max_position_change_m=float(np.linalg.norm(p0[:,:3].astype(float)-p1[:,:3],axis=1).max()),triangle_parity=same_tri))
    output.write_text(json.dumps(dict(current=digest(current),original=digest(source),receipt=digest(p),sidecars=sidecars,ranges=records),indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))

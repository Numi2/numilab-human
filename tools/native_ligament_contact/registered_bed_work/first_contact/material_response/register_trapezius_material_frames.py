"""Transport the connected, historically verified bilateral trapezius through the current skeletal frames."""
from pathlib import Path
import os,sys,struct,json,subprocess
import numpy as np
from scipy.spatial.transform import Rotation
from audit_current_tissue_parents import records,run as audit
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(job,base,historic,out):
    rd=lambda p:json.loads(p.read_bytes());cfg=rd(base/'launch.json');oldcfg=rd(historic/'launch.json')
    def read_sources(c):
        a=c['argv'];receipt=Path(a[a.index('--resting-anatomy-receipt')+1]);child=Path(a[a.index('--soft-tissue-payload')+1]);d=rd(receipt);p=d['passive_tissue_parent_bindings']
        if digest(child)!=p['child_payload_sha256'] or c['pins'][str(child)]!=digest(child):raise ValueError('Different child')
        return d,p,child
    doc,p,child=read_sources(cfg);_,op,oldchild=read_sources(oldcfg)
    ids=set(range(153,159));sources={};out.mkdir()
    poses=[rd(d/'registered-initial-bodies.json')['all_bodies'] for d in (historic,base)]
    def transform(z):
        m=np.eye(4);m[:3,:3]=Rotation.from_quat(z['orientation_world_xyzw']).as_matrix();m[:3,3]=z['position_world_m'];return m
    o,n=[np.array([transform(row['rest']) for row in pose]) for pose in poses]
    # Common torso frame is the reference anatomical coordinate system, including
    # translations/rotations of the entire assembly between separately produced files.
    common=n[20]@np.linalg.inv(o[20]);relative=np.linalg.inv(n)@common@o
    wanted={};all_rows=[]
    for s in p['surfaces']:
        terms={(i,b) for row in s['vertices'] for i,b,v,c in row}
        all_rows.append((s['stable_id'],terms))
        if s['stable_id'] in ids:
            for i,b in terms:wanted.setdefault(i,set()).add(b)
    if any(sid not in ids and any(b in wanted.get(i,set()) for i,b in terms) for sid,terms in all_rows):raise ValueError('Other surfaces share these material donors')
    if p['surfaces']!=op['surfaces']:raise ValueError('Historical stencils have changed')
    def reframe(src,old,selected,dest):
        raw=src.read_bytes();parent=old.read_bytes()
        layout=lambda t:struct.unpack_from('<8s6I32s',t)
        x,y=layout(raw),layout(parent)
        if x[:6]!=y[:6] or x[0]!=b'NHTISS4\0' or x[1]!=5:raise ValueError('Carrier layout changed')
        N,B=x[2:4];off=64+32*N
        if raw[64:off]!=parent[64:off] or raw[off+36*B:]!=parent[off+36*B:]:raise ValueError('Historical back vertices, influences or topology changed')
        rr=np.frombuffer(raw,'<u4',8*N,64).reshape(-1,8).astype(int)
        selected_rows=set();other_rows=set()
        for first,nb,_,_,_,_,sid,_ in rr:
            (selected_rows if sid in selected else other_rows).update(range(first,first+nb))
        if selected_rows&other_rows or not selected_rows:raise ValueError('Ambiguous source row')
        subprocess.run(['/bin/cp','-c',str(src),str(dest)],check=True)
        maxerr=0.;touched=0;owners=set()
        with dest.open('r+b') as f:
            for ix in sorted(selected_rows):
                at=off+36*ix;body,*a=struct.unpack_from('<I8f',parent,at)
                if struct.unpack_from('<I',raw,at)[0]!=body:raise ValueError('Different body identity')
                if np.max(abs(relative[body]-np.eye(4)))<1e-12:target=parent[at:at+36]
                else:
                    t=relative[body];rot=(Rotation.from_matrix(t[:3,:3])*Rotation.from_quat(a[3:7])).as_quat()
                    target=struct.pack('<I8f',body,*(t[:3,:3]@a[:3]+t[:3,3]),*rot,a[7])
                if target!=raw[at:at+36]:
                    f.seek(at);f.write(target);touched+=1;owners.add(int(body))
        A=records(old,[v['rest'] for v in poses[0]],selected)
        C=records(dest,[v['rest'] for v in poses[1]],selected)
        for sid in selected:
            error=np.linalg.norm(A[sid][0]@common[:3,:3].T+common[:3,3]-C[sid][0],axis=1)
            maxerr=max(maxerr,float(error.max()))
        if maxerr>2e-6:raise ValueError('Transport changed certified rest geometry')
        return dict(parent=digest(src),historical=digest(old),registered=digest(dest),changed_binding_records=touched,affected_bodies=sorted(owners),stable_ids=sorted(selected),max_historical_geometry_delta_m=maxerr)
    newchild=out/child.name;sources['child']=reframe(child,oldchild,ids,newchild)
    p['child_payload_sha256']=digest(newchild)
    for i,selected in sorted(wanted.items()):
        existing=p['source_payloads'][i];historical=op['source_payloads'][i];src=Path(existing['path']);old=Path(historical['path']);dst=out/src.name
        if digest(src)!=existing['sha256'] or digest(old)!=historical['sha256']:raise ValueError('Historical donor differs')
        sources[str(i)]=reframe(src,old,selected,dst);existing.update(path=str(dst),sha256=digest(dst))
    receipt=out/'resting-anatomy-receipt.json';receipt.write_text(json.dumps(doc,separators=(',',':'),allow_nan=False)+'\n')
    for row in doc['provenance']['cardiac_geometry_binding']['common_field'].values():
        if isinstance(row,dict) and 'path' in row:
            path=Path(row['path'])
            if path.is_absolute() or '..' in path.parts or digest(base/path)!=row['sha256']:raise ValueError('Cardiac sidecar mismatch')
            os.link(base/path,out/path);cfg['pins'][str(out/path)]=row['sha256']
    a=cfg['argv'];a[a.index('--soft-tissue-payload')+1]=str(newchild);a[a.index('--resting-anatomy-receipt')+1]=str(receipt);a[4]=str(out/'native-traces');cfg['pins'][str(newchild)]=digest(newchild);cfg['pins'][str(receipt)]=digest(receipt)
    env={k:v.replace(str(base)+'/',str(out)+'/') for k,v in cfg['environment'].items()};cfg['environment']=env;cfg['parent_launch']=digest(base/'launch.json')
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    # Numeric preflight uses the retained exact rigid frames. Remove the staged copy
    # before the executable emits its own fresh readback under the same name.
    snapshot=out/'registered-initial-bodies.json';os.link(base/'registered-initial-bodies.json',snapshot)
    try:audit(out,out/'rest-check')
    finally:snapshot.unlink()
    stats=rd(out/'rest-check/result.json')['surfaces'];fail=sum(z['failing'] for z in stats.values())
    result=dict(binding=sources,historical_launch=digest(historic/'launch.json'),rest_failures=fail,worst_rest_error_um=max(v['worst_um'] for v in stats.values()),source_mesh_edit_count=0)
    (out/'frame-registration.json').write_text(json.dumps(result,indent=2)+'\n')
    if fail:raise ValueError('Source-parent registration fails in the current body')
    print(execute_native(a,cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,out,minimum_free_bytes=4*2**30,timeout_s=400),flush=True)
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))

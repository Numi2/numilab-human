"""Replay identical material bindings at recorded poses and compare both ends to native pack coordinates."""
import csv,json,mmap,runpy,struct,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from bilateral_bindings import decode
from compare_hand_interfaces import query
from material_context import sha
from preserve_muscle_cells import det
from audit_gluteal_neighbors import inspect_native_neighbors
from condition_motion_weights import quality_denom

def read_glutes(pack,rows,script):
    frame={r['body_index']:r for r in rows}
    audit=runpy.run_path(str(script/'audit.py'))
    header=audit['clearance']._PACK_HEADER;directory=audit['clearance']._PACK_DIRECTORY
    with pack.open('rb') as fh,mmap.mmap(fh.fileno(),0,access=mmap.ACCESS_READ) as data:
        magic,abi,count,*_=header.unpack_from(data)
        assert (magic,abi)==(b'MRVPACK2',2)
        desc=[directory.unpack_from(data,header.size+j*directory.size) for j in range(count)]
        sections={d[0]:d for d in desc};verts,indices,prims,instances=[sections[j] for j in range(2,6)]
        xyz=np.ndarray((verts[4],20),'<f4',data,verts[2])[:,:3]
        idx=np.ndarray(indices[4],'<u4',data,indices[2]);result={}
        for i in range(prims[4]):
            first,n,_,inst,sem,_,_,sid=struct.unpack_from('<8I',data,prims[2]+i*prims[5])
            if sem!=51005 or sid not in (29,30):continue
            tr=struct.unpack_from('<8f12I',data,instances[2]+inst*instances[5])
            faces=idx[first:first+n].reshape(-1,3);used=np.unique(faces)
            v=audit['world_positions'](xyz[used],tr)
            if tr[10]==3:
                pose=frame[tr[9]];v=audit['world_positions'](v,(*pose['position_m'],1.,*pose['quaternion_xyzw']))
            elif tr[10]!=0:raise ValueError('Unresolved instance kind')
            result[sid]=(v.copy(),np.searchsorted(used,faces))
        if set(result)!={29,30}:raise ValueError('Both native muscles required')
        return result
def run(config,native,trace,volume,script,exe,out,step):
    args=json.loads(config.read_text())['argv'];asset=Path(args[args.index('--soft-tissue-payload')+1])
    directory=native/'accepted-geometry';r=directory/('step-%d.receipt.json'%step);receipt=json.loads(r.read_text())
    pack=native/'resting-human.mrvpack';end=Path(receipt['accepted_pack_path'])
    if sha(pack)!=receipt['base_pack_file_sha256'] or sha(end)!=receipt['pack_file_sha256']:raise ValueError('Unverified capture')
    erows=receipt['accepted_body_poses'];irows=receipt['initial_anatomical_registration']['body_poses']
    cap0=read_glutes(pack,irows,script);cap1=read_glutes(end,erows,script)
    out.mkdir(exist_ok=True)
    for label,banks in [('entrance',cap0),('accepted',cap1)]:
        for sid,(v,f) in banks.items():np.savez_compressed(out/('%s-%d.npz'%(label,sid)),xyz=v,faces=f)
    data=decode(asset,[29,30],np.tile([0,0,0,0,0,0,1.],(157,1)))
    bodies={128,131,145};times={};by={}
    for row in csv.DictReader(trace.open()):
        b=int(row['body_index'])
        if b not in bodies:continue
        t=int(row['accepted_step']);p=[float(row['root_'+d+'_m'])+float(row['com_relative_root_'+d+'_m']) for d in 'xyz'];q=[float(row['orientation_'+d]) for d in 'xyzw']
        if b in by.setdefault(t,{}):raise ValueError('Repeated accepted frame')
        by[t][b]=np.array(p+q);times[t]=float(row['time_s'])
    if not by or max(by)!=step or any(set(b)!=bodies for b in by.values()):raise ValueError('Incomplete donor trace')
    frames={r['body_index']:np.array(r['position_m']+r['quaternion_xyzw']) for r in irows}
    frames_end={r['body_index']:np.array(r['position_m']+r['quaternion_xyzw']) for r in erows}
    def world(sid,pose):
        src,local,weights,donors,f=data[sid];x=np.zeros_like(src,dtype=float)
        for k,b in enumerate(donors):
            p=pose[b];x+=weights[:,k,None]*(p[:3]+Rotation.from_quat(p[3:]).apply(local[:,k]))
        return x.astype('f4')
    def maximum(x):return float(np.linalg.norm(x,axis=1).max())
    start_errors={s:maximum(world(s,frames)-cap0[s][0]) for s in (29,30)}
    pose_errors={s:maximum(world(s,frames_end)-cap1[s][0]) for s in (29,30)}
    # Some owners log the articulated COM frame after their donor debit. The
    # fixed local correction is measured from the actual last body snapshot.
    correction={b:Rotation.from_quat(by[step][b][3:]).inv().apply(frames_end[b][:3]-by[step][b][:3]) for b in bodies}
    for bb in by.values():
        for b,p in bb.items():p[:3]+=Rotation.from_quat(p[3:]).apply(correction[b])
    trace_errors={s:maximum(world(s,by[step])-cap1[s][0]) for s in (29,30)}
    parity=max(list(start_errors.values())+list(pose_errors.values())+list(trace_errors.values()))
    banks={};used={}
    for s in (29,30):
        t=np.loadtxt(volume/str(s)/'source.1.ele',dtype=int,skiprows=1)[:,1:5]
        x=np.loadtxt(volume/str(s)/'source.1.node',skiprows=1)[:,1:4]
        if not np.array_equal(cap0[s][1],data[s][-1]) or not np.array_equal(cap1[s][1],data[s][-1]):raise ValueError('Native topology differs')
        banks[s]=(t,det(x,t),det(cap0[s][0].astype(float),t));used[s]=cap0[s][1]
    allfaces=np.vstack((used[29],used[30]+len(cap0[29][0])))
    rows=[]
    def inspect(k,pos):
        xx=np.vstack([pos[29],pos[30]]);pairs=query(exe,xx,allfaces,np.ones(len(allfaces),bool))
        n=len(used[29]);cell={s:float((det(pos[s].astype(float),banks[s][0])/banks[s][1]).min()) for s in (29,30)}
        ratio={s:float((det(pos[s].astype(float),banks[s][0])/banks[s][2]).min()) for s in (29,30)}
        quality={s:float((det(pos[s].astype(float),banks[s][0])/quality_denom(pos[s].astype(float),banks[s][0])).min()) for s in (29,30)}
        return dict(step=k,minimum_relative_to_registered_entrance=min(ratio.values()),minimum_normalized_determinant=min(quality.values()),bilateral=sum(i<n<=j for i,j in pairs),self_right=sum(j<n for i,j in pairs),self_left=sum(i>=n for i,j in pairs),minimum_volume_ratio=min(cell.values()))
    observed=[inspect(0,{s:x[0] for s,x in cap0.items()}),inspect(step,{s:x[0] for s,x in cap1.items()})]
    if parity<=2e-6:
        for k,p in sorted(by.items()):rows.append(inspect(k,{s:world(s,p) for s in (29,30)}))
    neighbors={'entrance':inspect_native_neighbors(pack,irows,script,exe),'accepted':inspect_native_neighbors(end,erows,script,exe)}
    (out/'neighbor-counts.json').write_text(json.dumps(neighbors,indent=2)+'\n')
    result=dict(source_payload_sha256=sha(asset),receipt_sha256=sha(r),initial_pack_sha256=sha(pack),accepted_pack_sha256=sha(end),trace_sha256=sha(trace),accepted_time_s=times[step],trace_samples=len(by),
        maximum_entrance_vertex_error_m=start_errors,maximum_accepted_vertex_error_m=pose_errors,maximum_trace_endpoint_vertex_error_m=trace_errors,frame_correction_local_m={k:u.tolist() for k,u in correction.items()},
        recorded_pose_replay_qualified=parity<=2e-6,observed_native_packs=observed,pose_samples=rows)
    (out/'motion.json').write_text(json.dumps(result,indent=2)+'\n')
    print('maximum endpoint registration error',parity,'sampled poses',len(rows),'captured',observed,flush=True)
if __name__=='__main__':run(*map(Path,sys.argv[1:-1]),int(sys.argv[-1]))

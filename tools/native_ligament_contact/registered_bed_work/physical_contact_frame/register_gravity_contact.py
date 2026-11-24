"""Place the existing fixed plane within a gravity closure distance of its actual unilateral rows."""
import itertools,json,shutil,struct,subprocess,re
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from update_witnesses import reproject

def run(root):
    out=root/'gravity-contact';out.mkdir(exist_ok=True);test=out/'test';test.mkdir(exist_ok=True)
    g=json.loads((root/'geometry.json').read_bytes());pid=np.asarray(g['contacts']);vol=root.parent/'volume'
    x=np.loadtxt(root.parent/'cortical-fit/current.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype('f8')
    outer=np.load(root.parent.parent/'registered-boundary/registered-boundary.npz')['O_node_joint']
    theta=np.array(g['normal']);z=x[pid];gravity=9.81;dt=7.8125e-6;eps=np.finfo('f4').eps
    width=1e-5;guard=float(np.float32(1e-4)*np.float32(width));h=g['height']
    # Solve for three geometrically reachable unilateral rows. No impulses are prescribed.
    A=np.c_[z[:,:2],-np.ones(3)]
    for _ in range(8):
        scale=np.abs(z)@np.abs(theta)+abs(theta[2]*h)
        floor=np.maximum(guard,16*eps*scale);target=floor+16*eps*scale
        d=target+.5*gravity*dt*dt*theta[2]
        sx,sy,h=np.linalg.solve(A,d/theta[2]-z[:,2]);theta=np.array([sx,sy,1.]);theta/=np.linalg.norm(theta)
    def neighbors(x,k):
        y=np.float32(x);a=b=y;v=[y]
        for _ in range(k):
            a=np.nextafter(a,np.float32(-np.inf));b=np.nextafter(b,np.float32(np.inf));v.extend([a,b])
        return v
    choices=[(np.asarray(g['normal'],'f4'),np.float32(g['height']))]
    for u,v,hh in itertools.product(neighbors(theta[0],5),neighbors(theta[1],5),neighbors(h,5)):
        normal=np.array([u,v,np.sqrt(1-float(u)**2-float(v)**2)],'f4')
        choices.append((normal,hh))
    nodeMass=np.fromfile(vol/'node-mass.f64','<f8')
    if len(nodeMass)!=len(x) or not (nodeMass[pid]>0).all():raise ValueError('Invalid registered node masses')
    program='''#include "matter_kernels.metal"
namespace numi_matter_metal {
kernel void contactFrameTest(device const float4* input [[buffer(0)]],device float4* output [[buffer(1)]],uint id [[thread_position_in_grid]]) {
    const auto r=input+6*id;const float3 p=r[1].xyz;const float dt=r[2].x;
    NMRigidStateGPU rigid={};rigid.centerAndRadius=r[0];float sep;float3 n,w;
    if(!planeContact(rigid,p,sep,n,w)){output[3*id]=float4(NAN);return;}
    const float scale=nmFixedPlaneContactArithmeticScale(n.x,n.y,n.z,p.x,p.y,p.z,rigid.centerAndRadius.w,r[2].y);
    const float floor=registeredPlaneCertificateFloor(n,p,rigid.centerAndRadius.w,r[2].y,r[2].z);
    const float margin=ccdDistanceRoundoff(floor,scale);const float target=floor+2.f*margin;
    const auto frame=registeredBedFrame(n);
    const auto rhs=transpose(frame)*float3(0,0,-r[1].w*r[2].w*dt);
    const auto row=projectBedMomentum(rhs,float3(0),r[1].w,sep-target,dt,float2(0));
    output[3*id]=float4(sep,floor,target,row.impulse.z/dt);
}}'''
    (test/'check.metal').write_text(program)
    src=root/'entry/metal/src';cmds=json.loads((root/'entry/build.json').read_bytes())['commands']
    cc=cmds[0][:cmds[0].index('-c')]+['-I',str(src),'-c','check.metal','-o','check.air']
    host=(root/'execute_frame_test.mm').read_text().replace('[lib newFunctionWithName:@"contactFrameTest"]','[lib newFunctionWithName:@"numi_matter_metal::contactFrameTest" constantValues:[MTLFunctionConstantValues new] error:&error]')
    (test/'run.mm').write_text(host)
    with (test/'build.log').open('w')as log:
        for cm in [cc,['xcrun','-sdk','macosx','metallib','check.air','-o','check.metallib'],['clang++','-std=c++17','-fobjc-arc','-framework','Metal','-framework','Foundation','run.mm','-o','check']]:subprocess.run(cm,cwd=test,stdout=log,stderr=subprocess.STDOUT,check=True)
    data=np.zeros((len(choices),3,6,4),'f4')
    for i,(n,hh) in enumerate(choices):
        # The source compiler's fixed plane stores the normalized normal and its offset.
        active=n.astype(float);active/=np.linalg.norm(active)
        data[i,:,0]=[*active,np.float32(float(hh)*active[2])]
    data[:,:,1,:3]=z;data[:,:,1,3]=nodeMass[pid];data[:,:,2]=[dt,width,guard,gravity]
    data.tofile(test/'input.bin');subprocess.run(['./check',str(len(choices)*3)],cwd=test,check=True)
    rows=np.fromfile(test/'out.bin','f4').reshape(len(choices),3,12)[:,:,:4].astype(float)
    if not np.isfinite(rows).all():raise ValueError('Invalid Metal outputs')
    effective=rows[:,:,0]-rows[:,:,2];closure=gravity*dt*dt*theta[2]
    valid=np.flatnonzero((effective>=0).all(1)&(effective<closure).all(1)&(rows[:,:,3]>0).all(1))
    if not len(valid):raise ValueError('No represented, gravity-reachable plane')
    k=int(valid[np.argmin(np.max(abs(effective[valid]/closure-.5),axis=1))]);n,hh=choices[k];n=n.astype(float);hh=float(hh)
    gap=x[:,0]*n[0]+x[:,1]*n[1]+(x[:,2]-hh)*n[2]
    if (gap<=0).any() or sorted(outer[gap[outer]<width])!=pid.tolist():raise ValueError('Changed contact identities')
    report=dict(device=(test/'device.txt').read_text(),placement_candidates=len(choices)-1,gravity_reachable_candidates=len(valid),selected=k,node_ids=pid.tolist(),mass_kg=nodeMass[pid].tolist(),root_seconds=dt,gravity=gravity,normal=n.tolist(),height_m=hh,minimum_gap_m=float(gap.min()),gravity_closure_m=closure,columns=['separation_m','certificate_floor_m','unilateral_target_m','gravity_only_normal_force_N'],original=rows[0].tolist(),placed=rows[k].tolist(),sample_scope='Source nodal mass and gravity only. Complete assembled residual and certified physical bed reactions require native motion.',registered_nodes=g['source_nodes'],original_bed=g['bed'])
    dataInc=(root/'app/NumiHumanLoadedSkinPoseData.inc').read_text()
    for name,value in [('minimumBedGap',float(gap.min()).hex()),('fixedBedHeight',hh.hex()),('bedNormal[]','{'+','.join(v.hex() for v in n)+'}')]:
        dataInc,count=re.subn(re.escape(name)+r'\s*=\s*[^;]*;',lambda m:name+' = '+value+';',dataInc)
        if count!=1:raise ValueError('Ambiguous registered constant '+name)
    app=out/'app';app.mkdir(exist_ok=True)
    for f in (root/'app').iterdir():
        dest=app/f.name
        if f.name=='NumiHumanLoadedSkinPoseData.inc':dest.write_text(dataInc)
        elif f.name in ['NumiHumanLoadedSkinPose.hpp','numilab_human_myosim_visual_probe.mm']:shutil.copy2(f,dest)
        elif f.suffix in ['.hpp','.inc','.h'] and not dest.exists():dest.symlink_to(f.resolve())
    control=json.loads((root.parent/'incident64-scene/launch.json').read_bytes());args=control['argv'];old=Path(args[args.index('--support-contact-payload')+1])
    if digest(old)!=g['source_bed']:raise ValueError('Mismatched original observation frame')
    packet=bytearray(old.read_bytes());struct.pack_into('<6f',packet,56,0,0,hh,*n);reproject(packet,hh,n);(out/'registered-bed.nhcnt').write_bytes(packet)
    report['bed']=digest(out/'registered-bed.nhcnt');(out/'geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    cc=[v.replace(str(root/'app')+'/',str(app)+'/')for v in json.loads((root/'build.json').read_bytes())['commands'][0]]
    link=json.loads((root/'tile-refresh/build.json').read_bytes())['commands'][3]
    ld=[{str(root/'app/numi-human-native.o'):str(app/'numi-human-native.o'),str(root/'tile-refresh/app/numi-human-native'):str(app/'numi-human-native')}.get(v,v) for v in link]
    if str(app/'numi-human-native.o') not in ld:raise ValueError('Wrong static owner')
    with (out/'build.log').open('w')as log:
        for cm in [cc,ld]:subprocess.run(cm,stdout=log,stderr=subprocess.STDOUT,check=True)
    m=dict(app=digest(app/'numi-human-native'),data=digest(app/'NumiHumanLoadedSkinPoseData.inc'),commands=[cc,ld],geometry=digest(out/'geometry.json'))
    (out/'build.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':run(Path(__file__).resolve().parent)

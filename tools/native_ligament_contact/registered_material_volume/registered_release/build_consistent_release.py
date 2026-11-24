"""Use a common geometric volume predicate in trial selection, sparse direction repair and final certification."""
from pathlib import Path
import json,shutil,subprocess,sys
from build_bed_audit import replace_once,sha

def build(parent,out):
    config=json.loads((parent/'build.json').read_bytes());orig=parent/'src';out.mkdir(exist_ok=True);dest=out/'src';(dest/'metal').mkdir(parents=True,exist_ok=True)
    for s in orig.iterdir():
        to=dest/s.name
        if s.name=='matter_kernels.metal':shutil.copyfile(s,to)
        elif s.name!='metal' and not to.exists():to.symlink_to(s.resolve())
    for s in (orig/'metal').iterdir():
        to=dest/'metal'/s.name
        if s.name!='fem.metalinc' and not to.exists():to.symlink_to(s.resolve())
    src=(orig/'metal/fem.metalinc').read_text();lo=src.index('inline MRCompensatedScalar registeredNeg(');hi=src.index('// One environment is admitted',lo)
    helpers=src[lo:hi];src=src[:lo]+src[hi:]
    helpers=helpers.replace('registeredVolumeWord(device const NMFEMNodeStateGPU* nodes,uint4 ids)','registeredStateVolumeWord(thread const NMFEMNodeStateGPU* nodes)').replace('nodes[ids[0]]','nodes[0]').replace('nodes[ids[i]]','nodes[i]')
    helpers+='''
inline MRCompensatedScalar registeredVolumeWord(device const NMFEMNodeStateGPU* nodes,uint4 ids){
    NMFEMNodeStateGPU p[4];
    for(uint k=0;k<4;++k)p[k]=nodes[ids[k]];
    return registeredStateVolumeWord(p);
}
inline MRCompensatedScalar registeredReleaseMargin(MRCompensatedScalar volume,float4 bound){
    return mrCompensatedAdd(volume,{-bound.x,-bound.y});
}
inline bool registeredReleasePass(MRCompensatedScalar margin,float4 bound){
    return isfinite(margin.high) && margin.high*bound.w>0.f;
}
'''
    marker='kernel void nm_fem_parallel_backtracking_tiles('
    src=replace_once(src,marker,helpers+'\n'+marker)
    # Trial validity must describe the exact represented positions that would be committed.
    start=src.index(marker);end=src.index('kernel void nm_fem_parallel_backtracking_finalize(',start)
    c=src[start:end];c=replace_once(c,'            float3 trialVelocity[4];','            float3 trialVelocity[4];\n            NMFEMNodeStateGPU releaseTrial[4];')
    c=replace_once(c,'                x[slot] = trialNode.positionAndMass.xyz;','                if(useReleaseBounds==2u) releaseTrial[slot]=trialNode;\n                x[slot] = trialNode.positionAndMass.xyz;')
    c=replace_once(c,'            laneValid = tetrahedronReason == 0u;','''            if(!tetrahedronReason && useReleaseBounds==2u){
                const auto bound=releaseFloors[object.elementOffset+localElement];
                if(!registeredReleasePass(registeredReleaseMargin(registeredStateVolumeWord(releaseTrial),bound),bound))
                    tetrahedronReason=nmFEMBacktrackingFailureReason(0.f,lowerJ,material.validity.y,true);
            }
            laneValid = tetrahedronReason == 0u;''')
    src=src[:start]+c+src[end:]
    marker='inline uint registeredVolumeCellDirection('
    sparse='''
inline NMRegisteredVolumeControl registeredTrialControl(float3x3 start, float3x3 end, float lo,float hi,
        thread const NMFEMNodeStateGPU* nodes, device const float4* bounds, uint index,uint mode){
    const auto original=registeredSweptControl(start,end,lo,hi);
    if(mode!=2u || !(original.value>lo && original.value<hi))return original;
    const float4 bound=bounds[index];
    const auto margin=registeredReleaseMargin(registeredStateVolumeWord(nodes),bound);
    if(registeredReleasePass(margin,bound))return original;
    // Express the geometric shortfall in the same material-J units as the
    // search gradient. Recompute the geometric predicate on the updated nodes.
    const float relative=margin.high/bound.x;
    return {isfinite(relative)?min(lo,fma(relative,bound.z,bound.z)):-INFINITY,3u};
}
'''
    src=replace_once(src,marker,sparse+marker)
    start=src.index(marker);end=src.index('kernel void nm_fem_registered_volume_active_incident64(',start);c=src[start:end]
    c=replace_once(c,'registeredSweptControl(startF,F,lo,hi)','registeredTrialControl(startF,F,lo,hi,p,releaseFloors,rows[row],releaseMode)')
    c=replace_once(c,'registeredSweptControl(startF,registeredTrialF(p,inv,reference),lo,hi)','registeredTrialControl(startF,registeredTrialF(p,inv,reference),lo,hi,p,releaseFloors,rows[row],releaseMode)')
    c=replace_once(c,'registeredSweptControl(registeredTrialF(base,inv,reference),\n            registeredTrialF(trial,inv,reference),lo,hi)','registeredTrialControl(registeredTrialF(base,inv,reference),\n            registeredTrialF(trial,inv,reference),lo,hi,trial,releaseFloors,e,releaseMode)')
    src=src[:start]+c+src[end:]
    marker='kernel void nm_fem_registered_release_bound_elements(';start=src.index(marker);end=src.index('kernel void nm_fem_registered_release_bound_finish(',start);c=src[start:end]
    c=replace_once(c,'if(!nmFEMBacktrackingFailureReason(J,lo,mat.validity.y,true) &&','if(!nmFEMBacktrackingFailureReason(J,lo,mat.validity.y,true) &&\n           registeredReleasePass(registeredReleaseMargin(registeredStateVolumeWord(p),floors[e]),floors[e]) &&')
    src=src[:start]+c+src[end:]
    src=replace_once(src,'const auto margin=mrCompensatedAdd(det,{-b.x,-b.y});\n    if(!(isfinite(margin.high) && margin.high*b.w>0.f))','const auto margin=registeredReleaseMargin(det,b);\n    if(!registeredReleasePass(margin,b))')
    (dest/'metal/fem.metalinc').write_text(src)
    commands=[]
    for cmd in config['commands']:
        cmd=[s.replace(str(parent)+'/',str(out)+'/')for s in cmd]
        with (out/f'compile-{len(commands)}.log').open('w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        commands.append(cmd)
    (out/'build.json').write_text(json.dumps(dict(commands=commands,original_metallib=sha(parent/'NumiMatter.metallib'),metallib=sha(out/'NumiMatter.metallib'),original_fem=sha(orig/'metal/fem.metalinc'),fem=sha(dest/'metal/fem.metalinc')),indent=2)+'\n')
if __name__=='__main__':build(*(Path(v).resolve()for v in sys.argv[1:]))

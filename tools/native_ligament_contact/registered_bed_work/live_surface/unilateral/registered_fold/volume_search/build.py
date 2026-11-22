"""Compile a sparse volume-feasible Newton search into the same qualified native owner."""
import json
from pathlib import Path
import subprocess
import sys
import re
import numpy as np
from fit_dorsal_material import inputs
from material_context import sha
from supported_material_admission import installed_topology

def read(p):return json.loads(p.read_text())
def write(p,data):p.write_text(json.dumps(data,indent=2)+'\n')

def build(root):
    exp=root/'material-direction';previous=root/'fold-repair'
    baseline=root/'feasible-bed/build';out=exp/'build';out.mkdir(exist_ok=True)
    release=previous/'accepted-2';failed=previous/'three'
    candidate=read(failed/'launch.json')
    if candidate['pins'][str(baseline/'NumiMatter.metallib')]!=sha(baseline/'NumiMatter.metallib'):
        raise ValueError('Different solver at failed release')
    if not read(release/'terminal.receipt.json')['accepted_step']==read(failed/'failure.json')['segment_start']:
        raise ValueError('Unmatched witness time')
    log=(failed/'stderr.log').read_text()
    samples=[json.loads(l.split('=',1)[1]) for l in log.splitlines() if l.startswith('fem_backtracking_trials=')]
    record=samples[-1]
    if record['root']!=read(failed/'failure.json')['segment_start']:raise ValueError('Different failed root')
    full,=[t for t in record['trials'] if t['alpha']==1]
    if not full['tile_witnesses_complete'] or not full['tile_witness_tets']:raise ValueError('Missing native material witnesses')
    _,_,tets,roles,pins=inputs(release);tets,roles=installed_topology(tets,roles,candidate['environment'])
    text=(failed/'stdout.log').read_text()
    def matches(pattern,n):
        val=re.findall(pattern,text,re.M)
        if len(val)!=n:raise ValueError(pattern)
        return np.asarray(val,dtype=int)
    fascia=matches(r'^resting_fascia_object=source_prepared object=\d+ .*?nodes=(\d+) tetrahedra=(\d+)',12)
    acl=matches(r'^resting_inferred_acl=admitted .*?nodes=(\d+) tetrahedra=(\d+)',2)
    outer,faces=matches(r'^resting_aortic_wall_outer_contact=accepted .*?outer_ring_rigid_query_nodes=(\d+) .*?retained_wall_faces=(\d+)',1)[0]
    node,element=fascia.sum(0)+acl.sum(0)+[2*outer,3*faces]
    if node!=matches(r'\bfirst_global_outer_fem_node=(\d+)\b',1)[0]:raise ValueError('Different native global node inventory')
    witness=[]
    for row in sorted(set(full['tile_witness_tets'])):
        local=row-element
        if not 0<=local<len(tets) or roles[local]!=16:raise ValueError('Expected registered adipose')
        witness.append((row,[int(i) for i in tets[local]+node]))
    # Separate overlapping stars into sequential colors to avoid write races.
    colors=[]
    for row,ids in witness:
        for entries,used in colors:
            if used.isdisjoint(ids):
                used.update(ids);entries.append(row);break
        else:colors.append(([row],set(ids)))
    if len(witness)>256:raise ValueError('Experimental sparse native argument extent exceeded')
    if not (out/'src').exists():subprocess.run(['cp','-cR',baseline/'src',out/'src'],check=True)
    marker='} // namespace numi_matter_metal'
    source=(baseline/'src/metal/fem.metalinc').read_text()
    if not source.rstrip().endswith(marker):raise ValueError('Unknown Metal namespace')
    dst=out/'src/metal/fem.metalinc';dst.unlink(missing_ok=True)
    dst.write_text(source[:source.rindex(marker)]+(exp/'direction.metalinc').read_text()+'\n'+marker+'\n')
    def change(code,site,new):
        if code.count(site)!=1:raise ValueError('Unknown native assembly site '+site[:55])
        return code.replace(site,new)
    check=[]
    for row,(i,j,k,l) in witness:
        check.append(f'''if(world.fem.tetrahedra.size()<={row}u) return false;
                {{const auto n=world.fem.tetrahedra[{row}u].nodes;
                if(n.x!={i}u||n.y!={j}u||n.z!={k}u||n.w!={l}u)return false;}}''')
    bindings=['setDispatch();','[encoder setBytes:&micro length:sizeof(micro) atIndex:1u];']
    for ix,field in enumerate(('objects','materials','femAccepted','femCandidate','femSolution','femTetrahedraCandidate','schedulers','adaptive','fgmresStates'),2):
        bindings.append(f'[encoder setBuffer:state.{field} offset:0u atIndex:{ix}u];')
    batches=[]
    for group,_ in colors:
        rows=','.join(str(r)+'u' for r in group)
        batches.append('''{
                    static constexpr std::uint32_t rows[]={'''+rows+'''};
                    const std::uint32_t count=sizeof(rows)/sizeof(rows[0]);
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    dispatchThreads("nm_fem_registered_volume_direction",count,[&]{
                        '''+'\n                        '.join(bindings)+'''
                        [encoder setBytes:rows length:sizeof(rows) atIndex:11u];
                        [encoder setBytes:&count length:sizeof(count) atIndex:12u];
                    });
                }''')
    def patch(s):
        field='    bool unilateralBed = false;';s=change(s,field,'    bool volumeDirection = false;\n'+field)
        site='        candidate->acceptedStateProofProgramFingerprint ='
        s=change(s,site,'''        if(const char* enabled=std::getenv("NM_FEM_REGISTERED_VOLUME_DIRECTION")){
            if(!std::strcmp(enabled,"1")){
                const bool matches=[&]{
                    '''+'\n                    '.join(check)+'''
                    return true;}();
                if(!candidate->unilateralBed || !matches){
                    diagnostics.message="volume-direction proposal requires its measured registered tissue topology and bed";return diagnostics;
                }
                candidate->volumeDirection=true;
                candidate->executionFingerprint=mixFingerprint(candidate->executionFingerprint,'''+str(int(sha(exp/'direction.metalinc')[:16],16))+'''ull);
            }else if(std::strcmp(enabled,"0")){diagnostics.message="invalid volume-direction mode";return diagnostics;}
        }
'''+site)
        s=change(s,'            "nm_fem_registered_bed_direction",','            "nm_fem_registered_volume_direction",\n            "nm_fem_registered_bed_direction",')
        site='        for (const char* name : kernelNames) {'
        s=change(s,site,site+'\n            if(!candidate->volumeDirection && !std::strcmp(name,"nm_fem_registered_volume_direction"))continue;')
        site='                const std::uint32_t collapseCompletedBacktracking ='
        return change(s,site,'                if(state.volumeDirection){\n'+'\n'.join(batches)+'\n                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];\n                }\n'+site)
    for name in ('runtime.mm','app-runtime.mm'):
        (out/name).write_text(patch((baseline/name).read_text()))
    cmds=[]
    def run(cmd,**kwargs):
        cmds.append(cmd);subprocess.run(cmd,check=True,**kwargs)
    for cmd in read(baseline/'build-pins.json')['commands'][:5]:
        run([x.replace(str(baseline)+'/',str(out)+'/') for x in cmd])
    parent=read(previous/'parent/launch.json');old=previous/'owner';new=exp/'owner'
    if not new.exists():subprocess.run(['cp','-cR',old,new],check=True)
    meta=read(old/'identities.json');cmd=[x.replace(str(old)+'/',str(new)+'/').replace(str(baseline)+'/',str(out)+'/') for x in meta['commands'][1]]
    run(cmd,cwd=parent['cwd'])
    binary=sha(new/'numi-human-native')
    meta.update(binary=binary,binary_sha256=binary,binary_path=str(new/'numi-human-native'))
    meta['commands'][1]=cmd;write(new/'identities.json',meta)
    env=parent['environment'];env['NM_FEM_REGISTERED_VOLUME_DIRECTION']='1'
    for name,oldpath,newpath in [('NUMI_MATTER_METALLIB_OVERRIDE',baseline/'NumiMatter.metallib',out/'NumiMatter.metallib'),('DYLD_LIBRARY_PATH',baseline/'libmetalrobo.dylib',out/'libmetalrobo.dylib')]:
        if parent['pins'].pop(str(oldpath))!=sha(oldpath):raise ValueError('Unmatched binary inputs')
        env[name]=str(newpath if name!='DYLD_LIBRARY_PATH' else newpath.parent)
        parent['pins'][str(newpath)]=sha(newpath)
    parentdir=exp/'parent';parentdir.mkdir(exist_ok=True);write(parentdir/'launch.json',parent)
    write(out/'receipt.json',dict(source_sha256=sha(exp/'direction.metalinc'),witness_source=sha(failed/'stderr.log'),
        selected_from=record['root'],geometry=pins,rows=witness,colors=[g for g,_ in colors],commands=cmds,
        binary=binary,metallib=sha(out/'NumiMatter.metallib'),dylib=sha(out/'libmetalrobo.dylib')))
if __name__=='__main__':build(Path(sys.argv[1]).resolve())

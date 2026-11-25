"""Build a pinned, opt-in material interface into the same Human force owner."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from compile_connected_joint import digest


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one source owner for {old[:75]}')
    return text.replace(old, new)


def literal(x):
    return f'{np.float32(x):.9e}f'


def vectors(name, x):
    n = x.shape[1]
    rows = [f'float{n}('+','.join(map(literal, row))+')' for row in x]
    return f'constant float{n} {name}[] = {{'+',\n'.join(rows)+'};\n'


def build(j, material, link, out):
    case = j/'compensated-start-two'
    parents = np.load(material/'parents.npz')
    local, donors = parents['fat'],parents['donors']
    if len(set(local)) != 51 or list(donors) != [128,131,145]:
        raise ValueError('Unrecognized material ports')
    ref = parents['reference'][:51]
    for name in ('mesh.node','current.node'):
        file = j/'conditioned/witness-material/volume'/name
        if not np.array_equal(np.loadtxt(file,skiprows=1,usecols=(1,2,3))[local], ref):
            raise ValueError('Unmatched material parent frame')
    first = 1666
    log = (case/'stdout.log').read_text()
    if f'first_global_outer_fem_node={first} ' not in log or 'FEM_nodes=321219 ' not in log:
        raise ValueError('Incorrect native domain')
    bodies = {b['body_index']: b['current'] for b in json.loads((case/'native-initial-bodies.json').read_bytes())['all_bodies']}
    origins = np.array([bodies[b]['position_world_m'] for b in donors])
    if np.max(abs(parents['origins'] - origins)) > 1e-8:
        raise ValueError('Unregistered muscle donors')
    orientation = np.array([bodies[b]['orientation_world_xyzw'] for b in donors])
    K = np.load(material/'reference-tangent.npz')['matrix']
    if K.shape != (171,171) or np.max(abs(K-K.T)) > 1e-8 or not np.isfinite(K).all():
        raise ValueError('Invalid condensed tangent')
    K = ((K+K.T)*.5).astype('<f4')
    out.mkdir(exist_ok=True)
    (out/'material.f32').write_bytes(K.tobytes())
    const = 'constant uint gcFat=51u,gcPorts=54u,gcTranslations=153u,gcDofs=171u;\n'
    const += 'constant uint gcNodes[] = {'+','.join(map(str,first+local))+'};\n'
    const += 'constant uint gcBodies[] = {'+','.join(map(str,donors))+'};\n'
    const += vectors('gcPoints',np.vstack([ref,origins]))+vectors('gcQuaternions',orientation)
    parent = j/'exact-release'
    spec = json.loads((parent/'build.json').read_bytes())
    root = out/'src'
    (root/'metal').mkdir(parents=True,exist_ok=True)
    for f in (parent/'src').iterdir():
        if f.name == 'matter_kernels.metal':
            shutil.copy2(f,root/f.name)
        elif f.name!='metal' and not (root/f.name).exists():
            (root/f.name).symlink_to(f.resolve())
    for f in (parent/'src/metal').iterdir():
        if f.name!='numi_human.metalinc' and not (root/'metal'/f.name).exists():
            (root/'metal'/f.name).symlink_to(f.resolve())
    gpu = parent/'src/metal/numi_human.metalinc'
    extra = Path(__file__).with_name('gluteal_interface.metalinc').read_text()
    mark = 'kernel void nm_numi_human_assemble_fem_surface_attachment_loads('
    (root/'metal'/gpu.name).write_text(replace_once(gpu.read_text(),mark,const+extra+'\n'+mark))
    cmds = [[s.replace(str(parent)+'/',str(out)+'/') for s in cmd] for cmd in spec['commands']]
    linkcmd=json.loads(link.read_bytes())['commands'][1]
    oldobj=Path(next(s for s in linkcmd if s.endswith('/numi_human.mm.o')))
    host=oldobj.with_suffix('')
    cpp=host.read_text()
    cpp=replace_once(cpp,'    __strong id<MTLBuffer> externalForceBuffer = nil;', '''    bool gluteal = false;
    __strong id<MTLComputePipelineState> glutealPipeline = nil;
    __strong id<MTLBuffer> glutealK = nil;
    __strong id<MTLBuffer> glutealObservation = nil;
    __strong id<MTLBuffer> externalForceBuffer = nil;''')
    tag=hashlib.sha256(const.encode()+extra.encode()+K.tobytes()).hexdigest()
    mark='    candidate->fingerprint = fingerprint == 0u ? 1u : fingerprint;'
    cpp=replace_once(cpp,mark,'''    if (const char* mode = std::getenv("NUMI_HUMAN_GLUTEAL_OPERATOR")) {
        if (std::strcmp(mode,"TAG") || candidate->environmentCount!=1 ||
            candidate->nodeLoads.size()!=321219 || !candidate->femBodyContactSamplesCount()) return false;
        candidate->gluteal=true;
        fingerprint=appendFingerprint(fingerprint, mode, std::strlen(mode));
    }
'''.replace('TAG',tag)+mark)
    mark='            state_->externalForceBuffer = [device'
    cpp=replace_once(cpp,mark,'''            if (state_->gluteal) {
                if (pass.articulationFirstBody>128u || 145u-pass.articulationFirstBody>=pass.bodyPoseStride ||
                    state_->femBodyContactWrenchBuffer==nil) return false;
                const char* path=std::getenv("NUMI_HUMAN_GLUTEAL_MATRIX");
                NSData* matrix=path ? [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]] : nil;
                if (matrix.length!=171u*171u*sizeof(float)) return false;
                unsigned char hashed[CC_SHA256_DIGEST_LENGTH];
                CC_SHA256(matrix.bytes,static_cast<CC_LONG>(matrix.length),hashed);
                const unsigned char expected[]={BYTES};
                if (std::memcmp(hashed,expected,sizeof(expected))) return false;
                state_->glutealK=[device newBufferWithBytes:matrix.bytes length:matrix.length options:MTLResourceStorageModeShared];
                state_->glutealObservation=[device newBufferWithLength:512u options:MTLResourceStorageModeShared];
                state_->glutealPipeline=pipeline("nm_human_gluteal_interface");
                if (!state_->glutealK || !state_->glutealObservation || !state_->glutealPipeline ||
                    state_->glutealPipeline.maxTotalThreadsPerThreadgroup<256u) return false;
                std::memset(state_->glutealObservation.contents,0,512u);
            }
'''.replace('BYTES',','.join(map(str,hashlib.sha256(K.tobytes()).digest())))+mark)
    mark='        const bool captureRequested ='
    cpp=replace_once(cpp,mark,'''        if (state_->gluteal && !encodeKernel(state_->glutealPipeline,256u,[&](id<MTLComputeCommandEncoder> e) {
            [e setBuffer:acceptedNodes offset:0 atIndex:1];
            [e setBuffer:bodyPoses offset:0 atIndex:2];
            [e setBuffer:state_->externalForceBuffer offset:0 atIndex:3];
            [e setBuffer:state_->femBodyContactWrenchBuffer offset:0 atIndex:4];
            [e setBuffer:state_->glutealK offset:0 atIndex:5];
            [e setBuffer:state_->glutealObservation offset:0 atIndex:6];
        })) return false;
'''+mark)
    mark='NumiHumanTendonFEMLoadAdapter::~NumiHumanTendonFEMLoadAdapter() = default;'
    cpp=replace_once(cpp,mark,'''NumiHumanTendonFEMLoadAdapter::~NumiHumanTendonFEMLoadAdapter() {
    const char* path=std::getenv("NUMI_HUMAN_GLUTEAL_OBSERVATION");
    if (path && state_ && state_->glutealObservation) {
        NSData* result=[NSData dataWithBytes:state_->glutealObservation.contents length:512u];
        [result writeToFile:[NSString stringWithUTF8String:path] atomically:YES];
    }
}''')
    (out/host.name).write_text(cpp)
    compilecmd=json.loads((host.parent/'build.json').read_bytes())['commands'][2]
    compilecmd=[str(out/host.name) if s==str(host) else str(out/oldobj.name) if s==str(oldobj) else s for s in compilecmd]
    compilecmd[1:1]=['-iquote',str(host.parent)]
    linkcmd=[str(out/oldobj.name) if s==str(oldobj) else s for s in linkcmd]
    linkcmd[linkcmd.index('-o')+1]=str(out/'libmetalrobo.dylib')
    cmds.extend([compilecmd,linkcmd])
    for i,cmd in enumerate(cmds):
        with (out/f'build-{i}.log').open('w') as log:
            subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    pins={p.name:digest(p) for p in [gpu,host,material/'parents.npz',material/'reference-tangent.npz',
        out/'material.f32',out/'NumiMatter.metallib',out/'libmetalrobo.dylib']}
    (out/'build.json').write_text(json.dumps(dict(operator=tag,pins=pins,commands=cmds),indent=2)+'\n')

if __name__=='__main__':
    build(*(Path(x).resolve() for x in sys.argv[1:]))

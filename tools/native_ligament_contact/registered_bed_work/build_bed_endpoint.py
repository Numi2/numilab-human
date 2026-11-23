"""Attach the bed material domain certificate to the existing native transaction."""
import json,subprocess,sys
from pathlib import Path
from material_context import sha
from export_contact_columns import change
base,source,owner=map(lambda x:Path(x).resolve(),sys.argv[1:])
root=Path(__file__).resolve().parent
read=lambda f: json.loads(f.read_text())
ids=read(base/'identities.json')
if any(sha(base/k)!=v for k,v in ids.items()):raise ValueError('stale native input')
subprocess.run(['cp','-cR',base,owner],check=True)
def edit(name,s,t):
    f=owner/name;f.write_text(change(f.read_text(),s,t))
name='NumiHumanResolvedBed.metalinc'
f=owner/name;f.write_text(f.read_text()+'\n'+(root/'bed_endpoint.metalinc').read_text())
n='NumiHumanResolvedBed.hpp'
edit(n,'    size_t vertices_,','    id<MTLComputePipelineState> endpoint_, endpointDecision_;\n    id<MTLBuffer> endpointIndex_;\n    size_t vertices_,')
edit(n,'        columns_=pipeline','''        MTLFunctionConstantValues* values=[MTLFunctionConstantValues new];
        bool tiled=stride==32u,validated=true;
        [values setConstantValue:&tiled type:MTLDataTypeBool atIndex:15];
        [values setConstantValue:&validated type:MTLDataTypeBool atIndex:42];
        NSError* err=nil;
        auto kernel=[lib newFunctionWithName:@"nm_registered_bed_endpoint" constantValues:values error:&err];
        check(kernel!=nil,"missing specialized endpoint function");
        endpoint_=[device newComputePipelineStateWithFunction:kernel error:&err];
        check(endpoint_!=nil,"missing endpoint pipeline");
        endpointDecision_=pipeline(lib,@"nm_registered_bed_endpoint_decision");
        endpointIndex_=allocate(sizeof(unsigned));
        columns_=pipeline''')
edit(n,'    ~NumiHumanResolvedBed()', (root/'bed_endpoint_host.inc').read_text()+'    ~NumiHumanResolvedBed()')
edit(n,'pre_dynamics_step','material_step_index')
n='NumiHumanRestingSupportGeometry.hpp'
edit(n,'    void installResolvedBed', (root/'bed_endpoint_support.inc').read_text()+'    void installResolvedBed')
n='NumiHumanRestingCoupling.hpp'
edit(n,'    bool commonAcceptedCoordinatesInitialized','    std::function<void(const metalrobo::MetalNumanXTransactionPass&)> certifyBedEndpoint;\n    bool commonAcceptedCoordinatesInitialized')
edit(n,'            bridge(validateBody);','            if(certifyBedEndpoint)certifyBedEndpoint(p);\n            bridge(validateBody);')
n='NumiHumanRestingVisual.hpp'
edit(n,'            std::cout<<"resting_bed_force=', '            coupled.certifyBedEndpoint=[support=skinSupport.get()](const auto& p){support->certifyBedEndpoint(p);};\n            std::cout<<"resting_bed_force=')
commands=[[arg.replace(str(base)+'/',str(owner)+'/') for arg in cmd] for cmd in read(base/'build-commands.json')]
(owner/'build-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
cwd=Path(read(source/'launch.json')['environment']['DYLD_LIBRARY_PATH']).parent
for cmd in commands:subprocess.run(cmd,check=True,cwd=cwd)
(owner/'identities.json').write_text(json.dumps({k:sha(owner/k) for k in ids},indent=2)+'\n')

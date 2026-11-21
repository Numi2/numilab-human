from pathlib import Path
import shutil,sys
src,dst=map(Path,sys.argv[1:]) if len(sys.argv)>1 else (Path('verified-loaded-owner'),Path('candidate-force-owner'));dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
p=dst/'pelvic_contact.metalinc';s=p.read_text()+'''
kernel void nm_pelvic_capture_force_witness(constant uint& count [[buffer(0)]],
    device const uint4* freeRows [[buffer(1)]],device const float4* bulk [[buffer(2)]],
    device const float4* delta [[buffer(3)]],device const uint* slots [[buffer(4)]],
    device const float* pressure [[buffer(5)]],device const float4* normal [[buffer(6)]],
    device const PelvicPoolSample* pool [[buffer(7)]],device const PelvicRow* row [[buffer(8)]],
    device float4* nodeResult [[buffer(9)]],device float4* portResult [[buffer(10)]],
    uint i [[thread_position_in_grid]]){
    if(i<count){nodeResult[2*i]=float4(bulk[i].xyz,float(freeRows[i].z!=0));nodeResult[2*i+1]=delta[i];}
    if(i<512){
        uint q=slots[i];portResult[4*i]=float4(normal[i].xyz,pressure[i]);
        portResult[4*i+1]=float4(q==UINT_MAX?-1.f:float(q),row[i].leverB.w,0,0);
        if(q!=UINT_MAX){portResult[4*i+2]=float4(pool[q].nodes);portResult[4*i+3]=pool[q].weights;}
        else {portResult[4*i+2]=0;portResult[4*i+3]=0;}
    }
}
''';p.write_text(s);p=dst/'NumiHumanPelvicContact.hpp';s=p.read_text()
def one(a,b):
    global s
    assert s.count(a)==1,a
    s=s.replace(a,b)
one('    std::map<unsigned,unsigned> spanIndex;', '''    unsigned observedForceRoot=UINT_MAX;
    id<MTLBuffer> forceNodes=nil,forcePorts=nil;
    id<MTLComputePipelineState> captureForceWitness=nil;
    std::map<unsigned,unsigned> spanIndex;''')
one('            motionOwners=load(45);','''            if(const char* root=std::getenv("NUMI_HUMAN_PELVIC_FORCE_WITNESS_ROOT")){
                observedForceRoot=unsigned(std::stoul(root));
                if(observedForceRoot>=countTrace)throw std::runtime_error("force witness root exceeds trace");
                forceNodes=alloc(countNodes*32,true);forcePorts=alloc(512*64,true);
                captureForceWitness=pipeline(@"nm_pelvic_capture_force_witness");
            }
            motionOwners=load(45);''')
m='        [e setComputePipelineState:gate];words(0,{p.stepIndex,0,0,unsigned(post)});'
one(m,'''        if(post && p.stepIndex==observedForceRoot && captureForceWitness){
            [e setComputePipelineState:captureForceWitness];[e setBytes:&countNodes length:4 atIndex:0];
            bind(1,kr);bind(2,bulkGrad);bind(3,totalDelta);bind(4,poolSlots);bind(5,pressures);bind(6,localN);
            bind(7,pool);bind(8,rows);bind(9,forceNodes);bind(10,forcePorts);run((countNodes+127)/128,128);
        }
'''+m)
m='        const float* saved=(const float*)accepted.contents;'
one(m,'''        if(forceNodes && encodedSteps>observedForceRoot){
            auto dir=std::filesystem::path(file).parent_path();
            auto emit=[&](const char* name,id<MTLBuffer> b){std::ofstream f(dir/name,std::ios::binary);f.write(static_cast<const char*>(b.contents),b.length);};
            emit("candidate-force-nodes.bin",forceNodes);emit("candidate-force-ports.bin",forcePorts);
        }
'''+m);p.write_text(s)

from pathlib import Path
import shutil,sys
src,dst=map(Path,sys.argv[1:]);dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
p=dst/'pelvic_contact.metalinc';s=p.read_text()
start=s.index('kernel void nm_pelvic_schur_columns(');end=s.index('inline float pelvicGram',start);base=s[start:end]
old=base
# Share the exact arithmetic and preserve the original uncompressed path for comparison.
a,b=base.split('kernel void nm_pelvic_schur_gram(')
a=a.replace('kernel void nm_pelvic_schur_columns(constant uint4& d [[buffer(0)]],','inline void pelvicComputeColumn(uint side,uint j,uint node,')
a=a.replace('uint gid [[thread_position_in_grid]])','uint conditional)')
a=a.replace('uint side=d.x;uint4 meta=pelvicFlex[side];uint j=gid/meta.y,node=gid%meta.y;', 'uint4 meta=pelvicFlex[side];')
a=a.replace('d.y&&!needed[side]', 'conditional&&!needed[side]')
import re
a=re.sub(r' \[\[buffer\(\d+\)\]\]','',a)
b='inline void pelvicComputeGram(uint side,uint i,uint j,'+b.split('constant uint4& d [[buffer(0)]],',1)[1]
b=b.replace('uint gid [[thread_position_in_grid]])','uint conditional)').replace('uint side=gid>>16,i=(gid>>8)&255,j=gid&255,base=side*256;', 'uint gid=65536*side+256*i+j,base=side*256;').replace('d.y&&!needed[side]','conditional&&!needed[side]')
b=re.sub(r' \[\[buffer\(\d+\)\]\]','',b)
start1=old[:old.index('{')+1];g=old[old.index('kernel void nm_pelvic_schur_gram'):];start2=g[:g.index('{')+1]
c=start1+'uint n=pelvicFlex[d.x].y;pelvicComputeColumn(d.x,gid/n,gid%n,slots,pool,normals,map,column,response,needed,d.y);\n}\n'
c+=start2+'pelvicComputeGram(gid>>16,(gid>>8)&255,gid&255,slots,pool,normals,response,W,needed,d.y);\n}\n'
c+=Path('pelvic_changed_columns.metalinc').read_text()
c+=start1.replace('nm_pelvic_schur_columns','nm_pelvic_changed_columns').replace('uint gid [[','device const uint* order [[buffer(8)]],device const uint* counts [[buffer(9)]],uint gid [[')+'''
    uint n=pelvicFlex[d.x].y,j=gid/n;if(j>=counts[d.x])return;
    pelvicComputeColumn(d.x,order[256*d.x+j],gid%n,slots,pool,normals,map,column,response,needed,1);
}
'''
c+=start2.replace('nm_pelvic_schur_gram','nm_pelvic_changed_gram').replace('uint gid [[','device const uint* order [[buffer(7)]],device const uint* counts [[buffer(8)]],device const uint* dirty [[buffer(9)]],uint gid [[')+'''
    uint side=d.x,k=gid/512;if(k>=counts[side])return;
    uint changed=order[side*256+k],other=gid&255;bool column=(gid&256)!=0;
    if(column&&dirty[side*256+other])return;
    pelvicComputeGram(side,column?other:changed,column?changed:other,slots,pool,normals,response,W,needed,1);
}
'''
s=s[:start]+a+b+c+s[end:];p.write_text(s)
p=dst/'NumiHumanPelvicContact.hpp';s=p.read_text()
s=s.replace('unsigned poolCount=0,', '''id<MTLBuffer> portKeys=nil,dirtyPorts=nil,orderedPorts=nil,complianceArgs=nil,changedCounts=nil,complianceTotals=nil;
    id<MTLComputePipelineState> markPorts=nil,changedColumns=nil,changedGram=nil;
    unsigned poolCount=0,''')
s=s.replace('motionOwners=load(45);','''portKeys=alloc(512*48,true);std::memset(portKeys.contents,0,portKeys.length);
            dirtyPorts=alloc(512*4);orderedPorts=alloc(512*4);changedCounts=alloc(8);complianceArgs=alloc(48);
            complianceTotals=alloc(8*8,true);std::memset(complianceTotals.contents,0,complianceTotals.length);
            markPorts=pipeline(@"nm_pelvic_changed_ports");changedColumns=pipeline(@"nm_pelvic_changed_columns");changedGram=pipeline(@"nm_pelvic_changed_gram");
            motionOwners=load(45);''')
s=s.replace('        auto compliance=[&](){','''        auto compliance=[&](){
            if constexpr(pelvicMotionRegistry){
                [e setComputePipelineState:markPorts];bind(0,poolSlots);bind(1,pool);bind(2,localN);bind(3,needed);
                bind(4,portKeys);bind(5,dirtyPorts);bind(6,orderedPorts);bind(7,complianceArgs);bind(8,changedCounts);bind(9,complianceTotals);
                // Clear stale indirect counts for asleep sides too.
                [e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
                for(unsigned side=0;side<2;++side){
                    [e setComputePipelineState:changedColumns];words(0,{side,1,0,0});bind(1,poolSlots);bind(2,pool);bind(3,localN);
                    bind(4,freeIndex);bind(5,flexibility);bind(6,columns);bind(7,needed);bind(8,orderedPorts);bind(9,changedCounts);
                    [e dispatchThreadgroupsWithIndirectBuffer:complianceArgs indirectBufferOffset:24*side threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                }
                [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
                for(unsigned side=0;side<2;++side){
                    [e setComputePipelineState:changedGram];words(0,{side,1,0,0});bind(1,poolSlots);bind(2,pool);bind(3,localN);bind(4,columns);
                    bind(5,gram);bind(6,needed);bind(7,orderedPorts);bind(8,changedCounts);bind(9,dirtyPorts);
                    [e dispatchThreadgroupsWithIndirectBuffer:complianceArgs indirectBufferOffset:24*side+12 threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                }
                [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
                return;
            }''')
s=s.replace('        const float* saved=', '''        if(complianceTotals){
            const auto* v=static_cast<const uint64_t*>(complianceTotals.contents);std::ofstream f(std::filesystem::path(file).parent_path()/"compliance-work.csv");
            f<<"side,refreshes,inspected_ports,changed_ports,recomputed_gram_entries\\n";
            for(unsigned j=0;j<2;++j)f<<j<<','<<v[4*j]<<','<<v[4*j+1]<<','<<v[4*j+2]<<','<<v[4*j+3]<<'\\n';
        }
        const float* saved=''')
p.write_text(s)

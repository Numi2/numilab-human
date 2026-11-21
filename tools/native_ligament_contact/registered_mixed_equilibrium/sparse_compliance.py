"""Rebuild only exact changed Schur rows and columns of the current material operator."""
from pathlib import Path
import shutil,sys
src,dst=map(Path,sys.argv[1:]);dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
p=dst/'pelvic_contact.metalinc';s=p.read_text()
i=s.index('kernel void nm_pelvic_schur_columns');j=s.index('inline float pelvicGram',i)
orig=s[i:j];c=orig[:orig.index('kernel void nm_pelvic_schur_gram')];g=orig[len(c):]
c=c.replace('nm_pelvic_schur_columns','nm_pelvic_changed_columns').replace('uint gid [[thread_position_in_grid]]','device const uint* ids [[buffer(8)]],uint gid [[thread_position_in_grid]]')
c=c.replace('uint j=gid/meta.y,node=gid%meta.y;', '''uint ord=gid/meta.y,node=gid%meta.y;
    if(ord>=d.z)return;uint j=ids[256*side+ord];''').replace('if(ord>=d.z)return;', 'if(ord>=ids[512+side])return;')
g=g.replace('nm_pelvic_schur_gram','nm_pelvic_changed_gram').replace('uint gid [[thread_position_in_grid]]','device const uint* ids [[buffer(7)]],device const uint* changed [[buffer(8)]],uint gid [[thread_position_in_grid]]')
old='uint side=gid>>16,i=(gid>>8)&255,j=gid&255,base=side*256;'
assert old in g
g=g.replace(old,'''uint side=d.x,base=side*256,count=ids[512+side],i,j;
    if(count>=128){if(gid>=65536)return;i=gid>>8;j=gid&255;}
    else{
        uint half=count*256;if(gid>=2*half)return;
        bool transpose=gid>=half;uint at=transpose?gid-half:gid;
        uint dirty=ids[base+(at>>8)],other=at&255;
        if(transpose && changed[base+other])return;
        i=transpose?other:dirty;j=transpose?dirty:other;
    }
    gid=(side<<16)+(i<<8)+j;'''.replace('half=count','area=count').replace('2*half','2*area').replace('>=half','>=area').replace('gid-half','gid-area'))
prep='''// All mutable inputs to G are compared bitwise. K, its factor, the material
// interpolation, and the row layout are immutable for this configured owner.
kernel void nm_pelvic_changed_rows(device const uint* slots [[buffer(0)]],device const PelvicPoolSample* pool [[buffer(1)]],
    device const float4* normals [[buffer(2)]],device const uint* needed [[buffer(3)]],
    device uint4* history [[buffer(4)]],device uint* ids [[buffer(5)]],device uint* changed [[buffer(6)]],
    device uint* args [[buffer(7)]],device atomic_uint* counts [[buffer(8)]],
    uint side [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    uint base=side*256,used=0,count=0;
    if(needed[side]){
        for(uint chunk=0;chunk<8;++chunk){
            uint row=32*chunk+lane,k=base+row,slot=slots[k],at=4*k;
            uint4 a=uint4(0,0,0,slot),b=0,c=0;
            if(slot!=UINT_MAX){a.xyz=as_type<uint3>(normals[k].xyz);b=pool[slot].nodes;c=as_type<uint4>(pool[slot].weights);}
            bool diff=history[at+3].x==0 || any(a!=history[at]) || any(b!=history[at+1]) || any(c!=history[at+2]);
            changed[k]=uint(diff);
            if(diff){history[at]=a;history[at+1]=b;history[at+2]=c;history[at+3]=uint4(1,0,0,0);}
            uint rank=simd_prefix_exclusive_sum(uint(diff));if(diff)ids[base+count+rank]=row;
            count+=simd_sum(uint(diff));used+=simd_sum(uint(slot!=UINT_MAX));
        }
    }
    if(!lane){
        ids[512+side]=count;
        uint offset=6*side;
        args[offset]=(count*pelvicFlex[side].y+127)/128;args[offset+1]=1;args[offset+2]=1;
        args[offset+3]=(count>=128?65536:count*512)/128;args[offset+4]=1;args[offset+5]=1;
        if(needed[side]){
            atomic_fetch_add_explicit(counts,1u,memory_order_relaxed);
            atomic_fetch_add_explicit(counts+1,count,memory_order_relaxed);
            atomic_fetch_add_explicit(counts+2,used,memory_order_relaxed);
        }
    }
}
'''
s=s[:j]+prep+c+g+s[j:];p.write_text(s)
p=dst/'NumiHumanPelvicContact.hpp';s=p.read_text()
def one(a,b):
    global s
    if s.count(a)!=1:raise ValueError(a)
    s=s.replace(a,b)
one('    std::map<unsigned,unsigned> spanIndex;', '''    id<MTLComputePipelineState> changedRows=nil,changedColumns=nil,changedGram=nil;
    id<MTLBuffer> rowHistory=nil,rowList=nil,rowDirty=nil,rowArgs=nil,rowStats=nil;
    std::map<unsigned,unsigned> spanIndex;''')
one('        samples=alloc(512*80);','''        rowHistory=alloc(512*64,true);std::memset(rowHistory.contents,0,rowHistory.length);
        rowList=alloc(514*4);rowDirty=alloc(512*4);rowArgs=alloc(12*4);
        rowStats=alloc(12,true);std::memset(rowStats.contents,0,rowStats.length);
        samples=alloc(512*80);''')
one('        factorApply=pipeline(', '        changedRows=pipeline(@"nm_pelvic_changed_rows");changedColumns=pipeline(@"nm_pelvic_changed_columns");changedGram=pipeline(@"nm_pelvic_changed_gram");\n        factorApply=pipeline(')
i=s.index('        auto compliance=[&](){');j=s.index('        auto certify=',i)
s=s[:i]+'''        auto compliance=[&](){
            bool wasOptional=optional;optional=false;
            [e setComputePipelineState:changedRows];bind(0,poolSlots);bind(1,pool);bind(2,localN);
            bind(3,needed);bind(4,rowHistory);bind(5,rowList);bind(6,rowDirty);bind(7,rowArgs);bind(8,rowStats);run(2,32);
            optional=wasOptional;
            for(unsigned side=0;side<2;++side){
                [e setComputePipelineState:changedColumns];words(0,{side,1,0,0});bind(1,poolSlots);bind(2,pool);bind(3,localN);
                bind(4,freeIndex);bind(5,flexibility);bind(6,columns);bind(7,needed);bind(8,rowList);
                [e dispatchThreadgroupsWithIndirectBuffer:rowArgs indirectBufferOffset:24*side threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            }
            [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
            for(unsigned side=0;side<2;++side){
                [e setComputePipelineState:changedGram];words(0,{side,1,0,0});bind(1,poolSlots);bind(2,pool);bind(3,localN);
                bind(4,columns);bind(5,gram);bind(6,needed);bind(7,rowList);bind(8,rowDirty);
                [e dispatchThreadgroupsWithIndirectBuffer:rowArgs indirectBufferOffset:24*side+12 threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            }
            [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        };
'''+s[j:]
one('        std::ofstream physical(', '''        auto* rc=static_cast<const uint32_t*>(rowStats.contents);
        std::ofstream schur(std::filesystem::path(file).parent_path()/"schur-row-builds.csv");
        schur<<"active_side_builds,changed_rows,occupied_rows\\n"<<rc[0]<<','<<rc[1]<<','<<rc[2]<<'\\n';
        std::ofstream physical(''')
p.write_text(s)

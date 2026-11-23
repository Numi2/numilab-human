"""Recover the actual outer material surface and reject a failed viewer audit before capture."""
import json,subprocess,sys
from pathlib import Path
from material_context import sha
from export_contact_columns import change
def build(base,dest,source):
    ids=json.loads((base/'identities.json').read_text())
    if dest.exists():raise FileExistsError(dest)
    if any(sha(base/k)!=v for k,v in ids.items()):raise ValueError('changed source')
    subprocess.run(['cp','-cR',base,dest],check=True)
    def edit(n,a,b):
        p=dest/n;p.write_text(change(p.read_text(),a,b))
    n='NumiHumanResolvedBed.metalinc';p=dest/n;s=p.read_text()
    start=s.index('    uint lo=0,hi=256;');end=s.index('    // The tangent',start)
    block=s[start:end]
    helper='''struct SeriesBedInterval { uint lo,hi; float a,b,t; };
inline SeriesBedInterval seriesBedInterval(float4 c,float depth) {
'''+block+'''    return {lo,hi,a,b,(depth-a)/(b-a)};
}
'''
    s=s[:start]+'''    const auto segment=seriesBedInterval(c,depth);
    const uint lo=segment.lo, hi=segment.hi;
    const float a=segment.a, b=segment.b;
'''+s[end:]
    s=change(s,'kernel void nm_registered_bed_columns(',helper+'\nkernel void nm_registered_bed_columns(')
    p.write_text(s)
    edit('NumiHumanResolvedBed.hpp','    ~NumiHumanResolvedBed()', '    id<MTLBuffer> materialStock() const { return stock_; }\n    ~NumiHumanResolvedBed()')
    n='NumiHumanRestingVisual.hpp'
    edit(n,'    unsigned skinFirstVertex=MR_INVALID_INDEX;', '    id<MTLBuffer> foundationSkinStock=nil;\n    unsigned skinFirstVertex=MR_INVALID_INDEX;')
    edit(n,'            skinSupport->installResolvedBed(std::make_unique<NumiHumanResolvedBed>','            auto materialBed=std::make_unique<NumiHumanResolvedBed>')
    edit(n,'supportInfluenceLayout==NumiHumanSkinInfluenceLayout::tile32?32u:1u,resolvedWitness));','''supportInfluenceLayout==NumiHumanSkinInfluenceLayout::tile32?32u:1u,resolvedWitness);
            foundationSkinStock=materialBed->materialStock();
            skinSupport->installResolvedBed(std::move(materialBed));''')
    anchor='        e.setBytes(e.context,&skinDispatch,sizeof(skinDispatch),0);'
    edit(n,anchor,anchor+'''
        const std::array<unsigned,2> boundSkin={self.skinFirstVertex,
            unsigned(self.foundationSkinStock.length/sizeof(mr_float4))};
        e.setBuffer(e.context,(__bridge void*)(self.foundationSkinStock ? self.foundationSkinStock : self.mapping),0,14);
        e.setBytes(e.context,boundSkin.data(),sizeof(boundSkin),15);''')
    checks='''        require(wallCorrection.w==0,"accepted ventricular material volume closure failed");
        require(skinAudit.w==0&&skinAudit.z==0,"accepted full skin intersects the bed beyond the 1 mm inspection tolerance");'''
    p=dest/n;s=p.read_text();s=change(s,checks,'')
    anchor='        if(captureThisFrame) {\n            NSDictionary* surfaceAuditOutcome=@{'
    s=change(s,anchor,checks+'\n'+anchor);p.write_text(s)
    n='human_respiration.metal'
    anchor='    device float4* splitLow [[buffer(13)]],\n    uint i'
    edit(n,anchor,'''    device float4* splitLow [[buffer(13)]],
    device const float4* foundationStock [[buffer(14)]], constant uint2& boundSkin [[buffer(15)]],
    uint i''')
    edit(n,'    if(m.deformationKind && anatomy.bodyAndFlags.y) {','''    const float carrierZ=p.z;
    if(m.deformationKind && anatomy.bodyAndFlags.y) {''')
    edit(n,'    if (m.deformationKind == 3u && d.w != 0u && stock > 0.0f &&','''    if(m.deformationKind==3u && i>=boundSkin.x && i-boundSkin.x<boundSkin.y) {
        const float4 c=foundationStock[i-boundSkin.x];
        const float compression=foamSurfaceZ-carrierZ;
        if(c.x>0.f && compression>0.f) {
            if(compression>bedClosure(c.yz,256u)) p.z=NAN;
            else {
                const auto interval=seriesBedInterval(c,compression);
                const float2 tissueStrain=mix(seriesBed[interval.lo].xy,seriesBed[interval.hi].xy,interval.t);
                p.z+=dot(c.yz,tissueStrain);
            }
        }
    } else if (m.deformationKind == 3u && d.w != 0u && stock > 0.0f &&''')
    cmds=[[v.replace(str(base)+'/',str(dest)+'/') for v in c] for c in json.loads((base/'build-commands.json').read_text())]
    cwd=Path(json.loads((source/'launch.json').read_text())['environment']['DYLD_LIBRARY_PATH']).parent
    for c in cmds:subprocess.run(c,cwd=cwd,check=True)
    (dest/'build-commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
    (dest/'identities.json').write_text(json.dumps({k:sha(dest/k) for k in ids},indent=2)+'\n')
if __name__=='__main__':build(*(Path(x).resolve() for x in sys.argv[1:]))

"""Reserve simultaneous material/contact recovery for a candidate that exhausted the selected owner."""
from pathlib import Path
import shutil
src=Path('mixed-recovery-owner');base=Path('nodal-equilibrium-owner');dst=Path('conditional-facet-owner');dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
p=dst/'pelvic_contact.metalinc';s=p.read_text()
old='device const float* volume [[buffer(11)]],device float4* coordinates [[buffer(12)]],'
assert s.count(old)==1;s=s.replace(old,old+'constant uint& coupled [[buffer(13)]],')
a='q.x<=tolerance)return;';assert s.count(a)==1
s=s.replace(a,'q.x<=tolerance || (!coupled && (q.z>=5.e-7f || q.w>2.e-9f)))return;');p.write_text(s)
p=dst/'NumiHumanPelvicContact.hpp';s=p.read_text()
a='unsigned(outer==15u || outer==31u || (outer>=48u && outer!=UINT_MAX))'
assert s.count(a)==1;s=s.replace(a,'unsigned(outer>=51u && outer!=UINT_MAX)')
def loops(text):
    m='        if constexpr(pelvicMotionRegistry){\n            for(unsigned local=0;local<4;++local){'
    a=text.index(m);b=text.index('        [e setComputePipelineState:gate];',a)
    return a,b,text[a:b]
a,b,mixed=loops(s);original=loops((base/p.name).read_text())[2]
needle='bind(12,coordinates);run(2,32);';assert original.count(needle)==mixed.count(needle)==1
original=original.replace(needle,'bind(12,coordinates);words(13,{0,0,0,0});run(2,32);')
mixed=mixed.replace(needle,'bind(12,coordinates);words(13,{1,0,0,0});run(2,32);').replace('certify(48+local)','certify(52+local)')
p.write_text(s[:a]+original+mixed+s[b:])

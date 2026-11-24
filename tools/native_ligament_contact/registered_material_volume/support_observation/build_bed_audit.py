"""Build the native surface observer against the same registered support as the contact assembly."""
import hashlib,json,subprocess,sys
from bind_loaded_exterior import bind
from pathlib import Path

def sha(p):
    return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()

def replace_once(text, before, after):
    if text.count(before)!=1: raise ValueError(f'Unexpected source at {before[:70]}')
    return text.replace(before,after)

def build(binding,recipe,out):
    cfg=json.loads((binding/'build.json').read_bytes())
    cc=cfg['commands'][0].copy();link=cfg['paired_link'].copy()
    key='-DNUMI_HUMAN_RESPIRATION_METALLIB='
    lib=Path(next(x[len(key):].strip('"') for x in cc if x.startswith(key)))
    oldsrc=lib.parent.parent/'human_respiration.metal';oldapp=binding/'app'
    root=out.resolve();app=root/'app';metal=root/'metal'
    app.mkdir(parents=True,exist_ok=True);metal.mkdir(exist_ok=True)
    # Headers with immutable relative includes retain the parent's search directory.
    src=app/'numilab_human_myosim_visual_probe.mm';src.write_text(bind((oldapp/src.name).read_text(),replace_once))
    h=oldapp/'NumiHumanRestingVisual.hpp';host=h.read_text()
    host=replace_once(host,'    MRHumanRestingBedGPU bedParameters{};','    MRHumanRestingBedGPU bedParameters{};\n    mr_float4 auditPlane{};\n    unsigned exteriorSkinCount=0;')
    host=replace_once(host,'                skinFirstVertex=base;','                skinFirstVertex=base;\n                exteriorSkinCount=skin.header.vertexCount;')
    host=replace_once(host,'        const float planeOffset = dotPoint(planeNormal, planePoint);','        const float planeOffset = dotPoint(planeNormal, planePoint);\n        auditPlane={planeNormal.x,planeNormal.y,planeNormal.z,-planeOffset};')
    host=replace_once(host,'{d.x,self.skinAuditGroupCount,0,0}','{d.x,self.skinAuditGroupCount,self.skinFirstVertex,self.exteriorSkinCount}')
    host=replace_once(host,'        if(self.bedParameters.counts.z) {','        e.setBytes(e.context,&self.auditPlane,sizeof(self.auditPlane),6);\n        if(self.bedParameters.counts.z) {')
    # Version the coupled host/shader interface so a mismatched old library fails at startup.
    oldkernel='nm_human_resting_audit_skin_partials';newkernel=oldkernel+'_support_plane'
    host=replace_once(host,oldkernel,newkernel);(app/h.name).write_text(host)
    shader=oldsrc.read_text();idx=shader.index('kernel void '+oldkernel+'(');end=shader.index('\nkernel void ',idx+1);kernel=shader[idx:end]
    kernel=replace_once(kernel,oldkernel,newkernel)
    # The source range excludes inner layers and other derived skin geometry.
    kernel=replace_once(kernel,'device const MRHumanRestingVertexMap* map [[buffer(1)]],','')
    host=replace_once(host,'e.setBuffer(e.context,(__bridge void*)self.mapping,0,1);e.setBuffer(e.context,lease.meshVertices,0,2);\n        e.setBuffer(e.context,(__bridge void*)self.skinAuditPartials','e.setBuffer(e.context,lease.meshVertices,0,2);\n        e.setBuffer(e.context,(__bridge void*)self.skinAuditPartials')
    (app/h.name).write_text(host)
    kernel=replace_once(kernel,'    uint lane [[thread_index_in_threadgroup]],','    constant float4& supportPlane [[buffer(6)]],\n    uint lane [[thread_index_in_threadgroup]],')
    kernel=replace_once(kernel,'if(map[i].deformationKind==3u)', 'if(i>=d.z && i-d.z<d.w)')
    kernel=replace_once(kernel,'float z=vertices[i].position.z;', 'float z=dot(vertices[i].position.xyz,supportPlane.xyz)+supportPlane.w;')
    ms=metal/oldsrc.name;ms.write_text(shader[:idx]+kernel+shader[end:])
    mc,ml=json.loads(recipe.read_bytes())[:2];mc=list(mc);ml=list(ml)
    air=metal/'HumanRestingSkin.air';newlib=metal/lib.name
    mc[mc.index('-c')+1]=str(ms);mc[mc.index('-o')+1]=str(air);mc[mc.index('-c'):mc.index('-c')]=['-I',str(oldsrc.parent)]
    ml=[str(air) if x.endswith('.air') else str(newlib) if x.endswith('.metallib') else x for x in ml]
    obj=app/'numi-human-native.o';exe=app/'numi-human-native'
    cc=[key+'"'+str(newlib)+'"' if x.startswith(key) else str(obj) if x==str(oldapp/obj.name) else str(src) if x==str(oldapp/src.name) else x for x in cc]
    cc[1:1]=['-iquote',str(oldapp)]
    link=[str(obj) if x==str(oldapp/obj.name) else str(exe) if x==str(oldapp/exe.name) else x for x in link]
    assert str(exe) in link and str(obj) in link
    for cmd in (mc,ml,cc,link):subprocess.run(cmd,check=True)
    pins={str(p):sha(p)for p in (oldsrc,h,lib,oldapp/src.name,src,app/h.name,ms,newlib,exe)}
    (root/'build.json').write_text(json.dumps(dict(commands=[mc,ml,cc,link],pins=pins,library=cfg['library']),indent=2)+'\n')
if __name__=='__main__':build(*(Path(x).resolve() for x in sys.argv[1:]))

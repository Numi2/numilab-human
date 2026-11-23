"""Preserve exact source-face associations through full86 consumption of cortical-cut boundaries."""
import json,shutil,subprocess,sys
from pathlib import Path
from compile_connected_joint import digest
from prepare_live_cortex import exact

def run(base,out):
    out.mkdir();app=out/'app';app.mkdir();src=base/'app'
    for f in src.iterdir():
        if f.suffix in ('.hpp','.h','.inc','.mm'):shutil.copy2(f,app/f.name)
    file=app/'NumiHumanPhysicalSkinAuthoringRoleConsumer.hpp';s=file.read_text()
    s=exact(s,'    std::vector<std::uint32_t> sourceDomainIndices,authoringVertexRows,volumeNodeRows;','''    std::vector<std::uint32_t> sourceDomainIndices,authoringVertexRows,volumeNodeRows;
    // Entries in this sparse list carry their original role-source face and
    // weights. The corresponding authoringVertexRows entry stays invalid.
    std::vector<std::pair<std::uint32_t,PhysicalSkinNodeSourceRow>> embeddedAuthoringRows;''')
    a=s.index('            if(domain.role!=expectedRoles[layer]');b=s.index('            const auto volumeNode=boundary[row];',a)
    original=s[a:b];split=original.index('            if(source.bary')
    direct=original[split:].rstrip()
    s=s[:a]+'''            const bool cutLayer=layer!=0u&&!bundle.corticalFacets.empty();
            const bool embedded=source.face>=0;
            if(domain.role!=expectedRoles[layer]||domain.domain.empty()||
               !validPhysicalSkinSourceDomainSHA256(domain.sourceSHA256)||source.domain!=domain.domain||
               (!cutLayer&&(domain.vertexCount<count||domain.faceCount!=surface.faces.size()))||
               ((cutLayer||domain.vertexCount>count)&&!referenced[row]))
                return fail("authoring row does not belong to its pinned role/source namespace");
            if(embedded){
                if(!cutLayer||source.vertex!=-1||std::uint64_t(source.face)>=domain.faceCount||
                   surface.sourceVertexIndices[row]!=NM_INVALID_INDEX||
                   surface.sourceEmbeddingFaceIndices.size()!=count||surface.sourceEmbeddingBarycentric.size()!=count||
                   surface.sourceEmbeddingResidualWorldM.size()!=count||
                   surface.sourceEmbeddingFaceIndices[row]!=std::uint32_t(source.face)||
                   surface.sourceEmbeddingBarycentric[row]!=source.bary||
                   surface.sourceEmbeddingResidualWorldM[row]!=source.residual||
                   !validateBarycentricCoordinates(source.bary,error))
                    return fail("authoring point does not reconstruct its registered source-face embedding");
                roleRows.embeddedAuthoringRows.emplace_back(std::uint32_t(row),source);
            }else{
                if(source.face!=-1||source.vertex<0||std::uint64_t(source.vertex)>=domain.vertexCount||
                   surface.sourceVertexIndices[row]!=std::uint32_t(source.vertex)||
                   !sourceVertices.insert(std::uint32_t(source.vertex)).second||
                   (!cutLayer&&domain.vertexCount==count&&std::uint64_t(source.vertex)!=row))
                    return fail("authoring vertex ID is not an exact unique active source row in its pinned namespace");
'''+ '\n'.join('    '+line for line in direct.splitlines())+'\n            }\n'+s[b:]
    s=exact(s,'bundle.volumeNodeSourceFaceIndices[volumeNode]!=NM_INVALID_INDEX||','''bundle.volumeNodeSourceFaceIndices[volumeNode]!=(embedded?std::uint32_t(source.face):NM_INVALID_INDEX)||
               (embedded&&(bundle.volumeNodeSourceBarycentric[volumeNode]!=source.bary||
                  importSidecars.sourceAncestryRows[volumeNode].residual!=source.residual))||''');file.write_text(s)
    # Resolve each fixed seed once during preparation. The original GPU BVH still
    # chooses the current closest triangle throughout physical execution.
    mm=app/'numilab_human_myosim_visual_probe.mm';s=mm.read_text()
    s=exact(s,'            for(const auto& row:registeredCorticalSkin::samples){','''            std::map<std::pair<unsigned,unsigned>,const Triangle*> corticalSeeds;
            for(const auto& row:registeredCorticalSkin::samples){''')
    s=exact(s,'''                const auto first=std::find_if(triangles.begin(),triangles.end(),[&](const auto& v){return v.boneStableId==row.bank;});
                require(first!=triangles.end(),"registered cortical master has no prepared triangle");''','''                auto& first=corticalSeeds[{row.body,row.bank}];
                if(!first){
                    const auto found=std::find_if(triangles.begin(),triangles.end(),
                        [&](const auto& v){return v.boneStableId==row.bank;});
                    require(found!=triangles.end(),"registered cortical master has no prepared triangle");
                    first=&*found;
                }''');mm.write_text(s)
    receipt=json.loads((base/'build.json').read_bytes());commands=[[x.replace(str(src)+'/',str(app)+'/') for x in cmd] for cmd in receipt['commands']]
    with (out/'build.log').open('wb') as log:
        for cmd in commands:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    (out/'build.json').write_text(json.dumps(dict(commands=commands,source=digest(base/'build.json'),consumer=digest(file),executable=digest(app/'numi-human-native')),indent=2)+'\n')
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))

"""Connect exact cortical cavity faces to native topology admission and the existing Metal contact owner."""
import json,sys,shutil,subprocess
from pathlib import Path
import numpy as np
from prepare_live_cortex import exact
from compile_connected_joint import digest

def run(base,contact,out):
    out.mkdir();src=base/'app';app=out/'app';app.mkdir()
    for f in src.iterdir():
        if f.suffix in ('.hpp','.inc','.h','.mm'):shutil.copy2(f,app/f.name)
    def patch(name,pairs):
        f=app/name;s=f.read_text()
        for a,b in pairs:s=exact(s,a,b)
        f.write_text(s)
    un='NumiHumanUnpairedSkinFatVolume.hpp';mass='NumiHumanPhysicalSkinChildMassMapping.hpp';integ='NumiHumanPhysicalSkinChildIntegration.hpp'
    faces='NumiHumanPhysicalSkinChildRegisteredFacetMaps.hpp';vol='NumiHumanPhysicalSkinChildRegisteredVolumeIntegration.hpp'
    section='struct ExplicitUnpairedVolumeTopology {'
    definition="""struct PhysicalSkinCorticalFacet {
    std::uint32_t child, side, material, bank, body;
    std::array<std::uint32_t,3> nodes,sourceTriangles;
    std::uint64_t slot() const {return 4ull*child+side;}
};

inline const PhysicalSkinCorticalFacet* corticalFacetAt(
        std::span<const PhysicalSkinCorticalFacet> rows, std::uint32_t cell, std::uint32_t side){
    const auto slot=4ull*cell+side;
    const auto it=std::lower_bound(rows.begin(),rows.end(),slot,[](const auto& row,auto s){return row.slot()<s;});
    return it!=rows.end()&&it->slot()==slot?&*it:nullptr;
}

inline bool corticalFacetMatches(const PhysicalSkinCorticalFacet& row,
        std::array<std::uint32_t,3> nodes, std::uint32_t material){
    auto authored=row.nodes;std::sort(authored.begin(),authored.end());std::sort(nodes.begin(),nodes.end());
    return row.material==material&&row.side<4u&&row.body!=NM_INVALID_INDEX&&row.bank!=NM_INVALID_INDEX&&
        authored==nodes&&std::none_of(row.sourceTriangles.begin(),row.sourceTriangles.end(),[](auto v){return v==NM_INVALID_INDEX;});
}

"""
    branch='''                if(declared!=invalid||declaredStable!=invalid||wall!=1u||wallID==invalid)
                    return fail("unpaired skin/fat external boundary is neither O/D registered surface nor explicitly authored head sidewall");'''
    patch(un,[(section,definition+section+'\n    std::span<const PhysicalSkinCorticalFacet> corticalFacets;'),('    for(const auto& [key,uses]:facetUses){','    std::size_t usedCorticalFacets=0;\n    for(const auto& [key,uses]:facetUses){'),(branch,'''                const auto* cortical=corticalFacetAt(topology.corticalFacets,use.child,use.omitted);
                if(cortical){
                    if(declared!=invalid||declaredStable!=invalid||wall||wallID!=invalid||
                       !corticalFacetMatches(*cortical,key,topology.childMaterialIndices[use.child]))
                        return fail("cortical tissue facet has inconsistent nodes, material, or duplicate ownership");
                    ++usedCorticalFacets;
                }else if(declared!=invalid||declaredStable!=invalid||wall!=1u||wallID==invalid)
                    return fail("unowned external skin/fat material facet");'''),('    for(unsigned role=0;role<3u;++role)for(std::size_t face=0;face<layerFaceCounts[role].size();++face){','    if(usedCorticalFacets!=topology.corticalFacets.size())return fail("unused cortical tissue facet");\n    for(unsigned role=0;role<3u;++role)for(std::size_t face=0;face<layerFaceCounts[role].size();++face){')])
    head='    hashIndices("child.parent_surface_faces",bundle.childParentSurfaceFaces);'
    patch(mass,[('struct PhysicalSkinChildMassMappingBundle {','struct PhysicalSkinChildMassMappingBundle {\n    std::vector<PhysicalSkinCorticalFacet> corticalFacets;'),('        topology.childAuthoredHeadSidewallFacetFlags=childAuthoredHeadSidewallFacetFlags;','        topology.corticalFacets=corticalFacets;\n        topology.childAuthoredHeadSidewallFacetFlags=childAuthoredHeadSidewallFacetFlags;'),(head,'''    if(!bundle.corticalFacets.empty()){
        std::vector<std::uint32_t> rows;
        for(const auto& r:bundle.corticalFacets){
            rows.insert(rows.end(),{r.child,r.side,r.material,r.body,r.bank});
            rows.insert(rows.end(),r.nodes.begin(),r.nodes.end());
            rows.insert(rows.end(),r.sourceTriangles.begin(),r.sourceTriangles.end());
        }
        hashIndices("child.cortical_contact_facets",rows);
    }
'''+head)])
    p='    copyRows(bundle.childAuthoredHeadSidewallFacetFlags, topology.childAuthoredHeadSidewallFacetFlags);'
    q='    for (const auto value : bundle.childAuthoredHeadSidewallFacetFlags) hash.u32(value);'
    patch(integ,[(p,'    copyRows(bundle.corticalFacets, topology.corticalFacets);\n'+p),(q,'''    if(!bundle.corticalFacets.empty()){
        hash.string("physical_skin_cortical_contact_facets");hash.u64(bundle.corticalFacets.size());
        for(const auto& r:bundle.corticalFacets){
            for(auto v:{r.child,r.side,r.material,r.body,r.bank})hash.u32(v);
            for(auto v:r.nodes)hash.u32(v);
            for(auto v:r.sourceTriangles)hash.u32(v);
        }
    }
'''+q)])
    a='''            if(wall==wallByFacet.end())
                return fail("external child facet has no exact registered surface or retained authored head wall");'''
    patch(faces,[('    std::size_t usedWalls=0u;','''    std::size_t usedWalls=0u,usedCortex=0u;
    for(std::size_t i=0;i<bundle.corticalFacets.size();++i){
        const auto& r=bundle.corticalFacets[i];
        if(r.child>=cells||r.side>=4u||(i&&r.slot()<=bundle.corticalFacets[i-1].slot()))
            return fail("noncanonical or out-of-bounds cortical material facet");
    }'''),(a,'''            const auto* cortex=corticalFacetAt(bundle.corticalFacets,use.cell,use.omitted);
            if(cortex){
                if(wall!=wallByFacet.end()||!corticalFacetMatches(*cortex,key,bundle.childMaterialIndices[use.cell]))
                    return fail("physical cortical facet conflicts with its source/material registration");
                ++usedCortex;continue;
            }
'''+a),('    measured.authoredHeadWallFaces=usedWalls;','    if(usedCortex!=bundle.corticalFacets.size())return fail("cortical facets not all used exactly once");\n    measured.authoredHeadWallFaces=usedWalls;')])
    h='    if(!preparePhysicalSkinRegisteredFacetMaps(nextBundle,'
    patch(vol,[('struct PhysicalSkinRegisteredVolumeAssetFiles {','struct PhysicalSkinRegisteredVolumeAssetFiles {\n    std::span<const PhysicalSkinCorticalFacet> corticalFacets;'),(h,'    nextBundle.corticalFacets.assign(files.corticalFacets.begin(),files.corticalFacets.end());\n'+h)])
    rec=json.loads((contact/'result.json').read_bytes());f=contact/'material-contact.npz'
    if digest(f)!=rec['program']:raise ValueError('Unregistered material program')
    z=np.load(f);rows=z['facets'];rows=rows[np.lexsort((rows[:,1],rows[:,0]))];s=z['samples']
    flat=lambda values:','.join(map(str,map(int,values)))
    texts=['namespace registeredCorticalSkin {','using Facet=metalrobo::human::coarse_tissue::PhysicalSkinCorticalFacet;','struct Sample{unsigned node,material,bank,body;double area,thickness;};','inline constexpr Facet faces[]={']
    for r in rows:texts.append('{'+flat([*r[:4],r[7]])+',{'+flat(r[4:7])+'},{'+flat(r[8:])+'}},')
    texts+=['};','inline constexpr Sample samples[]={']
    for row in s:texts.append('{'+flat(row[:4])+','+','.join(format(n,'.17g') for n in row[4:])+'},')
    texts+=['};']+['inline constexpr char '+k+'[]="'+rec[k]+'";' for k in ['material_reference','elements','loaded','bone']]+['}']
    (app/'NumiHumanCorticalSkinBinding.inc').write_text('\n'.join(texts)+'\n')
    mm='numilab_human_myosim_visual_probe.mm';line='#include "NumiHumanLoadedSkinPose.hpp"';v='                    PhysicalSkinRegisteredVolumeFileImportReport importReport;'
    step='        const std::size_t totalRows=wholeBodySkinCoarseRest.size()+wholeBodyFatDeepRest.size()-wholeBodyDeepAttachmentRows.size();'
    patch(mm,[(line,line+'\n#include "NumiHumanCorticalSkinBinding.inc"'),(v,v+'''
                    if(volumeFiles.expectedSHA256[1]==registeredCorticalSkin::elements){
                        const auto* skinBone=std::getenv("NUMI_HUMAN_SKIN_BONE_TRIANGLE_CONTACT");
                        const auto* bvh=std::getenv("NUMI_HUMAN_SKIN_BONE_STATIC_BVH");
                        require(volumeFiles.expectedSHA256[0]==registeredCorticalSkin::material_reference&&
                            numiLoadedSkin::enabled()&&std::string_view(numiLoadedData::nodeSHA)==registeredCorticalSkin::loaded&&
                            bones&&bones->sourceOwnerBindingsVerified&&loadedKneeSHA256Hex(bones->payloadSha256)==registeredCorticalSkin::bone&&
                            skinBone&&std::string_view(skinBone)=="1"&&bvh&&std::string_view(bvh)=="1"&&
                            !std::getenv("NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"),
                            "cortical contact lacks the registered skin pose, skeleton, connectivity or native BVH");
                        std::map<unsigned,std::pair<unsigned,unsigned>> sourceBanks;
                        for(const auto& b:bones->records)sourceBanks.emplace(b.stableId,std::pair{b.bodyIndex,b.indexCount/3});
                        for(const auto& r:registeredCorticalSkin::faces){
                            const auto b=sourceBanks.find(r.bank);
                            require(b!=sourceBanks.end()&&b->second.first==r.body&&
                                *std::max_element(r.sourceTriangles.begin(),r.sourceTriangles.end())<b->second.second,
                                "physical cortical facet does not belong to its registered source bone");
                        }
                        volumeFiles.corticalFacets=registeredCorticalSkin::faces;
                    }'''),(step,'''        if(numiLoadedSkin::enabled()&&std::string_view(numiLoadedData::elementsSHA)==registeredCorticalSkin::elements){
            require(staticBVHEnabled&&std::string_view(numiLoadedData::nodeSHA)==registeredCorticalSkin::loaded,
                    "registered cortical surface requires active all-body triangle contact and its loaded skin");
            for(const auto& row:registeredCorticalSkin::samples){
                const std::uint64_t node=std::uint64_t(descriptor.stateOffset)+row.node;
                require(row.node<descriptor.stateCount&&node<compiled.world.fem.nodes.size()&&row.body<bodyBanks.size()&&
                        bodyBankEligible[row.body]&&row.area>0&&row.thickness>0&&(row.material==15u||row.material==16u),
                        "cortical material contact sample is outside the prepared world");
                const auto& triangles=bodyBanks[row.body].triangles;
                const auto first=std::find_if(triangles.begin(),triangles.end(),[&](const auto& v){return v.boneStableId==row.bank;});
                require(first!=triangles.end(),"registered cortical master has no prepared triangle");
                const auto E=row.material==15u?skinYoung:fatYoung;
                const auto sample=triangleContact::makeTriangleContactSample(unsigned(node),row.body,
                    first->bodyLocalVertex0,first->bodyLocalVertex1,first->bodyLocalVertex2,
                    float(row.area),float(E/row.thickness),float(1./E));
                require(triangleContact::validTriangleContactSample(sample),"invalid native cortical contact row");
                fasciaBodyContactSamples.push_back(sample);fasciaBodyContactNeighborhoodRanges.push_back({});
            }
            std::cout<<"registered_cortical_contact=compiled faces="<<std::size(registeredCorticalSkin::faces)
                     <<" nodal_rows="<<std::size(registeredCorticalSkin::samples)<<" owner=existing_metal_body_triangle_bvh"<<std::endl;
        }
'''+step)])
    d=json.loads((base/'build.json').read_bytes());commands=[[x.replace(str(src)+'/',str(app)+'/') for x in c] for c in d['commands']]
    with (out/'build.log').open('wb') as log:
        for c in commands:subprocess.run(c,check=True,stdout=log,stderr=subprocess.STDOUT)
    (out/'build.json').write_text(json.dumps(dict(commands=commands,source=digest(base/'build.json'),compiled_contact=digest(f),executable=digest(app/'numi-human-native')),indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))

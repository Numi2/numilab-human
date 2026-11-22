#pragma once
// Retain the enthesis and reference length from the anatomical registration. Certify
// the transported fiber against the current cortical surfaces before using its force law.
inline void certifyLoadedLigament(const metalrobo::registered_ligaments::Built& fiber,
        const std::vector<MRBodyStateGPU>& reference,const std::vector<MRBodyStateGPU>& current,
        const std::vector<metalrobo::inferred_acl::BoneSurface>& source,
        const std::vector<metalrobo::inferred_acl::BoneSurface>& posed){
    namespace acl=metalrobo::inferred_acl;
    namespace lig=metalrobo::registered_ligaments::detail;
    const auto& q=fiber.provenance;const auto& r=fiber.row;
    const auto asMR=[](const auto& v){return mr_float4{v.x,v.y,v.z,v.w};};
    const auto start=lig::world(current.at(q.body0),asMR(r.firstLocalPoint));
    const auto end=lig::world(current.at(q.body1),asMR(r.secondLocalPoint));
    const auto axis=acl::detail::sub(end,start);
    const double length=acl::detail::norm(axis);
    require(std::isfinite(length)&&length>0.,"nonfinite transported ligament length");
    const auto direction=acl::detail::mul(axis,1./length);
    auto endpoint=[&](unsigned body,unsigned stable,unsigned triangle,mr_float4 p,const auto& world){
        auto ref=std::find_if(source.begin(),source.end(),[&](const auto& s){return s.body==body&&s.stableId==stable;});
        require(ref!=source.end(),"missing registered bone owner");
        const auto& mesh=posed.at(size_t(ref-source.begin()));
        require(mesh.body==body&&mesh.stableId==stable&&mesh.payloadSha256==ref->payloadSha256,"transported ligament source identity drift");
        const auto offset=size_t(triangle)*3;
        require(offset+2<ref->triangleIndices.size(),"invalid registered enthesis triangle");
        const unsigned a=ref->triangleIndices[offset],b=ref->triangleIndices[offset+1],c=ref->triangleIndices[offset+2];
        require(a==mesh.triangleIndices[offset]&&b==mesh.triangleIndices[offset+1]&&c==mesh.triangleIndices[offset+2],"transported ligament face changed");
        const auto old=lig::world(reference.at(body),p);
        auto w=acl::detail::barycentric(old,ref->worldVertices[a],ref->worldVertices[b],ref->worldVertices[c]);
        acl::V3 pos{},normal{};
        for(unsigned k=0;k<3;k++){
            unsigned vertex=mesh.triangleIndices[offset+k];
            for(unsigned j=0;j<3;++j){pos[j]+=w[k]*mesh.worldVertices[vertex][j];normal[j]+=w[k]*mesh.worldNormals[vertex][j];}
        }
        require(acl::detail::norm(acl::detail::sub(pos,world))<2e-6,"transported material enthesis left its registered face");
        const double mag=acl::detail::norm(normal);require(mag>0.&&std::isfinite(mag),"invalid transported normal");
        return acl::detail::mul(normal,1./mag);
    };
    const auto first=endpoint(q.body0,q.stable0,q.triangle0,asMR(r.firstLocalPoint),start);
    const auto last=endpoint(q.body1,q.stable1,q.triangle1,asMR(r.secondLocalPoint),end);
    const double facing0=acl::detail::dot(first,direction),facing1=-acl::detail::dot(last,direction);
    unsigned hits=0,firstHit=MR_INVALID_INDEX;
    for(const auto& mesh:posed)if(lig::segmentHitsSurface(start,end,mesh)){if(!hits)firstHit=mesh.stableId;++hits;}
    std::cout<<"registered_loaded_ligament name="<<q.name<<" side="<<q.side<<" reference_m="<<q.referenceLengthM
             <<" current_m="<<length<<" facing0="<<facing0<<" facing1="<<facing1<<" intersected_banks="<<hits
             <<" first_intersected_stable_id="<<firstHit<<std::endl;
    require(std::isfinite(facing0)&&std::isfinite(facing1)&&std::min(facing0,facing1)>=-1.e-4&&hits==0,"transported ligament requires cortical wrapping");
}

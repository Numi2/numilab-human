#pragma once
#include "NumiHumanPhysicalSkinChildRegisteredVolumeIntegration.hpp"
#include "NumiHumanPhysicalSkinAuthoringRoleConsumer.hpp"

static void verifyVolume(tissue::PhysicalSkinRegisteredSurfaceCSVFiles files,
                         const std::vector<tissue::PhysicalSkinDonorFrameRow>& frames) {
    using namespace metalrobo::human::coarse_tissue;
    std::string error;
    PhysicalSkinChildMassMappingBundle bundle;
    if (!loadCompiledSkinFatParentMassPacketCSV(setting("PARENT_PACKET_CSV"),bundle.parent,error) ||
        !loadCompiledSkinFatParentLabelsCSV(setting("PARENT_LABELS_CSV"),bundle.parent,error))
        throw std::runtime_error(error);
    files.parentRegistrationIsHistoricalProvenanceOnly=true;
    PhysicalSkinRegisteredSurfaceCSVSet surfaces;
    PhysicalSkinRegisteredSurfaceCSVReport sr;
    PhysicalSkinSurfaceParentRegistrationReport pr;
    if (!loadPhysicalSkinRegisteredSurfaceCSVSetForParent(files,bundle.parent,surfaces,sr,pr,error))
        throw std::runtime_error(error);
    std::vector<PhysicalSkinRegisteredBodyPose> poses(frames.size());
    for (size_t i=0; i<poses.size(); ++i) {
        const auto& f=frames[i];
        if (!f.available) throw std::runtime_error("Missing current or rest body pose");
        auto& p=poses[i];
        p.bodyIndex=i; p.binding=f.binding; p.available=true; p.arithmetic=f.arithmetic;
        p.currentOrigin=f.currentOrigin; p.restOrigin=f.restOrigin;
        p.currentLocalToWorld=f.currentR; p.restLocalToWorld=f.restR;
    }
    std::array<PhysicalSkinSourceDomainRegistration,3> domains;
    const std::array roles{PhysicalSkinSourceDomainRole::outerSurface, PhysicalSkinSourceDomainRole::innerSurface,
                           PhysicalSkinSourceDomainRole::deepSurface};
    for (size_t i=0;i<domains.size();++i) {
        const auto& src=files.currentSourceDomains[i];
        domains[i]={src.name,roles[i],src.assetSHA256,static_cast<uint32_t>(src.vertexCount),static_cast<uint32_t>(src.faceCount)};
    }
    std::vector<PhysicalSkinStableDeepAttachmentRequest> requests;
    std::ifstream input(setting("DEEP_ATTACHMENT_REQUESTS_CSV"));
    std::string line;
    if (!std::getline(input,line) || physical_skin_child_mass_detail::splitCSV(line) !=
        std::vector<std::string>{"stable_attachment_id","donor_body","stable_source_vertex","source_domain"})
        throw std::runtime_error("Invalid deep attachment input");
    while(std::getline(input,line)) {
        const auto v=physical_skin_child_mass_detail::splitCSV(line);
        if(v.size()!=4) throw std::runtime_error("Invalid deep attachment row");
        requests.push_back({std::stoull(v[0]),static_cast<uint32_t>(std::stoul(v[1])),
                           static_cast<uint32_t>(std::stoul(v[2])),v[3]});
    }
    if(requests.empty()) throw std::runtime_error("Missing deep attachments");
    PhysicalSkinRegisteredVolumeAssetFiles volume;
    volume.tetgenNode=setting("TETGEN_NODE");
    volume.tetgenElements=setting("TETGEN_ELEMENTS");
    volume.childLabels=setting("CHILD_LABELS");
    volume.combinedNodeMapCSV=setting("COMBINED_NODE_MAP_CSV");
    volume.facetAncestryCSV=setting("FACET_ANCESTRY_CSV");
    volume.expectedSHA256={setting("TETGEN_NODE_SHA256"),setting("TETGEN_ELEMENTS_SHA256"),
                           setting("CHILD_LABELS_SHA256"),setting("COMBINED_NODE_MAP_SHA256"),setting("FACET_ANCESTRY_SHA256")};
    volume.registrationDomain=PhysicalSkinRegisteredVolumeRegistrationDomain::currentFractionalReference;
    PhysicalSkinRegisteredVolumeImportSidecars sidecars;
    PhysicalSkinRegisteredVolumeFileImportReport report;
    if(!importPhysicalSkinRegisteredVolumeFiles(volume,setting("COORDINATE_FRAME_SHA256"),poses.size(),
          std::stod(setting("MAXIMUM_MASS_TRANSPORT_DISTANCE_M")),files.maxParentRegistrationDistanceM,
          "",0u,0u,surfaces,requests,poses,bundle,sidecars,report,error,domains))
        throw std::runtime_error(error);
    std::array<PhysicalSkinAuthoringFullWeightsCSV,3> weights;
    size_t layerIndex=0;
    for (std::string name : {"O","I","D"}) {
        auto& weight=weights[layerIndex++];
        weight.path=setting("ROLE_"+name+"_FULL86_CSV");
        weight.expectedSHA256=setting("ROLE_"+name+"_FULL86_SHA256");
    }
    PhysicalSkinAuthoringRoleConsumerState roleState;
    PhysicalSkinAuthoringRoleConsumerReport roleReport;
    if (!preparePhysicalSkinAuthoringRoleConsumerState(bundle,surfaces,sidecars,poses,weights,roleState,roleReport,error))
        throw std::runtime_error(error);
    std::cout<<"{\"mapped_volume_nodes\":"<<report.registeredVolume.volumeNodeCount
             <<",\"volume_cells\":"<<bundle.childTetrahedra.size()<<",\"registered_facets\":"<<report.facetAncestry.registeredRows
             <<",\"deep_attachment_requests\":"<<requests.size()<<",\"source_domains\":"<<report.registeredVolume.sourceDomainCount
             <<",\"maximum_current_parent_error_m\":"<<report.registeredVolume.maximumCurrentParentReconstructionErrorM
             <<",\"weight_bindings\":"<<roleReport.bindingCount
             <<",\"weight_mapping_error_m\":"<<*std::max_element(roleReport.maximumWorldLocalReconstructionErrorM.begin(),roleReport.maximumWorldLocalReconstructionErrorM.end())
             <<",\"domain_sha256\":\""<<report.operationalReferenceDomainSHA256<<"\"}\n";
}

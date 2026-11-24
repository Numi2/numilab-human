#pragma once
#include "NumiHumanLoadedSkinPoseData.inc"

namespace numiLoadedSkin {
using Point=std::array<double,3>;
inline bool enabled() {return std::getenv("NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE")!=nullptr;}
inline const std::vector<Point>& points() {
    static const std::vector<Point> result=[] {
        const char* file=std::getenv("NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE");
        require(file && loadedKneeSHA256Hex(loadedKneeFileSHA256(file))==numiLoadedData::nodeSHA,
                "loaded current-coordinate TetGen identity mismatch");
        std::ifstream input(file);
        size_t count=0; unsigned dimension=0,attributes=0,markers=0;
        require(bool(input>>count>>dimension>>attributes>>markers) && count==numiLoadedData::nodes &&
                dimension==3 && !attributes && !markers,"unexpected current material node header");
        std::vector<Point> out(count);
        for(size_t i=0;i<count;++i) {
            size_t id=0;auto& p=out[i];
            require(bool(input>>id>>p[0]>>p[1]>>p[2])&&id==i+1,"invalid current material node identity");
            for(double v:p)require(std::isfinite(v),"nonfinite loaded material position");
            require(p[2]>=0.,"loaded material crosses the registered bed plane");
        }
        input>>std::ws;require(input.eof(),"extra rows in loaded material pose");
        return out;
    }();
    return result;
}
inline void sourceFrames(std::vector<MRBodyStateGPU>& states) {
    if(!enabled())return;
    require(states.size()==std::size(numiLoadedData::reference),"reference donor count changed");
    for(const auto& t:numiLoadedData::target){
        const auto& b=states[t.id];
        const double miss=std::hypot(double(b.position.x)-t.p[0],double(b.position.y)-t.p[1],double(b.position.z)-t.p[2]);
        const double err=std::max({std::abs(double(b.orientation.x)-t.q[0]),std::abs(double(b.orientation.y)-t.q[1]),
                                  std::abs(double(b.orientation.z)-t.q[2]),std::abs(double(b.orientation.w)-t.q[3])});
        require(miss<=2e-6&&err<=2e-6,"loaded skin donor pose differs from measured skeleton at body "+std::to_string(t.id)+" position m "+std::to_string(miss));
    }
    for(const auto& r:numiLoadedData::reference) {
        auto& b=states[r.id];b.position={r.p[0],r.p[1],r.p[2],b.position.w};b.orientation={r.q[0],r.q[1],r.q[2],r.q[3]};
    }
}
template<class Ground>
bool align(std::span<const float> q,const Ground& bed,metalrobo::human::coarse_tissue::FEMOuterGroundAlignment& result,std::string& error){
    if(bed.groundNormalX!=0 || bed.groundNormalY!=0 || bed.groundNormalZ!=1 || bed.groundPointZ!=0){
        error="physical bed differs from the loaded material boundary";return false;}
    const auto& p=points();double minimum=std::numeric_limits<double>::infinity();
    for(const auto& v:p) minimum=std::min(minimum,v[2]);
    if(minimum!=0.){error="loaded material has no first contact";return false;}
    result.q.assign(q.begin(),q.end());result.outerNodeCount=numiLoadedData::outerNodes;
    result.initialMinimumGapM=result.alignedMinimumGapM=minimum;result.legacyNHCNTWitnessCount=bed.contactCount;return true;
}
template<class Object,class Bundle,class Sidecars>
void install(Object& object,const Bundle& bundle,const Sidecars& sidecars){
    if(!enabled())return;
    const auto& p=points();const auto& x=object.femReferenceNodes;
    require(x.size()==p.size() && object.femNodes.size()==p.size() &&
            object.tetrahedra.size()==numiLoadedData::cells,"current pose does not address this FEM volume");
    const char* ref=std::getenv("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE_SHA256");
    const char* ele=std::getenv("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_ELEMENTS_SHA256");
    require(ref&&ele&&std::string_view(ref)==numiLoadedData::referenceSHA&&std::string_view(ele)==numiLoadedData::elementsSHA,
            "current pose reference and cell identities differ");
    // The Matter compiler represents a registered coordinate as its exact float
    // reference plus a high and low float displacement. Check the coordinates
    // used by the geometry gates against that same representation before upload.
    for (size_t i=0;i<p.size();++i) {
        for (unsigned k=0;k<3;++k) {
            const double current=p[i][k];
            if (current==double(float(current))) continue;
            const double reference=x[i][k];
            require(object.femReferenceDisplacementGradient && reference==double(float(reference)),
                    "precise loaded position requires a Float32 reference and compensated displacement");
            const double displacement=current-reference;
            const float high=static_cast<float>(displacement);
            const float low=static_cast<float>(displacement-double(high));
            require(std::isfinite(high) && std::isfinite(low) &&
                    reference+(double(high)+double(low))==current,
                    "loaded coordinate cannot be recovered from the registered physical state");
        }
    }
    auto geometry=[](const auto& xyz,const auto& id){
        Point u{},v{},w{};
        for(unsigned k=0;k<3;k++){
            u[k]=xyz[id[1]][k]-xyz[id[0]][k];v[k]=xyz[id[2]][k]-xyz[id[0]][k];w[k]=xyz[id[3]][k]-xyz[id[0]][k];
        }
        double det=u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0]);
        return std::pair(det,numiHumanCellShape::normalizedDeterminant(det,u,v,w));
    };
    for(const auto& tet:object.tetrahedra){
        const auto [v,shape]=geometry(p,tet.nodes);const auto [rest,unused]=geometry(x,tet.nodes);
        require(v>6e-18 && shape>=1e-5 && rest>0 && v/rest>=.4 && v/rest<=2.5,"loaded skin/fat cell fails installed geometry");
    }
    size_t touch=0;
    require(bundle.boundaryMap.outer.size()==numiLoadedData::outerNodes,"registered outer boundary row count changed");
    for(auto n:bundle.boundaryMap.outer)touch+=p.at(n)[2]==0.;
    require(touch>0,"no registered physical skin point touches the bed");
    require(sidecars.deepAttachments.size()==std::size(numiLoadedData::anchors),"deep attachment count changed");
    for(const auto& a:sidecars.deepAttachments){
        auto it=std::find_if(std::begin(numiLoadedData::anchors),std::end(numiLoadedData::anchors),[&](const auto& b){return b.stable==a.stableAttachmentID;});
        require(it!=std::end(numiLoadedData::anchors)&&it->node==a.volumeNodeRow&&it->donor==a.donorBody,"unmatched physical attachment");
        const auto& actual=p.at(a.volumeNodeRow);
        require(std::hypot(actual[0]-it->p[0],actual[1]-it->p[1],actual[2]-it->p[2])<=2e-7,"loaded material attachment target mismatch");
    }
    object.femNodes=p;
    std::cout<<"registered_loaded_material_install=PASS cells="<<object.tetrahedra.size()<<" nodes="<<p.size()
             <<" outer_contact_nodes="<<touch<<" anchors="<<sidecars.deepAttachments.size()
             <<" pose_sha256="<<numiLoadedData::nodeSHA<<" reference_and_mass_changed=0 initial_forces_applied=0"<<std::endl;
}
}

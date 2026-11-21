#pragma once
// The constructor subsequently verifies every posed donor against native FK.
// The coordinates inspected here belong to that same hashed physical O table.
template<class GroundHeader>
bool checkRegisteredPhysicalOuterAtSeed(std::span<const float> q,const GroundHeader& ground,
    metalrobo::human::coarse_tissue::FEMOuterGroundAlignment& alignment,std::string& error) {
    using namespace metalrobo::human::coarse_tissue;
    const auto fail=[&](std::string why){error=std::move(why);return false;};
    const char* path=std::getenv("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_O_NODES_CSV");
    const char* pin=std::getenv("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_O_NODES_SHA256");
    if(!path||!pin||loadedKneeSHA256Hex(loadedKneeFileSHA256(path))!=pin)
        return fail("physical O first-touch table does not match its source hash");
    const std::array<double,3> normal{ground.groundNormalX,ground.groundNormalY,ground.groundNormalZ};
    const std::array<double,3> plane{ground.groundPointX,ground.groundPointY,ground.groundPointZ};
    const double length=std::hypot(normal[0],normal[1],normal[2]);
    if(!std::isfinite(length)||std::abs(length-1.0)>1e-5||
       !std::all_of(plane.begin(),plane.end(),[](double v){return std::isfinite(v);}))
        return fail("invalid registered bed plane");
    std::ifstream input(path);std::string line;uint32_t rows=0;
    if(!std::getline(input,line)||!line.starts_with("vertex,donor,wx,wy,wz,"))
        return fail("invalid registered O node header");
    double minimum=std::numeric_limits<double>::infinity();
    while(std::getline(input,line)){
        if(line.size()>4096||rows==std::numeric_limits<uint32_t>::max())return fail("overlarge physical O row or surface");
        std::string_view record(line);
        auto field=[&](auto& number){
            const auto sep=record.find(',');
            if(sep==record.npos||sep==0)return false;
            const auto* begin=record.data();const auto* end=begin+sep;
            auto result=std::from_chars(begin,end,number);record.remove_prefix(sep+1);
            return result.ec==std::errc()&&result.ptr==end;
        };
        uint32_t id,donor;
        if(!field(id)||!field(donor)||id!=rows)return fail("unordered physical O row");
        double gap=0.;
        for(unsigned a=0;a<3;++a){
            double value=0.;
            if(!field(value)||!std::isfinite(value)||double(float(value))!=value)return fail("invalid physical O coordinate");
            gap+=(value-plane[a])*normal[a];
        }
        minimum=std::min(minimum,gap/length);++rows;
    }
    if(!input.eof()||rows<3||!std::isfinite(minimum)||minimum<1e-8||minimum>1e-6)
        return fail("registered O surface outside native first-touch interval, gap m="+std::to_string(minimum));
    FEMOuterGroundAlignment checked;checked.q.assign(q.begin(),q.end());checked.outerNodeCount=rows;
    checked.initialMinimumGapM=checked.alignedMinimumGapM=minimum;
    checked.legacyNHCNTWitnessCount=ground.contactCount;alignment=std::move(checked);
    error.clear();return true;
}

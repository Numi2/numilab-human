#pragma once
// Immutable source sections prepared by walking the registered dermis and fat tetrahedra.
#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct NumiRegisteredBedLength {
    bool present=false;
    unsigned row=0, geometry=0;
    double x=0,y=0,z=0,area=0,skin=0,fat=0;
};

static std::vector<NumiRegisteredBedLength> numiReadBedLengths(const std::filesystem::path& input,
    size_t count,const std::string& identity,size_t expected) {
    if(loadedKneeSHA256Hex(loadedKneeFileSHA256(input))!=identity)
        throw std::runtime_error("registered bed material digest mismatch");
    std::ifstream file(input); std::string line;
    if(!std::getline(file,line) || line!="vertex,row,source_geometry_index,x_m,y_m,z_m,area_m2,skin_m,fat_m,thickness_source_vertex")
        throw std::runtime_error("unknown registered bed footprint columns");
    std::vector<NumiRegisteredBedLength> records(count);
    size_t seen=0;
    while(std::getline(file,line)){
        std::replace(line.begin(),line.end(),',',' ');
        std::istringstream values(line);
        unsigned v=0,source=0; NumiRegisteredBedLength r;
        if(!(values>>v>>r.row>>r.geometry>>r.x>>r.y>>r.z>>r.area>>r.skin>>r.fat>>source) ||
           v>=count || records[v].present || source!=std::numeric_limits<unsigned>::max() ||
           !(r.skin>0 && r.fat>0 && r.area>0) ||
           !std::isfinite(r.x+r.y+r.z+r.skin+r.fat+r.area))
            throw std::runtime_error("invalid registered bed material row");
        values>>std::ws;
        if(!values.eof()) throw std::runtime_error("extra registered bed material fields");
        r.present=true;records[v]=r;++seen;
    }
    if(!file.eof() || seen!=expected) throw std::runtime_error("incomplete registered bed material");
    return records;
}

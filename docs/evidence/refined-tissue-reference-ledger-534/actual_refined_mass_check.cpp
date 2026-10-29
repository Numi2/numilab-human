#include "NumiHumanParentReferenceMassTransfer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

namespace {
using Point = std::array<double,3>;
using Tet = std::array<std::uint32_t,4>;
struct Labels {
    std::vector<std::uint32_t> material, donor, region, stableFace;
};
bool splitCsv(const std::string& line, std::vector<std::string>& fields) {
    fields.clear();
    std::stringstream stream(line);
    std::string field;
    while(std::getline(stream,field,','))fields.push_back(field);
    for(auto& value:fields)if(!value.empty()&&value.back()=='\r')value.pop_back();
    return !fields.empty();
}
bool readNodes(const std::string& path, std::vector<Point>& points, std::string& error) {
    std::ifstream in(path);
    std::size_t count=0u;unsigned dimension=0u,attributes=0u,markers=0u;
    if(!in||!(in>>count>>dimension>>attributes>>markers)||count<4u||dimension!=3u||
       attributes!=0u||markers!=0u){error="invalid TetGen node header: "+path;return false;}
    std::vector<Point> next(count);
    std::vector<std::uint8_t> seen(count,0u);
    for(std::size_t row=0;row<count;++row){
        std::uint64_t id=0u;Point point{};
        if(!(in>>id>>point[0]>>point[1]>>point[2])||id==0u||id>count||
           seen[id-1u]!=0u){error="invalid or duplicate TetGen node row: "+path;return false;}
        seen[id-1u]=1u;next[id-1u]=point;
    }
    points=std::move(next);return true;
}
bool readTets(const std::string& path, std::vector<Tet>& tets, std::vector<std::uint32_t>& attribute,
              std::string& error) {
    std::ifstream in(path);
    std::size_t count=0u;unsigned corners=0u,attributes=0u;
    if(!in||!(in>>count>>corners>>attributes)||count==0u||corners!=4u||attributes!=1u){
        error="invalid TetGen element header: "+path;return false;
    }
    std::vector<Tet> next(count);
    std::vector<std::uint32_t> nextAttribute(count);
    std::vector<std::uint8_t> seen(count,0u);
    for(std::size_t row=0;row<count;++row){
        std::uint64_t id=0u;Tet tet{};std::uint64_t material=0u;
        if(!(in>>id>>tet[0]>>tet[1]>>tet[2]>>tet[3]>>material)||id==0u||id>count||
           seen[id-1u]!=0u||material>std::numeric_limits<std::uint32_t>::max()){
            error="invalid or duplicate TetGen element row: "+path;return false;
        }
        seen[id-1u]=1u;
        for(auto& node:tet){if(node==0u||node>std::numeric_limits<std::uint32_t>::max()){
            error="invalid TetGen node ID in: "+path;return false;}--node;}
        next[id-1u]=tet;nextAttribute[id-1u]=static_cast<std::uint32_t>(material);
    }
    tets=std::move(next);attribute=std::move(nextAttribute);return true;
}
bool readLabels(const std::string& path, Labels& labels, std::string& error, const std::string_view rowColumn="child_tet_row") {
    std::ifstream in(path);std::string line;std::vector<std::string> fields;
    if(!in||!std::getline(in,line)||!splitCsv(line,fields)){
        error="missing label CSV header: "+path;return false;
    }
    std::unordered_map<std::string,std::size_t> column;
    for(std::size_t i=0;i<fields.size();++i)column.emplace(fields[i],i);
    const std::string rowKey(rowColumn);
    for(const auto* key:{"material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"})
        if(!column.contains(key)){error="label CSV missing required column "+std::string(key);return false;}
    if(!column.contains(rowKey)){error="label CSV missing required row column "+rowKey;return false;}
    std::vector<std::uint32_t> material,donor,region,stable;
    std::vector<std::uint8_t> seen;
    while(std::getline(in,line)){
        if(line.empty())continue;
        if(!splitCsv(line,fields)||fields.size()!=column.size()){error="malformed label CSV row: "+path;return false;}
        try{
            const auto row=static_cast<std::size_t>(std::stoull(fields[column.at(rowKey)]));
            const auto m=static_cast<std::uint32_t>(std::stoul(fields[column.at("material_index")]));
            const auto d=static_cast<std::uint32_t>(std::stoul(fields[column.at("geometric_donor_body")]));
            const auto r=static_cast<std::uint32_t>(std::stoul(fields[column.at("anatomical_region_id")]));
            const auto s=static_cast<std::uint32_t>(std::stoul(fields[column.at("stable_source_face_id")]));
            if(row>10'000'000u){error="unbounded label row index";return false;}
            if(material.size()<=row){material.resize(row+1u);donor.resize(row+1u);region.resize(row+1u);stable.resize(row+1u);seen.resize(row+1u,0u);}
            if(seen[row]!=0u){error="duplicate label row";return false;}
            seen[row]=1u;material[row]=m;donor[row]=d;region[row]=r;stable[row]=s;
        }catch(...){error="invalid numeric label CSV row: "+path;return false;}
    }
    if(material.empty()||std::any_of(seen.begin(),seen.end(),[](auto x){return x!=1u;})){
        error="label CSV rows are not contiguous";return false;
    }
    labels={std::move(material),std::move(donor),std::move(region),std::move(stable)};
    return true;
}
bool readMass(const std::string& path, const Labels& source,
              const std::vector<Tet>& sourceTets, std::vector<double>& parentReferenceVolume,
              std::vector<double>& mass, std::string& error) {
    std::ifstream in(path);std::string line;std::vector<std::string> fields;
    if(!in||!std::getline(in,line)||!splitCsv(line,fields)){error="missing parent mass CSV header";return false;}
    std::unordered_map<std::string,std::size_t> column;
    for(std::size_t i=0;i<fields.size();++i)column.emplace(fields[i],i);
    for(const auto* key:{"parent_tet_row","material_index","donor_body","parent_mass_kg",
                          "parent_reference_volume_m3","parent_node0","parent_node1","parent_node2","parent_node3"})
        if(!column.contains(key)){error="parent mass CSV missing column "+std::string(key);return false;}
    std::vector<double> next,nextVolume;
    std::vector<std::uint8_t> seen;
    while(std::getline(in,line)){
        if(line.empty())continue;
        if(!splitCsv(line,fields)||fields.size()!=column.size()){error="malformed parent mass CSV row";return false;}
        try{
            const auto row=static_cast<std::size_t>(std::stoull(fields[column.at("parent_tet_row")]));
            const auto material=static_cast<std::uint32_t>(std::stoul(fields[column.at("material_index")]));
            const auto donor=static_cast<std::uint32_t>(std::stoul(fields[column.at("donor_body")]));
            const double value=std::stod(fields[column.at("parent_mass_kg")]);
            const double volume=std::stod(fields[column.at("parent_reference_volume_m3")]);
            Tet parentTet{};
            for(unsigned corner=0;corner<4u;++corner)
                parentTet[corner]=static_cast<std::uint32_t>(std::stoul(fields[column.at("parent_node"+std::to_string(corner))]));
            if(row>10'000'000u){error="unbounded parent mass row";return false;}
            if(next.size()<=row){next.resize(row+1u);nextVolume.resize(row+1u);seen.resize(row+1u,0u);}
            if(seen[row]!=0u||row>=source.material.size()||row>=sourceTets.size()||
               material!=source.material[row]||donor!=source.donor[row]||parentTet!=sourceTets[row]||
               !std::isfinite(value)||value<=0.0||!std::isfinite(volume)||volume<=0.0){
                error="parent mass row does not match source topology/labels or is invalid at row "+std::to_string(row);return false;
            }
            seen[row]=1u;next[row]=value;nextVolume[row]=volume;
        }catch(...){error="invalid numeric parent mass CSV row";return false;}
    }
    if(next.empty()||std::any_of(seen.begin(),seen.end(),[](auto x){return x!=1u;})){
        error="parent mass rows are not contiguous";return false;
    }
    mass=std::move(next);parentReferenceVolume=std::move(nextVolume);return true;
}
bool readSourceRows(const std::string& path,
        std::vector<ConservativeSpatialMassTransportParentNodeSourceMass>& rows,std::string& error) {
    std::ifstream in(path);std::string line;std::vector<std::string> fields;
    if(!in||!std::getline(in,line)||!splitCsv(line,fields)){error="missing source contribution CSV header";return false;}
    std::unordered_map<std::string,std::size_t> column;
    for(std::size_t i=0;i<fields.size();++i)column.emplace(fields[i],i);
    for(const auto* key:{"object_local_node","material_index","donor_body","mass_kg"})
        if(!column.contains(key)){error="source contribution CSV missing column "+std::string(key);return false;}
    std::vector<ConservativeSpatialMassTransportParentNodeSourceMass> next;
    while(std::getline(in,line)){
        if(line.empty())continue;
        if(!splitCsv(line,fields)||fields.size()!=column.size()){error="malformed source contribution CSV row";return false;}
        try{
            next.push_back({.parentNodeIndex=static_cast<std::uint32_t>(std::stoul(fields[column.at("object_local_node")])),
                .materialIndex=static_cast<std::uint32_t>(std::stoul(fields[column.at("material_index")])),
                .sourceDonorBodyIndex=static_cast<std::uint32_t>(std::stoul(fields[column.at("donor_body")])),
                .massKg=std::stod(fields[column.at("mass_kg")])});
        }catch(...){error="invalid numeric source contribution row";return false;}
    }
    rows=std::move(next);return !rows.empty();
}
std::string fail(const std::string& text){std::cerr<<"FAIL "<<text<<"\n";return text;}
}


int main(int argc,char** argv) {
 if(argc!=10){std::cerr<<"usage: source.node source.ele source.labels mass.tets.csv source.csv child.node child.ele ledger.csv donor-output.csv\n";return 2;}
 std::vector<Point> sp,cp;std::vector<Tet> st,ct;std::vector<std::uint32_t> sm,cm;Labels labels;
 std::vector<double> mass,pv;std::vector<ConservativeSpatialMassTransportParentNodeSourceMass> src;std::string error;
 if(!readNodes(argv[1],sp,error)||!readTets(argv[2],st,sm,error)||!readLabels(argv[3],labels,error)||
    !readMass(argv[4],labels,st,pv,mass,error)||!readSourceRows(argv[5],src,error)||
    !readNodes(argv[6],cp,error)||!readTets(argv[7],ct,cm,error)){fail(error);return 3;}
 if(sm!=labels.material){fail("source material mismatch");return 4;}
 std::vector<std::uint32_t> offsets{0},parent;std::vector<double> fraction,measure,recordedMass,recordedVol,geom;
 std::ifstream ledger(argv[8]);std::string line;std::vector<std::string> f;
 if(!std::getline(ledger,line)){fail("missing ledger header");return 4;}
 if(!line.empty()&&line.back()=='\r')line.pop_back();
 if(line!="child_tet_row,parent_tet_row,source_fraction,parent_reference_measure_m3,child_mass_kg,child_reference_volume_m3,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id"){fail("unexpected ledger header");return 4;}
 while(std::getline(ledger,line)){
  if(line.empty())continue;
  if(!splitCsv(line,f)||f.size()!=10){fail("invalid ledger row");return 4;}
  const auto c=parent.size();const auto p=static_cast<std::uint32_t>(std::stoul(f[1]));
  if(std::stoull(f[0])!=c||c>=ct.size()||p>=mass.size()||std::stoul(f[6])!=cm[c]||
     cm[c]!=labels.material[p]||std::stoul(f[7])!=labels.donor[p]||std::stoul(f[8])!=labels.region[p]||
     std::stoul(f[9])!=labels.stableFace[p]){fail("ledger identity mismatch");return 4;}
  const auto v=parentReferenceTetVolumeM3(ct[c],cp);
  const double frac=std::stod(f[2]),m=std::stod(f[3]),kg=std::stod(f[4]),rv=std::stod(f[5]);
  if(!std::isfinite(frac)||frac<=0||std::abs(v-rv)>1e-11*rv||
     std::abs(m-pv[p]*frac)>1e-15||std::abs(kg-mass[p]*frac)>1e-13){fail("ledger source fraction or canonical geometry mismatch");return 4;}
  parent.push_back(p);offsets.push_back(static_cast<std::uint32_t>(parent.size()));
  fraction.push_back(frac);measure.push_back(m);recordedMass.push_back(kg);recordedVol.push_back(rv);geom.push_back(v);
 }
 if(parent.size()!=ct.size()){fail("ledger cell count mismatch");return 4;}
 std::vector<double> lower(17u,0),upper(17u,0);lower[15]=900;upper[15]=1100;lower[16]=850;upper[16]=950;
 ParentReferenceMassTransferInput input{pv,mass,labels.material,labels.donor,geom,cm,offsets,parent,measure,lower,upper};
 ParentReferenceMassTransferResult out;
 if(!assembleParentReferenceMassTransfer(input,out,error)){fail(error);return 5;}
 for(std::size_t c=0;c<parent.size();++c)if(std::abs(out.childMassKg[c]-recordedMass[c])>1e-13){fail("native mass differs from authored ledger");return 5;}
 std::vector<std::uint32_t> so,sd;std::vector<double> sk;
 if(!assembleConservativeSpatialMassTransportParentSourceDonorSplits(mass,sm,st,src,1e-10,5e-4,so,sd,sk,error)){fail(error);return 6;}
 std::ofstream donor(argv[9]);donor<<"child_tet_row,material_index,source_donor_body,mass_kg\n";
 donor<<std::setprecision(17);
 std::map<std::pair<std::uint32_t,std::uint32_t>,long double> before,after;
 for(const auto& row:src)before[{row.materialIndex,row.sourceDonorBodyIndex}]+=row.massKg;
 std::size_t sourceRows=0;
 for(std::size_t c=0;c<parent.size();++c){
  const auto p=parent[c];long double total=0;
  for(std::uint32_t j=so[p];j<so[p+1u];++j){
   const double kg=sk[j]*fraction[c];total+=kg;after[{cm[c],sd[j]}]+=kg;
   donor<<c<<','<<cm[c]<<','<<sd[j]<<','<<kg<<'\n';++sourceRows;
  }
  if(std::abs(static_cast<double>(total)-out.childMassKg[c])>1e-13){fail("child source donor rows do not close");return 6;}
 }
 donor.close();double sourceError=0;
 for(const auto& [key,kg]:before)sourceError=std::max(sourceError,std::abs(static_cast<double>(kg-after[key])));
 if(sourceError>1e-10||before.size()!=after.size()){fail("source donor group closure failed");return 6;}
 auto wrong=input;wrong.childParentReferenceMeasureM3=geom;ParentReferenceMassTransferResult sentinel;sentinel.childMassKg={123};
 if(assembleParentReferenceMassTransfer(wrong,sentinel,error)||sentinel.childMassKg!=std::vector<double>{123}){fail("wrong operational measure accepted or output mutated");return 7;}
 auto cross=cm;cross[0]=cross[0]==15?16:15;wrong=input;wrong.childMaterialIds=cross;
 if(assembleParentReferenceMassTransfer(wrong,sentinel,error)||sentinel.childMassKg!=std::vector<double>{123}){fail("cross material ancestry accepted");return 7;}
 auto missing=measure;missing[0]*=.5;wrong=input;wrong.childParentReferenceMeasureM3=missing;
 if(assembleParentReferenceMassTransfer(wrong,sentinel,error)||sentinel.childMassKg!=std::vector<double>{123}){fail("missing parent measure accepted");return 7;}
 long double total=0;for(double kg:out.childMassKg)total+=kg;
 std::cout<<std::setprecision(17)<<"PASS child_nodes="<<cp.size()<<" child_cells="<<ct.size()
  <<" total_mass_kg="<<static_cast<double>(total)<<" max_parent_mass_closure_kg="<<out.maximumParentMassClosureErrorKg
  <<" max_parent_measure_closure_m3="<<out.maximumParentMeasureClosureErrorM3
  <<" source_groups="<<before.size()<<" source_contribution_input_rows="<<src.size()<<" child_source_donor_rows="<<sourceRows
  <<" source_group_closure_max_kg="<<sourceError
  <<" density_min_kg_m3="<<*std::min_element(out.childImpliedDensityKgM3.begin(),out.childImpliedDensityKgM3.end())
  <<" density_max_kg_m3="<<*std::max_element(out.childImpliedDensityKgM3.begin(),out.childImpliedDensityKgM3.end())
  <<" negative_operational_measure=REJECTED negative_cross_material=REJECTED negative_missing_measure=REJECTED"
  <<" unprojected_source_control=true physical_admission=false runtime=0 accepted_steps=0\n";
 return 0;
}

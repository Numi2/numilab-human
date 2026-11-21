#pragma once
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <metalrobo/NumiHumanTissueMass.hpp>

// Use the current composite body's residual inertia after all existing full-scene tissue debits.
template<class Offset>
void checkPelvicMassLedger(const char* filename,const metalrobo::EngineModel& current,Offset offset) {
    std::ifstream input(filename);
    if(!input) throw std::runtime_error("could not open pelvic mass ledger");
    std::vector<metalrobo::NumiHumanTissueMassNode> nodes;
    std::vector<metalrobo::NumiHumanTissueMassContribution> contributions;
    std::string line;
    while(std::getline(input,line)) {
        std::istringstream row(line);
        std::string kind,extra;
        std::uint32_t node,donor;
        double mass;
        std::array<double,3> local{};
        if(!(row>>kind>>node>>donor>>mass>>local[0]>>local[1]>>local[2]) || row>>extra ||
           donor>=current.bodies.size()) throw std::runtime_error("invalid pelvic mass row");
        auto rebase=offset(donor);
        for(unsigned axis=0;axis<3u;++axis) local[axis]-=rebase[axis];
        if(kind=="node") nodes.push_back({node,donor,mass,local});
        else if(kind=="source") contributions.push_back({node,donor,mass,local});
        else throw std::runtime_error("unknown pelvic mass record");
    }
    if(input.bad() || nodes.empty() || contributions.empty()) throw std::runtime_error("incomplete pelvic mass ledger");
    auto result=metalrobo::compileNumiHumanTissueMassPartition(current.bodies,nodes,contributions);
    if(!result.succeeded()) throw std::runtime_error("pelvic residual donor partition rejected: "+result.error);
    metalrobo::EngineModel candidate;
    std::string error;
    if(!metalrobo::rebaseNumiHumanTissueMassPartition(current,result.partitions,candidate,error))
        throw std::runtime_error("pelvic donor COM rebase rejected: "+error);
    for(const auto& p:result.partitions) {
        std::cout<<std::setprecision(17)<<"pelvic_material_candidate_partition body="<<p.donorBody
                 <<" source_kg="<<p.sourceBody.massAndInverseMass.x<<" debit_kg="<<p.tissueMassKg
                 <<" remaining_kg="<<candidate.bodies[p.donorBody].massAndInverseMass.x
                 <<" packed_moment_error="<<p.packedMomentRelativeError<<"\n";
    }
    std::cout<<"pelvic_mass_preflight=passed nodes="<<nodes.size()<<" sources="<<contributions.size()
             <<" applied_to_live_world=false\n";
}

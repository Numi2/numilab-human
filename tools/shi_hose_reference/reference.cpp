#include "shi_hose_dopri.hpp"
#include "shi_hose_observables.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>

using namespace numi_shi_hose_reference;
int main(int argc,char** argv) {
    try {
        if(argc!=5)throw std::runtime_error("usage: shi_hose_reference output.csv seconds output_interval tolerance");
        const double duration=std::stod(argv[2]),interval=std::stod(argv[3]),tolerance=std::stod(argv[4]);
        if(!std::isfinite(duration)||!std::isfinite(interval)||!std::isfinite(tolerance)||duration<10.0||interval<=0||tolerance<=0||duration/interval>10000000)throw std::runtime_error("invalid ten-cycle reference request");
        const auto samples=static_cast<std::uint64_t>(std::llround(duration/interval));
        if(std::abs(samples*interval-duration)>1e-10)throw std::runtime_error("invalid ten-cycle reference request");
        Integrator run;run.relative_tolerance=tolerance;run.absolute_tolerance=tolerance;
        std::ofstream out(argv[1]);if(!out)throw std::runtime_error("cannot open CSV output");out<<std::setprecision(17)<<"time_s";
        for(const auto& o:observables)out<<','<<o.name;
        for(unsigned i=0;i<20;++i)out<<",native_hydraulic_"<<i;
        out<<'\n';
        double initial_storage=0,max_storage_error=0;State previous_cycle{};double last_cycle_difference=0;
        for(std::uint64_t row=0;row<=samples;++row) {
            run.advance(row*interval);State dy;Values values;evaluate(run.time,run.state,dy,&values);
            const auto hydraulic=native_hydraulic_state(values);double storage=std::accumulate(hydraulic.begin(),hydraulic.begin()+10,0.0);
            if(row==0)initial_storage=storage;max_storage_error=std::max(max_storage_error,std::abs(storage-initial_storage));
            out<<run.time;for(const auto& o:observables){const double v=values[o.index]*o.scale;if(!std::isfinite(v))throw std::runtime_error("nonfinite source observable");out<<','<<v;}
            for(double v:hydraulic)out<<','<<v;out<<'\n';
            if(std::abs(run.time-std::round(run.time))<1e-12){last_cycle_difference=0;for(unsigned i=0;i<state_count;++i)last_cycle_difference=std::max(last_cycle_difference,std::abs(run.state[i]-previous_cycle[i])/(1+std::abs(run.state[i])));previous_cycle=run.state;}
        }
        if(!out.flush())throw std::runtime_error("CSV output failed");
        std::cout<<std::setprecision(17)<<"{\"schema\":\"NumiHuman.CellML-reference-run.v1\",\"source_revision\":\""<<revision<<"\",\"boundary\":\"source_hydraulic_reproduction_only\",\"seconds\":"<<duration<<",\"output_interval_seconds\":"<<interval<<",\"source_coordinate_rtol\":"<<tolerance<<",\"source_coordinate_atol\":"<<tolerance<<",\"samples\":"<<samples+1<<",\"accepted_internal_steps\":"<<run.accepted_steps<<",\"rejected_internal_steps\":"<<run.rejected_steps<<",\"rhs_evaluations\":"<<run.evaluations<<",\"minimum_accepted_dt\":"<<run.smallest_accepted_step<<",\"maximum_local_error_ratio\":"<<run.largest_accepted_error_ratio<<",\"initial_hydraulic_storage_m3\":"<<initial_storage<<",\"maximum_storage_error_m3\":"<<max_storage_error<<",\"last_cycle_scaled_state_difference\":"<<last_cycle_difference<<",\"initial_states_modified\":false,\"native_equations_reused\":false,\"absolute_vascular_volume_calibrated\":false}\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"reference_failure="<<e.what()<<'\n';return 1;}
}

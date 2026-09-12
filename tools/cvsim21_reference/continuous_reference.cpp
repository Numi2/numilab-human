#include "dopri.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace cvsim21_reference;
static double sum_volume(const Data_vector& point) {
  double sum=0.;for(double v:point.v) sum+=v;return sum;
}
static void header(std::ostream& file) {
  file << "time_s";
  for(int i=0;i<21;++i)file<<",P_"<<i<<"_mmHg";
  for(int i=0;i<21;++i)file<<",V_"<<i<<"_mL";
  for(int i=0;i<24;++i)file<<",Q_"<<i<<"_mL_per_s";
  file<<",volume_sum_mL,source_atrial_phase_s,source_ventricular_phase_s\n";
}
static void row(std::ostream& file,double time,const Data_vector& point) {
  file<<time;
  for(int i=0;i<21;++i)file<<','<<point.x[i];
  for(double v:point.v)file<<','<<v;
  for(double v:point.q)file<<','<<v;
  file<<','<<sum_volume(point)<<','<<point.time[1]<<','<<point.time[6]<<'\n';
}
int main(int argc,char** argv) {
  try {
    if(argc!=6 && argc!=7) throw std::runtime_error("usage: continuous_reference tolerance cycles trace.csv report.json sample_dt_seconds [original.csv]");
    Integrator integrator;
    char *end;
    integrator.relative_tolerance=integrator.absolute_tolerance=std::strtod(argv[1],&end);
    if(*end || !std::isfinite(integrator.relative_tolerance) || integrator.relative_tolerance<=0) throw std::runtime_error("invalid tolerance");
    const auto cycles=std::strtoul(argv[2],&end,10);
    if(*end || cycles<1 || cycles>500)throw std::runtime_error("invalid cycles");
    const double sample_dt=std::strtod(argv[5],&end);
    if(*end || !std::isfinite(sample_dt) || sample_dt<=0. || sample_dt>.1)throw std::runtime_error("invalid output sampling interval");
    const double duration=cycles*integrator.model.period;
    std::ofstream trace(argv[3]);if(!trace)throw std::runtime_error("cannot write trace");
    trace<<std::setprecision(17);header(trace);
    const auto initial=integrator.model.observables(0.,integrator.state);
    const double initial_volume=sum_volume(initial);
    double maximum_volume_error=0.,maximum_source_volume_error=0.,maximum_source_pressure_error=0.,maximum_source_flow_error=0.;
    std::size_t samples=0;
    auto sample=[&](double t) {
      integrator.advance(t);
      const auto point=integrator.model.observables(t,integrator.state);
      row(trace,t,point);++samples;
      maximum_volume_error=std::max(maximum_volume_error,std::abs(sum_volume(point)-initial_volume));
      return point;
    };
    if(argc==7) {
      std::ifstream original(argv[6]);if(!original)throw std::runtime_error("missing original trace");
      std::string line;std::getline(original,line);
      while(std::getline(original,line)) {
        std::vector<double> values;std::stringstream stream(line);std::string value;
        while(std::getline(stream,value,','))values.push_back(std::stod(value));
        if(values.size()<69)throw std::runtime_error("original trace shape");
        const double time=values[1];if(time>duration)break;
        const auto point=sample(time);
        for(int i=0;i<21;++i) {
          maximum_source_pressure_error=std::max(maximum_source_pressure_error,std::abs(point.x[i]-values[3+i]));
          maximum_source_volume_error=std::max(maximum_source_volume_error,std::abs(point.v[i]-values[24+i]));
        }
        for(int i=0;i<24;++i)maximum_source_flow_error=std::max(maximum_source_flow_error,std::abs(point.q[i]-values[45+i]));
      }
    } else {
      const auto sample_count=static_cast<std::size_t>(std::ceil(duration/sample_dt));
      for(std::size_t i=0;i<=sample_count;++i)sample(std::min(duration,i*sample_dt));
    }
    const auto final=integrator.model.observables(integrator.time,integrator.state);
    std::ofstream report(argv[4]);report<<std::setprecision(17);
    report<<"{\n\"schema\":\"NumiHuman.CVSim21-continuous-reference.v1\",\n\"status\":\"pass\",\n"
      <<"\"source_mode\":\"unmodified_C_equations_prescribed_rational_HR_phase\",\n"
      <<"\"source_clock_interpretation\":\"phase=t-floor(t/(60/70))*(60/70); no sanode or queue\",\n"
      <<"\"source_clock_equivalence_to_original_step_sim\":false,\n"
      <<"\"source_units\":{\"pressure\":\"mmHg\",\"volume\":\"mL\",\"flow\":\"mL/s\"},\n"
      <<"\"downstream_native_pressure_conversion_Pa_per_mmHg\":133.3224,\n"
      <<"\"pressure_conversion_is_source_constant\":false,\n"
      <<"\"upstream_equations_unchanged\":true,\n\"upstream_initial_conditions_unchanged\":true,\n"
      <<"\"ABReflexOn\":false,\n\"CPReflexOn\":false,\n\"tiltTestOn\":false,\n"
      <<"\"biological_qualification\":\"unqualified\",\n"
      <<"\"relative_tolerance\":"<<integrator.relative_tolerance<<",\n\"absolute_tolerance_mmHg\":"<<integrator.absolute_tolerance<<",\n"
      <<"\"cycles_requested\":"<<cycles<<",\n\"duration_s\":"<<integrator.time<<",\n\"samples\":"<<samples<<",\n"
      <<"\"sample_dt_s\":"<<sample_dt<<",\n"
      <<"\"accepted_steps\":"<<integrator.accepted_steps<<",\n\"rejected_steps\":"<<integrator.rejected_steps<<",\n"
      <<"\"rhs_evaluations\":"<<integrator.evaluations<<",\n\"minimum_step_s\":"<<integrator.smallest_accepted_step<<",\n"
      <<"\"initial_volume_sum_mL\":"<<initial_volume<<",\n\"final_volume_sum_mL\":"<<sum_volume(final)<<",\n"
      <<"\"maximum_volume_change_mL\":"<<maximum_volume_error<<",\n"
      <<"\"minimum_evaluated_leg_transmural_mmHg\":"<<integrator.model.minimum_evaluated_leg_transmural<<",\n"
      <<"\"compared_at_original_accepted_times\":"<<(argc==7?"true":"false")<<",\n"
      <<"\"max_original_pressure_difference_mmHg\":"<<maximum_source_pressure_error<<",\n"
      <<"\"max_original_volume_difference_mL\":"<<maximum_source_volume_error<<",\n"
      <<"\"max_original_flow_difference_mL_per_s\":"<<maximum_source_flow_error<<"\n}\n";
    std::cout<<"continuous_reference=pass samples="<<samples<<" steps="<<integrator.accepted_steps<<" rejected="<<integrator.rejected_steps<<" max_volume_change_mL="<<maximum_volume_error<<'\n';
  } catch(const std::exception& error) { std::cerr<<"continuous_reference=fail error="<<error.what()<<'\n';return 1; }
}

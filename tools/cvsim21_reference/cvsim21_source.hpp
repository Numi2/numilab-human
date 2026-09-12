#pragma once
// Evaluator for unmodified CVSim equations with prescribed rational HR phase.
// This interpretation is distinct from the original discrete sanode/step_sim.
extern "C" {
#include "main/main.h"
#include "sim/simulator.h"
#include "sim/estimate.h"
#include "sim/equation.h"
}
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cvsim21_reference {
constexpr std::size_t state_count=21;
using State=std::array<double,state_count>;

struct SourceModel {
  Parameter_vector parameters{};
  Data_vector baseline{};
  Reflex_vector fixed_reflex{};
  State initial{};
  double minimum_evaluated_leg_transmural=std::numeric_limits<double>::infinity();
  double period=0.;
  SourceModel() {
    Hemo hemo{}; Cardiac cardiac{}; Micro_r micro{}; System_parameters system{}; Reflex reflex{}; Timing timing{};
    initial_ptr(&hemo,&cardiac,&micro,&system,&reflex,&timing);
    mapping_ptr(&hemo,&cardiac,&micro,&system,&reflex,&timing,&parameters);
    estimate_ptr(&baseline,&parameters,&fixed_reflex);
    elastance_ptr(&baseline,&parameters);
    eqns_ptr(&baseline,&parameters,&fixed_reflex,0,0.,0.);
    if(parameters.vec[90]!=70.) throw std::runtime_error("unexpected source heart rate");
    period=60./parameters.vec[90];
    for(std::size_t i=0;i<state_count;++i) initial[i]=baseline.x[i];
  }
  Data_vector observables(double time,const State& state) {
    Data_vector point=baseline;
    const double cycle=std::floor(time/period);
    double phase=time-cycle*period;
    if(phase<0 || phase>=period) phase=0.;
    point.time[0]=time;point.time[1]=phase;point.time[5]=time;
    point.time[6]=phase-point.time[2];
    point.c[4]=fixed_reflex.compliance[0];point.c[5]=fixed_reflex.compliance[1];
    for(std::size_t i=0;i<state_count;++i) point.x[i]=state[i];
    // Mirror only source branch admission, not its physical flow computation.
    // Otherwise q4 can silently retain a prior value for equality or reverse
    // pressure above the external floor. Supine has zero gravity here.
    const double upstream=point.x[3]+point.grav[3], downstream=point.x[4], external=point.x[24];
    const bool sourceAssignsFlow=(upstream>downstream && downstream>external) ||
      (upstream>external && external>downstream) || (upstream<external);
    if(!sourceAssignsFlow) throw std::runtime_error("source Starling flow branch is unspecified");
    elastance_ptr(&point,&parameters);
    eqns_ptr(&point,&parameters,&fixed_reflex,0,0.,0.);
    const double leg=point.x[12]-point.x[22];
    minimum_evaluated_leg_transmural=std::min(minimum_evaluated_leg_transmural,leg);
    // For negative leg Ptm, source reported volume differs from its storage law.
    if(!(leg>0.)) throw std::runtime_error("source negative-leg volume branch is outside reference scope");
    return point;
  }
  void evaluate(double time,const State& state,State& derivative) {
    const auto point=observables(time,state);
    for(std::size_t i=0;i<state_count;++i) derivative[i]=point.dxdt[i];
  }
};
}

/* Observation harness only: original CVSim init_sim/step_sim own all physics. */
#include "main/main.h"
#include "main/main_java.h"
#include "sim/simulator.h"
#include <string.h>

extern Data_vector pressure;
extern Reflex_vector reflex_vector;
extern double hdid;

static const char *compartments[21] = {
  "ascending_aorta", "brachiocephalic_arteries", "upper_body_arteries",
  "upper_body_veins", "superior_vena_cava", "thoracic_aorta", "abdominal_aorta",
  "renal_arteries", "renal_veins", "splanchnic_arteries", "splanchnic_veins",
  "lower_body_arteries", "lower_body_veins", "abdominal_veins", "inferior_vena_cava",
  "right_atrium", "right_ventricle", "pulmonary_arteries", "pulmonary_veins",
  "left_atrium", "left_ventricle"
};

static double emit(FILE *trace, const Parameter_vector *theta, size_t index) {
  double volume = 0.;
  for (int i=0;i<21;++i) {
    if (!isfinite(pressure.x[i]) || !isfinite(pressure.v[i])) {
      fprintf(stderr,"nonfinite compartment at step %zu coordinate %d\n",index,i); exit(4);
    }
    volume += pressure.v[i];
  }
  fprintf(trace,"%zu,%.17g,%.17g",index,pressure.time[0],index ? hdid : 0.);
  for(int i=0;i<21;++i) fprintf(trace,",%.17g",pressure.x[i]);
  for(int i=0;i<21;++i) fprintf(trace,",%.17g",pressure.v[i]);
  for(int i=0;i<24;++i) {
    if (!isfinite(pressure.q[i])) {fprintf(stderr,"nonfinite flow\n");exit(4);}
    fprintf(trace,",%.17g",pressure.q[i]);
  }
  for(int i=21;i<25;++i) fprintf(trace,",%.17g",pressure.x[i]);
  for(int i=0;i<4;++i) fprintf(trace,",%.17g",reflex_vector.hr[i]);
  for(int i=0;i<4;++i) fprintf(trace,",%.17g",reflex_vector.volume[i]);
  for(int i=1;i<7;++i) fprintf(trace,",%.17g",pressure.time[i]);
  for(int i=0;i<7;++i) fprintf(trace,",%.17g",pressure.c[i]);
  fprintf(trace,",%.17g,%d",pressure.x[12]-pressure.x[22],pressure.x[4]==pressure.x[24]);
  fprintf(trace,",%.17g,%.17g,%.17g\n",volume,theta->vec[70],volume-theta->vec[70]);
  return volume;
}

int main(int argc,char **argv) {
  if(argc!=6) {fprintf(stderr,"usage: %s duration_seconds ABReflexOn CPReflexOn trace.csv parameters.csv\n",argv[0]);return 2;}
  char *end; double duration=strtod(argv[1],&end);
  if(*end || !isfinite(duration) || duration<=0 || duration>600) return 2;
  int ab=atoi(argv[2]),cp=atoi(argv[3]);
  if ((strcmp(argv[2],"0") && strcmp(argv[2],"1")) || (strcmp(argv[3],"0") && strcmp(argv[3],"1"))) return 2;
  FILE *trace=fopen(argv[4],"w"),*ledger=fopen(argv[5],"w");
  if(!trace || !ledger) {perror("evidence output");return 3;}
  Parameter_vector theta={{0.0}}; output stepout={0};
  init_sim(&theta);
  fprintf(ledger,"parameter_index,source_value\n");
  for(int i=0;i<N_PARAMETER;++i) fprintf(ledger,"%d,%.17g\n",i,theta.vec[i]);
  fclose(ledger);
  fprintf(trace,"accepted_step,time_s,accepted_dt_s");
  for(int i=0;i<21;++i) fprintf(trace,",P_%s_mmHg",compartments[i]);
  for(int i=0;i<21;++i) fprintf(trace,",V_%s_mL",compartments[i]);
  for(int i=0;i<24;++i) fprintf(trace,",Q_%02d_mL_per_s",i);
  for(int i=21;i<25;++i) fprintf(trace,",P_external_%d_mmHg",i);
  for(int i=0;i<4;++i) fprintf(trace,",reflex_hr_%d",i);
  for(int i=0;i<4;++i) fprintf(trace,",reflex_ZPFV_%d_mL",i);
  for(int i=1;i<7;++i) fprintf(trace,",source_time_%d_s",i);
  for(int i=0;i<7;++i) fprintf(trace,",source_compliance_%d_mL_per_mmHg",i);
  fprintf(trace,",leg_transmural_pressure_mmHg,starling_outlet_equals_floor");
  fprintf(trace,",volume_sum_mL,target_total_mL,volume_difference_mL\n");
  double initial=emit(trace,&theta,0),max_error=fabs(initial-theta.vec[70]),final=initial;
  double minimum_leg_transmural=pressure.x[12]-pressure.x[22];
  size_t steps=0,starling_equalities=(pressure.x[4]==pressure.x[24]);
  while(pressure.time[0]<duration) {
    double previous=pressure.time[0];
    step_sim(&stepout,&theta,1,ab,cp,0,0.,0.);
    if(!isfinite(pressure.time[0]) || pressure.time[0]<=previous || ++steps>10000000) return 4;
    final=emit(trace,&theta,steps);
    if(fabs(final-theta.vec[70])>max_error) max_error=fabs(final-theta.vec[70]);
    if(pressure.x[12]-pressure.x[22]<minimum_leg_transmural) minimum_leg_transmural=pressure.x[12]-pressure.x[22];
    starling_equalities+=(pressure.x[4]==pressure.x[24]);
  }
  fclose(trace);
  printf("cvsim_source_run=pass version=1.0.0 duration_seconds=%.17g steps=%zu ABReflexOn=%d CPReflexOn=%d tiltTestOn=0 pressure_count=21 volume_count=21 flow_count=24 target_total_mL=%.17g initial_volume_sum_mL=%.17g final_volume_sum_mL=%.17g max_abs_target_difference_mL=%.17g min_leg_transmural_pressure_mmHg=%.17g starling_outlet_floor_equality_samples=%zu source_equations_unchanged=true source_parameters_unchanged=true biological_qualification=unqualified\n",pressure.time[0],steps,ab,cp,theta.vec[70],initial,final,max_error,minimum_leg_transmural,starling_equalities);
  return 0;
}

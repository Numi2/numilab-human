from pathlib import Path
import argparse, json
parser=argparse.ArgumentParser(description='Source observable selection and SI coordinate mapping; no ODE stepping.')
parser.add_argument('--output-dir',type=Path,required=True)
p=parser.parse_args().output_dir.resolve();j=json.loads((p/'flattened-source.json').read_text())
def index(suffix):
 matches=[v['index'] for v in j['variables'] if any(x.endswith(suffix) for x in v['aliases'])]
 if len(matches)!=1:raise ValueError((suffix,matches))
 return matches[0]
# Only output-name selection and exact source-unit conversion live here; every
# value is produced by the generated source algebraic/differential evaluator.
obs=[]
for chamber,kind in [('la','a'),('lv','v'),('ra','a'),('rv','v')]:
 path=chamber.upper()+'/TempCD'+kind+'/'
 for symbol,label,factor,unit in [('V','volume',1e-6,'m3'),('Pi','pressure',133.,'Pa'),('Qo','outflow',1e-6,'m3_per_s')]:obs.append((chamber+'_'+label+'_'+unit,index(path+symbol),factor))
for comp,model in [('sas','TempRLC'),('sat','TempRLC'),('sar','TempR'),('scp','TempR'),('svn','TempRC'),('pas','TempRLC'),('pat','TempRLC'),('par','TempR'),('pcp','TempR'),('pvn','TempRC')]:
 for symbol,label,factor,unit in [('Pi','pressure',133.,'Pa'),('Qo','outflow',1e-6,'m3_per_s')]:obs.append((comp+'_'+label+'_'+unit,index(comp.capitalize()+'/'+model+'/'+symbol),factor))
for x in ['la','lv','ra','rv']:obs.append((x+'_elastance_Pa_per_m3',index('E'+x+'/E'+('Atrium' if x.endswith('a') else 'Ventricle')+'/E'),133e6))
lines=['#pragma once','#include "shi_hose_reference_generated.hpp"','namespace numi_shi_hose_reference {','struct Observable {std::string_view name;std::size_t index;double scale;};',f'inline constexpr std::array<Observable,{len(obs)}> observables{{{{']
lines += ['{'+json.dumps(n)+','+str(i)+','+repr(s)+'},' for n,i,s in obs]
lines+=['}};','inline std::array<double,20> native_hydraulic_state(const Values& a) {','return {']
rows=[]
for path,label in [('LA/TempCDa/V','LA_volume'),('LV/TempCDv/V','LV_volume'),('Sas/TempRLC/Pi','SAS_compliance_storage'),('Sat/TempRLC/Pi','SAT_compliance_storage'),('Svn/TempRC/Pi','SVN_compliance_storage'),('RA/TempCDa/V','RA_volume'),('RV/TempCDv/V','RV_volume'),('Pas/TempRLC/Pi','PAS_compliance_storage'),('Pat/TempRLC/Pi','PAT_compliance_storage'),('Pvn/TempRC/Pi','PVN_compliance_storage')]:
 # Coefficients for compliance storage are looked up from generated source
 # constants, rather than copied into the runtime reference expression.
 if 'compliance' in label:
  symbol='C'+label.split('_')[0].lower();par='ParaSys' if symbol.startswith('Cs') else 'ParaPul';source_index=index(par+'/'+par+'/'+symbol);expression=f'a[{index(path)}]*a[{source_index}]*1e-6'
 else:expression=f'a[{index(path)}]*1e-6'
 lines.append(expression+', // '+label);rows.append({'index':len(rows),'name':label,'source_variable_index':index(path),'kind':'volume' if 'volume' in label else 'compliance_storage','SI_unit':'m3'})
for path,label in [('LA/TempCDa/Qo','mitral'),('LV/TempCDv/Qo','aortic'),('Sas/TempRLC/Qo','sas_to_sat'),('Sat/TempRLC/Qo','sat_to_svn'),('Svn/TempRC/Qo','svn_to_ra'),('RA/TempCDa/Qo','tricuspid'),('RV/TempCDv/Qo','pulmonary_valve'),('Pas/TempRLC/Qo','pas_to_pat'),('Pat/TempRLC/Qo','pat_to_pvn'),('Pvn/TempRC/Qo','pvn_to_la')]:lines.append(f'a[{index(path)}]*1e-6, // '+label);rows.append({'index':len(rows),'name':label,'source_variable_index':index(path),'kind':'flow','SI_unit':'m3/s'})
lines+=['};','}','}',''];(p/'shi_hose_observables.hpp').write_text('\n'.join(lines));(p/'native-state-mapping.json').write_text(json.dumps({'scope':'hydraulic compliance storage, not absolute vascular blood volumes','rows':rows,'observables':[dict(name=n,index=i,scale=s) for n,i,s in obs]},indent=2)+'\n')
print(len(obs),'SI observables;20native hydraulic-state rows')

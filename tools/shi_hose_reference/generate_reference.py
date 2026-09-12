"""Offline CellML1.1 import/encapsulation/connection compiler; no ODE stepping."""
from pathlib import Path
import xml.etree.ElementTree as ET
import argparse, hashlib, json, re
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args();SRC=args.source_dir.resolve();OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
LOCK=json.loads((HERE/'source-lock.json').read_text())
NS={'c':'http://www.cellml.org/cellml/1.1#','m':'http://www.w3.org/1998/Math/MathML'}
XL='{http://www.w3.org/1999/xlink}href'
REV='a679cdc2e97429fb5280af8132c119758626c1f2'
def tag(e):return e.tag.rsplit('}',1)[-1]
class UF:
 def __init__(self):self.p={}
 def add(self,x):self.p.setdefault(x,x)
 def find(self,x):
  if x not in self.p:raise ValueError('Unknown variable '+x)
  if self.p[x]!=x:self.p[x]=self.find(self.p[x])
  return self.p[x]
 def union(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.p[max(a,b)]=min(a,b)
uf=UF();variables={};equations=[];files={};instances=[]
def document(name):
 p=(SRC/name).resolve()
 if p.parent!=SRC or name not in LOCK['cellml_files']:raise ValueError('Unpinned source dependency '+name)
 data=p.read_bytes();digest=hashlib.sha256(data).hexdigest()
 if digest!=LOCK['cellml_files'][name]:raise ValueError('Pinned source digest mismatch '+name)
 files[name]=digest;return ET.fromstring(data)
def instantiate(name,prefix='',selection=None):
 root=document(name)
 local={c.attrib['name']:c for c in root.findall('c:component',NS)}
 imports={}
 for imp in root.findall('c:import',NS):
  for comp in imp.findall('c:component',NS):imports[comp.attrib['name']]=(imp.attrib[XL],comp.attrib['component_ref'])
 children={}
 def walk(e):
  n=e.attrib['component'];children.setdefault(n,set())
  for child in e.findall('c:component_ref',NS):children[n].add(child.attrib['component']);walk(child)
 for group in root.findall('c:group',NS):
  if any(r.attrib.get('relationship')=='encapsulation' for r in group.findall('c:relationship_ref',NS)):
   for e in group.findall('c:component_ref',NS):walk(e)
 selected=set(local)|set(imports) if selection is None else set(selection)
 while True:
  expanded=selected|set().union(*(children.get(n,set()) for n in selected))
  if expanded==selected:break
  selected=expanded
 if not selected <= set(local)|set(imports):raise ValueError('Unresolved component selection')
 result={}
 for n in sorted(selected):
  if n in imports:
   file,target=imports[n];nested=instantiate(file,prefix+n+'/',{target});result[n]=nested[target]
  else:
   component=local[n];qualified=prefix+n;result[n]={}
   for v in component.findall('c:variable',NS):
    key=qualified+'/'+v.attrib['name'];uf.add(key);variables[key]={'file':name,'component':qualified,**v.attrib};result[n][v.attrib['name']]=key
   for math in component.findall('m:math',NS):
    for equation in math:
     if tag(equation)!='apply' or tag(equation[0])!='eq':raise ValueError('Non-equation MathML')
     equations.append((qualified,equation))
   instances.append({'file':name,'source_component':n,'instance':qualified})
 for connection in root.findall('c:connection',NS):
  pair=connection.find('c:map_components',NS).attrib;a,b=pair['component_1'],pair['component_2']
  if a not in selected or b not in selected:continue
  for mp in connection.findall('c:map_variables',NS):
   va,vb=result[a][mp.attrib['variable_1']],result[b][mp.attrib['variable_2']]
   # All connected variables in this model use identical source units. Fail
   # rather than silently assume a conversion when the source changes.
   if variables[va]['units']!=variables[vb]['units']:raise ValueError('Connected unit conversion not supported')
   uf.union(va,vb)
 return result
if LOCK['revision']!=REV:raise ValueError('Source revision mismatch')
instantiate('ModelMain.cellml')
# Import every unit-definition dependency into the evidence inventory.
for f in list(files):
 for imp in document(f).findall('c:import',NS):
  if imp.findall('c:units',NS):document(imp.attrib[XL])
if set(files)!=set(LOCK['cellml_files']):raise ValueError('Source dependency inventory mismatch')
roots=sorted({uf.find(v) for v in variables});alias={r:sorted(k for k in variables if uf.find(k)==r) for r in roots}
def resolve(comp,name):return uf.find(comp+'/'+name.strip())
state_rhs={};alg={};bound=set()
for comp,e in equations:
 lhs,rhs=e[1],e[2]
 if tag(lhs)=='ci': key=resolve(comp,lhs.text);dest=alg
 elif tag(lhs)=='apply' and tag(lhs[0])=='diff':
  key=resolve(comp,lhs[-1].text);bound.add(resolve(comp,lhs[1][0].text));dest=state_rhs
 else:raise ValueError('Unsupported equation lhs')
 if key in dest:raise ValueError('Multiply defined equation '+key)
 dest[key]=(comp,rhs)
if len(bound)!=1:raise ValueError('Expected one shared source time')
time=bound.pop();states=sorted(state_rhs)
if len(states)!=14:raise ValueError('Pinned source state count changed')
initial={}
for v,a in variables.items():
 if 'initial_value' not in a:continue
 value=a['initial_value'];key=uf.find(v)
 if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?',value):definition=('literal',value)
 else:definition=('alias',resolve(a['component'],value))
 if key in initial and initial[key]!=definition:raise ValueError('Conflicting initial values '+key)
 initial[key]=definition
constants={k:v for k,v in initial.items() if k not in states}
def literal(k,seen=()):
 if k in seen or k not in initial:raise ValueError('Unresolved initial reference '+k)
 typ,value=initial[k]
 return value if typ=='literal' else literal(value,seen+(k,))
for k in constants:literal(k)
var_index={k:i for i,k in enumerate(roots)};state_index={k:i for i,k in enumerate(states)}
def deps(comp,e):return {resolve(comp,ci.text) for ci in e.iter() if tag(ci)=='ci'}
ready=set(constants)|set(states)|{time};ordered=[];remaining=dict(alg)
while remaining:
 nexts=sorted(k for k,(c,e) in remaining.items() if deps(c,e)<=ready)
 if not nexts:raise ValueError('Algebraic cycle or missing definition '+repr({k:deps(*v)-ready for k,v in remaining.items()}))
 for k in nexts:ordered.append(k);ready.add(k);del remaining[k]
for k,(c,e) in state_rhs.items():
 if not deps(c,e)<=ready:raise ValueError('ODE depends on unassigned variables')
def expr(c,e):
 t=tag(e)
 if t=='ci':return 'a['+str(var_index[resolve(c,e.text)])+']'
 if t=='cn':
  s=e.text.strip();return s if any(ch in s for ch in '.eE') else s+'.0'
 if t=='piecewise':
  parts=list(e);other='std::numeric_limits<double>::quiet_NaN()'
  for p in reversed(parts):
   if tag(p)=='otherwise':other=expr(c,p[0])
   elif tag(p)=='piece':other=f'({expr(c,p[1])} ? {expr(c,p[0])} : {other})'
   else:raise ValueError('Unknown piecewise child')
  return other
 if t!='apply':raise ValueError('Unhandled MathML '+t)
 op=tag(e[0]);v=[expr(c,x) for x in e[1:]]
 infix={'plus':'+','minus':'-','times':'*','divide':'/','geq':'>=','leq':'<=','gt':'>','lt':'<','eq':'==','and':'&&','or':'||'}
 if op in infix:
  if len(v)==1 and op=='minus':return '(-'+v[0]+')'
  return '('+(' '+infix[op]+' ').join(v)+')'
 if op=='power':return 'std::pow('+','.join(v)+')'
 if op in ['cos','sin','floor','abs','exp','ln','root']:
  fname={'ln':'log','root':'sqrt','abs':'fabs'}.get(op,op);return 'std::'+fname+'('+','.join(v)+')'
 raise ValueError('Unhandled MathML operator '+op)
lines=['// Generated solely from the pinned CellML1.1 source imports/MathML.', '// Source: Zero dimensional (lumped parameter) modelling of native human cardiovascular dynamics.', '// Original CellML by Yubing Shi, Rod Hose and PMR contributors; CC BY 3.0.', '// https://models.physiomeproject.org/exposure/c49d416ae3a5132882e6ea7479ba50f5', '// License: https://creativecommons.org/licenses/by/3.0/', '// Numi translation changes representation only; not biological calibration.', '#pragma once','#include <array>','#include <cmath>','#include <limits>','#include <string_view>','namespace numi_shi_hose_reference {',f'inline constexpr std::string_view revision="{REV}";',f'inline constexpr std::size_t state_count={len(states)};',f'inline constexpr std::size_t variable_count={len(roots)};','using State=std::array<double,state_count>;','using Values=std::array<double,variable_count>;']
lines+=['inline constexpr std::array<std::string_view,state_count> state_names{{'+','.join(json.dumps(k) for k in states)+'}};']
lines+=['inline constexpr std::array<std::string_view,variable_count> variable_names{{'+','.join(json.dumps(k) for k in roots)+'}};']
lines+=['inline State initial_state() {return State{'+','.join(str(float(literal(k))) for k in states)+'};}']
lines+=['inline void evaluate(double t, const State& y, State& dy, Values* outputs=nullptr) {','Values a{};',f'a[{var_index[time]}]=t;']
for k in sorted(constants):lines+=[f'a[{var_index[k]}]={float(literal(k))!r}; // {k}']
for k,i in state_index.items():lines+=[f'a[{var_index[k]}]=y[{i}]; // {k}']
for k in ordered:lines+=[f'a[{var_index[k]}]={expr(*alg[k])}; // {k}']
for k,i in state_index.items():lines+=[f'dy[{i}]={expr(*state_rhs[k])}; // d/dt {k}']
lines+=['if(outputs)*outputs=a;','}','} // namespace numi_shi_hose_reference','']
(OUT/'shi_hose_reference_generated.hpp').write_text('\n'.join(lines))
manifest={'revision':REV,'license':'CC-BY-3.0','source_files':files,'instances':instances,'state_count':len(states),'states':[{'index':i,'canonical_name':k,'aliases':alias[k],'source_initial_value':literal(k),'unit':variables[k]['units']} for i,k in enumerate(states)],'variables':[{'index':i,'canonical_name':k,'aliases':alias[k],'unit':variables[k]['units'],'role':'state' if k in states else 'time' if k==time else 'constant' if k in constants else 'algebraic' if k in alg else 'unused'} for i,k in enumerate(roots)],'algebraic_equations':len(ordered),'generated_header_sha256':hashlib.sha256((OUT/'shi_hose_reference_generated.hpp').read_bytes()).hexdigest()}
(OUT/'flattened-source.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'states':len(states),'variables':len(roots),'algebraics':len(ordered),'instances':len(instances),'source_files':len(files),'header_sha256':manifest['generated_header_sha256']}))
for s in manifest['states']:print(s['index'],s['canonical_name'],s['source_initial_value'],s['unit'])

#!/usr/bin/env python3
"""Compare Matter's connector equations with compiled public FEBio source.

Archived poses select configurations, not force truth: the archived pose text
is rounded to six significant digits. Both operators receive identical,
normalized inputs. No equilibrium, integration or native knee admission follows.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from numilab_human.open_knee_febio_log import parse_febio_log

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def vec(value):return [float(x) for x in value.split(',')]
def normalize(v):
    n=math.sqrt(sum(x*x for x in v))
    if not math.isfinite(n) or n==0:raise ValueError('invalid source direction/quaternion')
    return [x/n for x in v]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deck',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--matter',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if sha(a.deck)!='00b6efb53ad7e7330296cbb9569d358d48ed60819e22732e6149db6fb98a158a':
        raise ValueError('this comparison is qualified only for the frozen oks003 deck')
    if sha(a.log)!='d47631c09ce7fc93154c7d6c84caa299aab03709abe9ab7d2a41a60a1b78e426':
        raise ValueError('archived pose observations differ from the frozen oks003 log')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    root=ET.parse(a.deck).getroot();observations=parse_febio_log(a.log.read_text())
    if observations['status']!='normal_termination_with_complete_observations':
        raise ValueError('requires complete archived source observations')
    bodies={int(x.get('id')):vec(x.findtext('center_of_mass')) for x in root.findall('Material/material')
            if x.get('type')=='rigid body'}
    joints=[x for x in root.findall('.//constraint') if x.get('type')=='rigid cylindrical joint']
    if len(joints)!=6:raise ValueError('this qualification requires the six source cylindrical joints')
    # No silent substitution for augmented history, dynamic interpolation, or
    # a changed load program. The current source uses a pure penalty joint.
    controls=root.findall('Step/Control/analysis')
    if len(controls)!=1 or controls[0].get('type')!='static':raise ValueError('only alpha=1 static source supported')
    curves={x.get('id'):x for x in root.findall('LoadData/loadcurve')}
    def scalar(joint,name,time):
        element=joint.find(name)
        if element is None:raise ValueError('missing joint coordinate value')
        value=float(element.text)
        if 'lc' in element.attrib:
            curve=curves[element.get('lc')]
            if curve.get('type')!='linear':raise ValueError('unsupported joint load curve')
            points=[vec2.text.split(',') for vec2 in curve.findall('point')]
            points=[tuple(map(float,x)) for x in points]
            if len(points)<2 or any(x[0]>=y[0] for x,y in zip(points,points[1:])):raise ValueError('invalid curve')
            if not points[0][0]<=time<=points[-1][0]:raise ValueError('load curve extrapolation unsupported')
            for (t0,y0),(t1,y1) in zip(points,points[1:]):
                if t0<=time<=t1:return value*(y0+(y1-y0)*(time-t0)/(t1-t0))
        return value
    groups={}
    for r in observations['records']:groups.setdefault(r['step'],{})[r['field']]=r
    rows=[];keys=[];graph=[]
    for index,j in enumerate(joints):
        if int(j.findtext('maxaug'))!=0 or int(j.findtext('minaug'))!=0:
            raise ValueError('archived joint multipliers not available for augmented execution')
        graph.append({'name':j.get('name'),'body_a':int(j.findtext('body_a')),
                      'body_b':int(j.findtext('body_b')),'authored_xml':ET.tostring(j,encoding='unicode')})
    for step,g in groups.items():
        positions=g['center_of_mass']['values'];rotations=g['rotation_quaternion']['values']
        time=g['center_of_mass']['continuation_time']
        for index,j in enumerate(joints):
            ba,bb=int(j.findtext('body_a')),int(j.findtext('body_b'))
            row=bodies[ba]+bodies[bb]+positions[ba]+positions[bb]+vec(j.findtext('joint_origin'))+normalize(vec(j.findtext('joint_axis')))
            row+=normalize(rotations[ba])+normalize(rotations[bb])+[0]*6
            row += [float(j.findtext('force_penalty')),float(j.findtext('moment_penalty')),
                    scalar(j,'translation',time),scalar(j,'rotation',time),
                    float(j.findtext('force','0')),float(j.findtext('moment','0')),
                    int(j.findtext('prescribed_translation')),int(j.findtext('prescribed_rotation'))]
            rows.append(row);keys.append({'step':step,'time':time,'joint':j.get('name')})
    inputs=out/'inputs.txt';inputs.write_text(str(len(rows))+'\n'+'\n'.join(' '.join(format(x,'.17g') for x in row) for row in rows)+'\n')
    (out/'source-rigid-graph.json').write_text(json.dumps({'bodies':bodies,'cylindrical_joints':graph},indent=2)+'\n')
    (out/'row-keys.json').write_text(json.dumps(keys,indent=2)+'\n')
    receipt={'status':'running','scope':'connector equation comparison at archived poses; no knee equilibrium qualification',
             'source_equivalence':False,'rigid_graph_execution':False,'rows':len(rows),
             'identities':{name:{'file':str(path.resolve()),'sha256':sha(path)} for name,path in
                [('source_deck',a.deck),('archived_log',a.log),('matter_binary',a.matter),('febio_oracle',a.oracle),('inputs',inputs)]},
             'archived_version':observations['version'],'oracle_version':'public FEBio 2.9.0, not original 2.9.1 executable',
             'input_normalization':'unit quaternions and axes; no geometry shifts',
             'missing_axial_force_moment':'zero defaults confirmed in pinned public 2.9.0 constructor'}
    try:
        for name,command in [('febio',[str(a.oracle.resolve()),str(inputs),str(out/'febio.txt')]),
                             ('matter',[str(a.matter.resolve()),'--source-rows',str(inputs),str(out/'matter.txt')])]:
            with (out/(name+'.stdout')).open('x') as stream:
                subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        native=np.loadtxt(out/'matter.txt')[:,[0,1,2,18,19,20]];reference=np.loadtxt(out/'febio.txt')
        if native.shape!=(len(rows),6) or reference.shape!=native.shape:raise ValueError('incomplete comparison')
        difference=np.abs(native-reference)
        # Fixed absolute gates include the legacy acos loss near zero rotation.
        # Preserve individual force/moment maxima as well as pass/fail.
        receipt['maximum_absolute_force_error_N']=float(difference[:,:3].max())
        receipt['maximum_absolute_moment_error_N_mm']=float(difference[:,3:].max())
        receipt['gates']={'force_N':1e-8,'moment_N_mm':1e-3}
        passed=np.isfinite(difference).all() and difference[:,:3].max()<1e-8 and difference[:,3:].max()<1e-3
        receipt['status']='passed_equations_only' if passed else 'failed_equations'
        if not passed:raise ValueError('source joint equations disagree')
    except BaseException as error:
        receipt.update(status='failed_comparison',error=str(error));raise
    finally:
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()

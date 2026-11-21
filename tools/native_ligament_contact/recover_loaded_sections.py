"""Recover loaded layer positions from the native condensed contact coordinate.

The rigid point is the outer surface position before layer compression. Its
negative normal coordinate measures series compression. In a resolved volume
the physical exterior belongs on the bed and the deep material end stays with
the body. This is a reference for the fine contact solve, with no force input.
"""
import argparse
import csv
import math
from pathlib import Path

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def recover(skin, fat, skin_compliance, fat_compliance, marker_z, load):
    if not all(math.isfinite(v) for v in (skin,fat,skin_compliance,fat_compliance,marker_z,load)):
        raise ValueError('Nonfinite bed section')
    if min(skin,fat,skin_compliance,fat_compliance)<=0 or load<0:
        raise ValueError('Invalid material thickness, compliance or load')
    delta=max(-marker_z,0.0)
    elastic=delta/(skin_compliance+fat_compliance)
    hs=skin-elastic*skin_compliance
    hf=fat-elastic*fat_compliance
    if min(hs,hf)<=0:
        raise ValueError('Nonpositive reconstructed layer thickness')
    outer=marker_z+delta
    interface=outer+hs
    deep=interface+hf
    if not math.isclose(deep,marker_z+skin+fat,abs_tol=5e-15,rel_tol=0):
        raise ValueError('Material end moved during layer reconstruction')
    return dict(outer_z_m=outer,skin_fat_interface_z_m=interface,deep_fat_z_m=deep,
                loaded_skin_thickness_m=hs,loaded_fat_thickness_m=hf,elastic_reaction_N=elastic,
                impulse_force_minus_elastic_N=load-elastic,elastic_energy_J=0.5*elastic*delta)

def run(run_dir,output,dt):
    loaded=read(run_dir/'resting-com-support-impulses.csv')
    poses=read(run_dir/'resting-support-point-motion.csv')
    sections=read(run_dir/'resting-local-bed-sections.csv')
    end=max(int(x['accepted_step']) for x in loaded)
    key=lambda r:(int(r['contact_index']),int(r['source_geometry_index']))
    points={key(r):r for r in poses if int(r['accepted_step'])==end}
    inputs={key(r):r for r in loaded if int(r['accepted_step'])==end and r['row_class']=='external_support_contact'}
    if len(points)!=len(sections) or set(inputs)!=set(points):raise ValueError('Unmatched accepted support point')
    records=[]
    for row in sections:
        idx,source=key(row);p=points[idx,source];f=inputs[idx,source]
        if int(p['pre_step_index'])!=end-1 or int(p['body_index'])!=int(f['body_index']):
            raise ValueError('Stale point or changed body identity')
        normal=tuple(float(f['normal_'+axis]) for axis in 'xyz')
        if normal!=(0.0,0.0,1.0):raise ValueError('Requires the registered horizontal bed at zero')
        x,y,z=(float(p['point_world_'+a+'_m']) for a in 'xyz');load=float(f['normal_impulse_ns'])/dt
        physical=recover(float(row['skin_thickness_m']),float(row['fat_thickness_m']),
                         float(row['skin_compliance_m_per_n']),float(row['fat_compliance_m_per_n']),z,load)
        records.append(dict(accepted_step=end,contact_index=idx,source_geometry_index=source,body_index=p['body_index'],
                       x_m=x,y_m=y,rigid_marker_z_m=z,measured_normal_N=load,**physical))
    with output.open('w',newline='') as stream:
        out=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator="\n");out.writeheader();out.writerows(records)
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();import json
    args=json.loads((a.run/'launch.json').read_text())['argv']
    run(a.run,a.output,float(args[args.index('--muscle-step-seconds')+1]))

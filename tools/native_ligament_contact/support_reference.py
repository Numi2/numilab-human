"""Compare archived, source-identified bed sections with the full scene's release witnesses."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def option(config, name):
    argv=config['argv']
    return argv[argv.index(name)+1]

def compare(reference, release):
    a,b=[json.loads((p/'launch.json').read_text()) for p in (reference,release)]
    payloads=[Path(option(c,'--support-contact-payload')) for c in (a,b)]
    if digest(payloads[0])!=digest(payloads[1]) or option(a,'--resting-reference-mass-kg')!=option(b,'--resting-reference-mass-kg'):
        raise ValueError('Different support source or reference mass')
    dt=float(option(a,'--muscle-step-seconds'))
    forces=read(reference/'resting-com-support-impulses.csv')
    sections=read(reference/'resting-local-bed-sections.csv')
    witnesses=read(release/'resting-support-selected-skin-witness.csv')
    end=max(int(r['accepted_step']) for r in forces)
    by_source=defaultdict(dict)
    for r in forces:
        step=int(r['accepted_step'])
        if (end-step)*dt>=2 or r['row_class']!='external_support_contact':continue
        key=(int(r['contact_index']),int(r['source_geometry_index']))
        if step in by_source[key]:raise ValueError('Duplicate support sample')
        by_source[key][step]=float(r['normal_impulse_ns'])/dt
    first=min(int(row['pre_dynamics_step_index']) for row in witnesses)
    sites={(int(r['region']),int(r['source_geometry_index'])):r for r in witnesses if int(r['pre_dynamics_step_index'])==first}
    if len(sites)!=32 or set(by_source)!=set(sites):raise ValueError('NHCNT region identity mismatch')
    records=[];steps=set(next(iter(by_source.values())))
    for section in sections:
        key=int(section['contact_index']),int(section['source_geometry_index'])
        samples=by_source[key]
        if set(samples)!=steps:raise ValueError('Missing late support samples')
        f=math.fsum(samples.values())/len(samples)
        w=sites[key];gap=float(w['selected_point_world_z_m'])
        compression=f*(float(section['skin_compliance_m_per_n'])+float(section['fat_compliance_m_per_n']))
        records.append(dict(region=key[0],source_geometry_index=key[1],reference_sample_mean_N=f,
                            reference_elastic_compression_m=compression,full_release_observer_height_m=gap))
    if len(set(r['region'] for r in records))!=len(sites):raise ValueError('Incomplete section roster')
    total=math.fsum(r['reference_sample_mean_N'] for r in records)
    return dict(scope='Mean of sampled last-step forces in the reduced native control, compared by NHCNT identity to the full release observer. Selected skin points are observational and can differ between topologies. Compression uses the reference elastic law and excludes damping.',support_sha256=digest(payloads[0]),reference_launch_sha256=digest(reference/'launch.json'),full_launch_sha256=digest(release/'launch.json'),release_observer_pre_step=first,late_sample_count=len(steps),sampled_reference_force_N=total,
                force_fraction_on_regions_with_full_observer_over_2mm=math.fsum(r['reference_sample_mean_N'] for r in records if r['full_release_observer_height_m']>.002)/total,
                force_fraction_on_regions_with_full_observer_over_5mm=math.fsum(r['reference_sample_mean_N'] for r in records if r['full_release_observer_height_m']>.005)/total,rows=sorted(records,key=lambda r:-r['reference_sample_mean_N']))

if __name__=='__main__':
    reference, release, output=map(Path,sys.argv[1:])
    output.write_text(json.dumps(compare(reference,release),indent=2,allow_nan=False)+'\n')

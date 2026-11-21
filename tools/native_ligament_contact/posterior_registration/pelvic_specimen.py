"""Read the existing passive specimen parameters used for authoring checks."""
from pathlib import Path
import re
def properties(path):
    src=Path(path).read_text()
    if 'energy = neo_hookean(mu, lambda);' not in src:raise ValueError('Unknown material law')
    def number(pattern):
        m=re.findall(pattern,src)
        if len(m)!=1:raise ValueError(pattern)
        return float(m[0])
    return dict(density=number(r'parameter density : kg/m\^3 = ([0-9.]+)'),mu=1000*number(r'parameter mu : kPa = ([0-9.]+)'),lam=1000*number(r'parameter lambda : kPa = ([0-9.]+)'),jmin=number(r'minimum_J = ([0-9.]+)'),jmax=number(r'maximum_J = ([0-9.]+)'))

"""Refresh exact geometry as the active pressure and material solve changes."""
from pathlib import Path
import shutil
src=Path('nodal-equilibrium-owner');dst=Path('active-facet-owner');dst.mkdir(exist_ok=True)
for p in src.iterdir():
    if p.is_file():shutil.copyfile(p,dst/p.name)
p=dst/'pelvic_end_geometry.metalinc';s=p.read_text()
old='bool test=!control[0]&&!needed[0]&&!needed[1];';assert s.count(old)==1
s=s.replace(old,'''// A stalled material solve may be missing a newly intersecting facet.
    // Revisit the actual geometry at two bounded work transitions. A clear
    // observation remains provisional while either side can still move.
    bool test=!control[0] && (d.z || (!needed[0] && !needed[1]));''')
old='if(r[0]==0)control[0]=1;';assert s.count(old)==1;s=s.replace(old,'if(r[0]==0)control[0]=(!needed[0] && !needed[1]);')
p.write_text(s)
p=dst/'NumiHumanPelvicContact.hpp';s=p.read_text();old='words(0,{nCrossingGroups,unsigned(outer==UINT_MAX),0,0})';assert s.count(old)==1
p.write_text(s.replace(old,'words(0,{nCrossingGroups,unsigned(outer==UINT_MAX),unsigned(outer==15u || outer==31u),0})'))

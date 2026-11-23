"""Attach registered lengths to the existing native contact, presentation and source-table owners."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from material_context import sha
from export_contact_columns import change
from remap_pose_tables import build as rebuild_tables


def build(base,fields,stage,out,sourcefile):
    ids=json.loads((base/"identities.json").read_text())
    cfg=json.loads((fields/"assignment.json").read_text());stock=fields/"registered-stock.csv"
    if (sha(stock)!=cfg["stock"] or stage.exists() or out.exists() or any(sha(base/k)!=v for k,v in ids.items())):
        raise ValueError("Changed build inputs or existing destination")
    subprocess.run(["cp","-cR",base,stage],check=True)
    header="NumiHumanRegisteredBedStock.hpp"; shutil.copy2(sourcefile,stage/header)
    p=stage/"NumiHumanRestingVisual.hpp";s=p.read_text()
    anchor='#include "NumiHumanRestingSupportGeometry.hpp"';s=change(s,anchor,anchor+'\n#include "'+header+'"')
    anchor="                for (unsigned v = 0; v < posedSkin.size(); ++v) {\n                    if (!resolvedWitness"
    new='''                const char* registeredInput=std::getenv("NUMI_HUMAN_REGISTERED_BED_STOCK_CSV");
                require(resolvedWitness && registeredInput,"registered bed material requires its certified column source");
                const auto lengths=numiReadBedLengths(registeredInput,posedSkin.size(),"DIGEST",ROWCOUNT);
'''.replace('DIGEST',cfg['stock']).replace('ROWCOUNT',str(cfg['row_count']))
    s=change(s,anchor,new+anchor)
    a=s.index("                    if(resolvedWitness)resolvedStock[v]=")
    e=s.index("                for (std::size_t r = 0; r < regions.size(); ++r) {",a)
    body=s[a:e].replace("kWholeBodySkinThicknessM[nearest]","skinM").replace("kWholeBodyFatThicknessM[nearest]","fatM").replace("kWholeBodySkinSourceVertices[nearest]","stockSource")
    new='''                    const auto& reg=lengths[v];
                    double skinM=kWholeBodySkinThicknessM[nearest],fatM=kWholeBodyFatThicknessM[nearest];
                    unsigned stockSource=kWholeBodySkinSourceVertices[nearest];
                    if(reg.present) {
                        const auto& p=posedSkin[v];
                        require(reg.row==row && reg.geometry==regions[row].sourceGeometryIndex &&
                            std::abs(p.x-reg.x)<5e-7 && std::abs(p.y-reg.y)<5e-7 && std::abs(p.z-reg.z)<5e-7 &&
                            float(reg.area)==float(fullAreas[v]),"registered bed material belongs to a different footprint");
                        skinM=reg.skin;fatM=reg.fat;stockSource=std::numeric_limits<unsigned>::max();
                    }
'''
    s=s[:a]+new+body+s[e:]
    s=change(s,'assignment=within_region_nearest_parallel_material_columns','assignment=registered_contiguous_cells_with_retained_prior_columns')
    p.write_text(s)
    ids[p.name]=sha(p);ids[header]=sha(stage/header)
    (stage/"identities.json").write_text(json.dumps(ids,indent=2)+"\n")
    rebuild_tables(stage,out,fields)


if __name__=="__main__":build(*(Path(s).resolve() for s in sys.argv[1:]))

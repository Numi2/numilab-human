"""Bound what interior-only elimination can save on the delivered physical mesh."""
import json
from pathlib import Path
import numpy as np
from compile_connected_joint import digest


def run(study):
    volume = study/'volume'
    walls = volume/'material-walls.npz'
    shape = study/'motion-quality.json'
    importer = json.loads((volume/'native-import.json').read_bytes())
    witness = json.loads(shape.read_bytes())
    points = importer['nodes']
    incident = np.zeros(points,np.uint8)
    families = ('O','I','D','C')
    per_wall = {}
    with np.load(walls) as data:
        if set(data.files) != set(families):
            raise ValueError('Unexpected interface family')
        for index,name in enumerate(families):
            faces = data[name]
            if faces.ndim!=2 or faces.shape[1]!=3 or faces.min()<0 or faces.max()>=points:
                raise ValueError('Invalid wall connectivity')
            nodes = np.unique(faces)
            incident[nodes] |= 1<<index
            per_wall[name] = len(nodes)
    # For this upper bound, every node outside the four interfaces is considered
    # eliminable, even though attachments and regional boundaries can impose more restrictions.
    protected = int(np.count_nonzero(incident))
    cells = np.array([r['nodes'] for r in witness['witnesses']],dtype=np.int32)
    counts = np.bincount((incident[cells]!=0).sum(axis=1),minlength=5)
    out = dict(native_import_sha256=digest(volume/'native-import.json'),elements_sha256=importer['element_sha256'],wall_sha256=digest(walls),quality_sha256=digest(shape),nodes=points,material_wall_nodes=protected,maximum_eliminable_interior_nodes=points-protected,maximum_eliminable_fraction=(points-protected)/points,maximum_coordinate_count_ratio=points/protected,wall_node_counts=per_wall,affected_cells_by_number_of_wall_vertices={str(i):int(k) for i,k in enumerate(counts)},scope='Coordinate count bound for exact interior elimination when all four registered material walls remain independent. This is neither a runtime bound nor a restriction on embedded wall quadrature.')
    return out

if __name__=='__main__':
    study=Path(__file__).resolve().parent/'witness-material'
    (study/'interface-coverage.json').write_text(json.dumps(run(study),indent=2)+'\n')

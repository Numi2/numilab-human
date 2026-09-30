"""Render real Rodero case18 node projections and electrical DOF ownership."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from numilab_human import model as human


ROOT = human.REPOSITORY_ROOT
HERE = Path(__file__).resolve().parent
ASSET = ROOT/'Build/cardiac-electrical-source-20260930/asset'
COLORS = {1: '#e67a87', 2: '#ffc47b',
          4: '#56d5c6', 8: '#9ab0ff'}


def main() -> None:
    source = human.read_json(ASSET/'manifest.json')
    nodes_file = ASSET/'nodes.f64le'
    assert human.sha256(nodes_file) == source['buffers'][nodes_file.name]['sha256']
    nodes = np.fromfile(nodes_file, '<f8').reshape(-1, 3)
    ids = np.fromfile(HERE/'electrical-dof-source-nodes.u32le', '<u4')
    regions = np.fromfile(HERE/'electrical-dof-region-labels.u32le', '<u4')
    assert len(nodes) == 300965 and len(ids) == len(regions) == 281704
    mask = np.zeros(len(nodes), dtype=np.uint8)
    np.bitwise_or.at(mask, ids, (1 << (regions-1)).astype(np.uint8))
    assert np.count_nonzero(mask) == 274315

    picture = Image.new('RGB', (1800, 1100), '#0b1424')
    draw = ImageDraw.Draw(picture)
    regular = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 23)
    bold = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 35)
    draw.text((83, 29), 'REAL HEART SOURCE  /  ELECTRICAL DOMAIN OWNERSHIP',
              font=bold, fill='#eef4ff')
    draw.text((83, 78),
              'Rodero case18 node projection; shared geometric nodes get separate region-specific electrical DOFs',
              font=regular, fill='#b2bfd2')
    panels = [(0, 2, 'SOURCE FRONT  /  x,z', (95, 156, 850, 936)),
              (1, 2, 'SOURCE SIDE  /  y,z', (950, 156, 1705, 936))]
    for axis, _, label, bounds in panels:
        x0, y0, x1, y1 = bounds
        draw.rectangle(bounds, outline='#38516b', width=2)
        draw.text((x0+18, y0+13), label, font=regular, fill='#eef4ff')
        low = nodes[:, [axis, 2]].min(axis=0)
        high = nodes[:, [axis, 2]].max(axis=0)
        mid = (low+high)/2
        scale = min((x1-x0-65)/(high[0]-low[0]),
                    (y1-y0-105)/(high[1]-low[1]))

        def project(point):
            return (round((x0+x1)/2+(point[axis]-mid[0])*scale),
                    round((y0+y1)/2-(point[2]-mid[1])*scale))

        for code in (0, 1, 2, 4, 8):
            color = '#405063' if code == 0 else COLORS[code]
            for point in nodes[mask == code][::2 if code else 4]:
                draw.point(project(point), fill=color)
        shared = (mask != 0) & ((mask & (mask-1)) != 0)
        for point in nodes[shared]:
            x, y = project(point)
            draw.ellipse((x-1, y-1, x+1, y+1), fill='#eef4ff')
    legend = [('LV', 1), ('RV', 2), ('LA', 4), ('RA', 8)]
    for i, (name, bit) in enumerate(legend):
        x = 110+i*190
        draw.ellipse((x, 975, x+18, 993), fill=COLORS[bit])
        draw.text((x+29, 967), name, font=regular, fill='#eef4ff')
    draw.ellipse((875, 975, 893, 993), fill='#eef4ff')
    draw.text((906, 967), 'shared source node', font=regular, fill='#eef4ff')
    draw.text((85, 1030),
              '1,337,558 source myocardial tetrahedra  ·  281,704 distinct electrical DOFs  ·  0 admitted inter-region links',
              font=regular, fill='#b2bfd2')
    picture.save(HERE/'cardiac-electrical-region-map.png', optimize=True)


if __name__ == '__main__':
    main()

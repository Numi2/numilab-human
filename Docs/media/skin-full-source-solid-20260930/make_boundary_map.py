"""Render a source-coordinate map of the selected outer visual skin openings."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from numilab_human import model as human
from numilab_human.skin_embeddedness_gate import _source_mesh


ROOT = human.REPOSITORY_ROOT
HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'Build/skin-seam-continuity-20260929/production/payload'


def main() -> None:
    payload = SOURCE/'bodyparts3d-myosim-skinned-shell.nhskin'
    manifest = human.read_json(SOURCE/'bodyparts3d-myosim-skinned-shell.manifest.json')
    _, _, _, _, source, faces = _source_mesh(payload, manifest)
    points, inverse = np.unique(source, axis=0, return_inverse=True)
    quotient_faces = inverse[faces]
    edges = np.sort(np.concatenate((quotient_faces[:, [0, 1]],
                                    quotient_faces[:, [1, 2]],
                                    quotient_faces[:, [2, 0]])), axis=1)
    count = Counter(map(tuple, edges.tolist()))
    boundary = np.array([pair for pair, occurrences in count.items()
                         if occurrences == 1], dtype=np.int64)
    assert len(boundary) == 171

    picture = Image.new('RGB', (1800, 1100), '#0b1424')
    draw = ImageDraw.Draw(picture)
    regular = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 24)
    bold = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 34)
    draw.text((85, 29), 'SOURCE SKIN: WHERE OUTER-ONLY SELECTION IS OPEN',
              font=bold, fill='#eef4ff')
    draw.text((85, 75),
              'Actual FJ2810 source vertices and edges; complete source solid has zero boundary edges',
              font=regular, fill='#b2bfd2')

    for axis, label, box in ((0, 'SOURCE FRONT  /  x,z', (95, 155, 810, 1015)),
                             (1, 'SOURCE SIDE  /  y,z', (990, 155, 1705, 1015))):
        x0, y0, x1, y1 = box
        draw.rectangle(box, outline='#38516b', width=2)
        draw.text((x0+18, y0+12), label, font=regular, fill='#56d5c6')
        dims = (axis, 2)
        low = points[:, dims].min(axis=0)
        high = points[:, dims].max(axis=0)
        scale = min((x1-x0-50)/(high[0]-low[0]),
                    (y1-y0-85)/(high[1]-low[1]))
        mid = (low+high)/2

        def project(point):
            return (round((x0+x1)/2+(point[axis]-mid[0])*scale),
                    round((y0+y1)/2-(point[2]-mid[1])*scale))

        for point in points[::2]:
            draw.point(project(point), fill='#435166')
        for first, second in boundary:
            draw.line((project(points[first]), project(points[second])),
                      fill='#ff8e83', width=4)
    draw.text((99, 1034),
              'OUTER VISUAL SHEET  ·  109,183 faces  ·  171 red boundary edges',
              font=regular, fill='#ff8e83')
    draw.text((975, 1034),
              'FULL SOURCE SOLID  ·  203,382 faces  ·  0 boundary edges',
              font=regular, fill='#56d5c6')
    picture.save(HERE/'source-boundary-map.png', optimize=True)


if __name__ == '__main__':
    main()

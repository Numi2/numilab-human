"""Render the measured eight-organ native separation matrix for board review."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
receipt = json.loads((HERE/'receipt-v2.json').read_text())
assert receipt['status'] == 'failed_cross_surface_separation'
assert receipt['pair_count'] == 28 and receipt['clear_pair_count'] == 21
names = [row['label'] for row in receipt['organs']]
assert names == ['stomach', 'pancreas', 'right kidney', 'left kidney', 'spleen',
                 'gallbladder', 'left adrenal gland', 'right adrenal gland']
pairs = {frozenset((row['first'], row['second'])): row for row in receipt['pairs']}
assert len(pairs) == 28
crossing_count = sum(row['exact_segment_or_polygon_crossing_pairs'] for row in receipt['pairs'])
assert crossing_count == 814

regular = '/System/Library/Fonts/Supplemental/Arial.ttf'
bold = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'


def font(size: int, strong: bool = False):
    return ImageFont.truetype(bold if strong else regular, size)


canvas = Image.new('RGB', (1920, 1080), '#0b1424')
draw = ImageDraw.Draw(canvas)
white, muted, teal, amber, coral = '#eef2fa', '#aebed2', '#58cfbf', '#ffc77f', '#f48688'
draw.text((78, 43), 'NUMI HUMAN  /  INTERNAL ORGAN PLACEMENT', fill=teal, font=font(29, True))
draw.text((78, 92), 'Eight organ surfaces, seven crossing pairs',
          fill=white, font=font(53, True))
draw.text((82, 169), 'Exact compiled Float32 mesh audit  |  shared native Abdomen frame',
          fill=muted, font=font(27))

draw.rounded_rectangle((70, 231, 1417, 842), radius=18,
                       fill='#152238', outline='#33425c', width=2)
draw.text((103, 257), 'Exact intersecting triangle pairs', fill=white, font=font(32, True))

short = ['Stomach', 'Pancreas', 'R kidney', 'L kidney', 'Spleen',
         'Gallbladder', 'L adrenal', 'R adrenal']
left, top, step_x, step_y = 337, 367, 132, 57
for col, name in enumerate(short):
    x = left + step_x*col
    draw.text((x+4, 328), name, fill=muted, font=font(18, True))
for row, name in enumerate(short):
    draw.text((106, top+step_y*row+11), name, fill=white, font=font(23, True))
    for col in range(8):
        x, y = left+step_x*col, top+step_y*row
        if row == col:
            color, label, ink = '#24364d', 'SELF', muted
        elif row > col:
            color, label, ink = '#18283f', '', muted
        else:
            result = pairs[frozenset((names[row], names[col]))]
            count = result['exact_intersecting_triangle_pairs']
            color, label, ink = ('#6e3445', str(count), coral) if count else ('#1c514f', '0', teal)
        draw.rounded_rectangle((x, y, x+119, y+45), radius=10, fill=color)
        if label:
            size = 22 if row != col else 16
            box = draw.textbbox((0, 0), label, font=font(size, True))
            draw.text((x+(119-(box[2]-box[0]))/2,
                       y+(45-(box[3]-box[1]))/2-2), label, fill=ink,
                      font=font(size, True))

draw.rounded_rectangle((1441, 231, 1850, 842), radius=18,
                       fill='#152238', outline='#33425c', width=2)
draw.text((1474, 268), 'Gate result', fill=white, font=font(34, True))
draw.text((1474, 344), '21 / 28', fill=teal, font=font(49, True))
draw.text((1474, 409), 'pairs separate', fill=muted, font=font(24))
draw.line((1474, 467, 1810, 467), fill='#33425c', width=2)
draw.text((1474, 502), '7 / 28', fill=coral, font=font(49, True))
draw.text((1474, 568), 'pairs cross', fill=muted, font=font(24))
draw.line((1474, 628, 1810, 628), fill='#33425c', width=2)
draw.text((1474, 668), '814', fill=amber, font=font(49, True))
draw.text((1474, 729), 'triangle-pair crossings', fill=muted, font=font(22))
draw.text((1474, 775), '7 strict witnesses', fill=muted, font=font(21))

draw.rounded_rectangle((70, 865, 1850, 981), radius=16, fill='#283142')
draw.text((108, 887), 'Admission blocked', fill=amber, font=font(29, True))
draw.text((108, 930), 'Independent source registration is needed; no organs were shifted to make this pass.',
          fill=white, font=font(26))
draw.text((78, 1015), '29 Sep 2026  |  BodyParts3D CC-BY-SA 2.1 Japan  |  selected 602-surface native packet',
          fill='#8293ab', font=font(21))
canvas.save(HERE/'executive-organ-separation-v2.png', optimize=True)

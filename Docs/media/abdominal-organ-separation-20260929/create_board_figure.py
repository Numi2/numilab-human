"""Draw the measured five-organ exact separation matrix for board review."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
receipt = json.loads((HERE/'receipt-v1.json').read_text())
assert receipt['status'] == 'failed_cross_surface_separation'
assert receipt['pair_count'] == 10 and receipt['clear_pair_count'] == 6
names = [row['label'] for row in receipt['organs']]
assert names == ['stomach', 'pancreas', 'right kidney', 'left kidney', 'spleen']
pairs = {frozenset((row['first'], row['second'])): row for row in receipt['pairs']}

regular = '/System/Library/Fonts/Supplemental/Arial.ttf'
bold = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
def font(size: int, strong: bool = False):
    return ImageFont.truetype(bold if strong else regular, size)


canvas = Image.new('RGB', (1920, 1080), '#0b1424')
draw = ImageDraw.Draw(canvas)
white, muted, teal, amber, coral = '#eef2fa', '#aebed2', '#58cfbf', '#ffc77f', '#f48688'
draw.text((78, 43), 'NUMI HUMAN  /  INTERNAL ORGAN PLACEMENT', fill=teal, font=font(29, True))
draw.text((78, 92), 'Five self-embedded surfaces, four crossings',
          fill=white, font=font(57, True))
draw.text((82, 177), 'Exact compiled Float32 surface-to-surface audit  |  one native Abdomen owner',
          fill=muted, font=font(29))

draw.rounded_rectangle((70, 245, 1240, 832), radius=18,
                       fill='#152238', outline='#33425c', width=2)
draw.text((110, 276), 'Exact intersecting triangle pairs', fill=white, font=font(36, True))

left = 360
top = 380
step_x = 166
step_y = 82
short = ['Stomach', 'Pancreas', 'R kidney', 'L kidney', 'Spleen']
for col, name in enumerate(short):
    draw.text((left+step_x*col+7, 345), name, fill=muted, font=font(22, True))
for row, name in enumerate(short):
    draw.text((110, top+step_y*row+20), name, fill=white, font=font(25, True))
    for col in range(5):
        x = left + step_x*col
        y = top + step_y*row
        if row == col:
            color, label, ink = '#24364d', 'SELF', muted
        elif row > col:
            color, label, ink = '#18283f', '', muted
        else:
            result = pairs[frozenset((names[row], names[col]))]
            count = result['exact_intersecting_triangle_pairs']
            color, label, ink = ('#6e3445', str(count), coral) if count else ('#1c514f', '0', teal)
        draw.rounded_rectangle((x, y, x+145, y+62), radius=12, fill=color)
        if label:
            box = draw.textbbox((0, 0), label, font=font(29 if row != col else 20, True))
            draw.text((x+(145-(box[2]-box[0]))/2,
                       y+(62-(box[3]-box[1]))/2-3), label, fill=ink,
                      font=font(29 if row != col else 20, True))

draw.rounded_rectangle((1270, 245, 1850, 832), radius=18,
                       fill='#152238', outline='#33425c', width=2)
draw.text((1310, 280), 'Gate result', fill=white, font=font(37, True))
draw.text((1310, 357), '6 / 10', fill=teal, font=font(58, True))
draw.text((1310, 426), 'pairs separate', fill=muted, font=font(27))
draw.line((1310, 480, 1810, 480), fill='#33425c', width=2)
draw.text((1310, 515), '4 / 10', fill=coral, font=font(58, True))
draw.text((1310, 582), 'pairs cross', fill=muted, font=font(27))
draw.line((1310, 635, 1810, 635), fill='#33425c', width=2)
draw.text((1310, 672), '182 exact crossing', fill=amber, font=font(29, True))
draw.text((1310, 715), 'triangle pairs', fill=muted, font=font(27))
draw.text((1310, 767), 'Longest segment: 4.814 mm', fill=muted, font=font(22))

draw.rounded_rectangle((70, 862, 1850, 981), radius=16, fill='#283142')
draw.text((110, 885), 'Admission blocked', fill=amber, font=font(29, True))
draw.text((110, 929), 'Atlas organ domains are not disjoint. No surfaces moved; clinical registration, physical volume and mechanics remain open.',
          fill=white, font=font(25))
draw.text((78, 1015), '29 Sep 2026  |  BodyParts3D CC-BY-SA 2.1 Japan  |  selected 602-surface native packet',
          fill='#8293ab', font=font(21))
canvas.save(HERE/'executive-organ-separation.png', optimize=True)

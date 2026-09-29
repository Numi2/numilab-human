"""Make a board-readable chart directly from the audited tendon delta receipt."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
receipt = json.loads((HERE/'receipt-v1.json').read_text())
right, left = receipt['tendons']
assert [right['stable_id'], left['stable_id']] == [7, 8]
assert [right['old_exact_intersection_pairs'], left['old_exact_intersection_pairs']] == [145, 163]
assert [right['new_exact_intersection_pairs'], left['new_exact_intersection_pairs']] == [0, 0]

regular = '/System/Library/Fonts/Supplemental/Arial.ttf'
bold = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
def font(size: int, strong: bool = False):
    return ImageFont.truetype(bold if strong else regular, size)


canvas = Image.new('RGB', (1920, 1080), '#0b1424')
draw = ImageDraw.Draw(canvas)
white, muted, teal, violet, amber = '#eef2fa', '#aebed2', '#58cfbf', '#a69cfb', '#ffc77f'

draw.text((80, 45), 'NUMI HUMAN  /  CALCANEAL TENDON GEOMETRY', fill=teal, font=font(29, True))
draw.text((80, 93), 'Compiled tendon intersections removed', fill=white, font=font(63, True))
draw.text((82, 178), 'Source-bound visual repair  |  full 150-surface Float32 exact-predicate census',
          fill=muted, font=font(30))

draw.rounded_rectangle((75, 245, 1170, 795), radius=18, fill='#152238', outline='#33425c', width=2)
draw.text((110, 281), 'Exact self-intersection pairs', fill=white, font=font(39, True))
draw.text((110, 334), 'Prior generated-strip payload vs current source-bound boundary fit',
          fill=muted, font=font(25))

for y, label, row, color in [(430, 'RIGHT', right, teal), (610, 'LEFT', left, violet)]:
    draw.text((111, y), label, fill=white, font=font(28, True))
    draw.text((250, y-3), f"{row['member_id']}", fill=muted, font=font(25))
    draw.text((111, y+52), 'PRIOR', fill=muted, font=font(23, True))
    width = int(690 * row['old_exact_intersection_pairs'] / 180)
    draw.rounded_rectangle((250, y+48, 250+width, y+90), radius=10, fill='#f48688')
    draw.text((275+width, y+48), str(row['old_exact_intersection_pairs']), fill=white,
              font=font(28, True))
    draw.text((111, y+110), 'NOW', fill=color, font=font(23, True))
    draw.rounded_rectangle((250, y+109, 940, y+150), radius=10, fill='#24364d')
    draw.text((960, y+107), '0', fill=color, font=font(33, True))

draw.rounded_rectangle((1200, 245, 1845, 795), radius=18, fill='#152238', outline='#33425c', width=2)
draw.text((1240, 281), 'Unresolved geometry', fill=white, font=font(38, True))
draw.text((1240, 353), 'OPEN BOUNDARY EDGES', fill=muted, font=font(24, True))
draw.text((1240, 393), f"{right['open_boundary_edge_count']} right  /  {left['open_boundary_edge_count']} left",
          fill=amber, font=font(34, True))
draw.line((1240, 467, 1800, 467), fill='#33425c', width=2)
draw.text((1240, 500), 'LOCALLY REVERSED FACES', fill=muted, font=font(24, True))
draw.text((1240, 542), f"{len(right['source_face_normal_reversal_ids'])} right  /  "
          f"{len(left['source_face_normal_reversal_ids'])} left",
          fill=amber, font=font(34, True))
draw.line((1240, 616, 1800, 616), fill='#33425c', width=2)
draw.text((1240, 645), 'SURFACES CHANGED', fill=muted, font=font(24, True))
draw.text((1240, 686), '2 of 150', fill=teal, font=font(34, True))
draw.text((1240, 739), '148 local geometries unchanged', fill=muted, font=font(23))

draw.rounded_rectangle((75, 830, 1845, 982), radius=16, fill='#283142')
draw.text((110, 860), 'Evidence boundary', fill=amber, font=font(27, True))
draw.text((110, 903), 'Visual surface QA only. Open tendon sheets, local face reversals, anatomical enthesis,',
          fill=white, font=font(28))
draw.text((110, 941), 'force transfer, loaded contact and sustained standing remain unqualified.',
          fill=white, font=font(28))

draw.text((80, 1015), '29 Sep 2026  |  BodyParts3D CC-BY-SA 2.1 Japan  |  '
          'NHTISS4 v5 source-bound exact-predicate audit', fill='#8293ab', font=font(21))

canvas.save(HERE/'executive-tendon-geometry.png', optimize=True)

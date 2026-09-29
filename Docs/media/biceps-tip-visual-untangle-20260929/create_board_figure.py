"""Render measured bilateral muscle tip geometry change for board review."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
receipt = json.loads((HERE/'receipt-v1.json').read_text())
assert receipt['changed_vertex_record_count'] == 32
assert receipt['single_embedded_muscle_surfaces_before'] == 52
assert receipt['single_embedded_muscle_surfaces_after'] == 54
assert [row['member_id'] for row in receipt['changed_surfaces']] == ['FJ1444', 'FJ1444M']

regular = '/System/Library/Fonts/Supplemental/Arial.ttf'
bold = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'


def font(size: int, strong: bool = False):
    return ImageFont.truetype(bold if strong else regular, size)


canvas = Image.new('RGB', (1920, 1080), '#0b1424')
draw = ImageDraw.Draw(canvas)
white, muted, teal, amber, coral = '#eef2fa', '#aebed2', '#58cfbf', '#ffc77f', '#f48688'
draw.text((78, 44), 'NUMI HUMAN  /  MUSCLE SURFACE GEOMETRY', fill=teal, font=font(29, True))
draw.text((78, 96), 'Bilateral biceps femoris tips untangled', fill=white, font=font(53, True))
draw.text((80, 173), 'Pinned BodyParts3D source  |  exact compiled Float32 intersection audit',
          fill=muted, font=font(28))

draw.rounded_rectangle((70, 240, 1850, 771), radius=18,
                       fill='#152238', outline='#33425c', width=2)
draw.text((108, 276), 'Exact self-intersecting triangle pairs', fill=white,
          font=font(38, True))
draw.text((660, 354), 'BEFORE', fill=muted, font=font(28, True))
draw.text((1255, 354), 'DERIVED VISUAL', fill=muted, font=font(28, True))
for i, (name, row) in enumerate(zip(('Right short head', 'Left short head'),
                                    receipt['changed_surfaces'], strict=True)):
    y = 438+i*134
    draw.text((108, y+27), name, fill=white, font=font(31, True))
    draw.rounded_rectangle((642, y, 1033, y+91), radius=14, fill='#6e3445')
    draw.rounded_rectangle((1228, y, 1619, y+91), radius=14, fill='#1c514f')
    draw.text((804, y+15), str(row['old_exact_intersection_pairs']),
              fill=coral, font=font(54, True))
    draw.text((1390, y+15), str(row['new_exact_intersection_pairs']),
              fill=teal, font=font(54, True))
    draw.line((1052, y+45, 1203, y+45), fill=amber, width=7)
    draw.polygon([(1203, y+45), (1176, y+29), (1176, y+61)], fill=amber)

draw.rounded_rectangle((70, 798, 870, 965), radius=16, fill='#1c514f')
draw.text((109, 823), '52  →  54', fill=teal, font=font(56, True))
draw.text((109, 902), 'single embedded muscle visual candidates',
          fill=white, font=font(25))
draw.rounded_rectangle((895, 798, 1850, 965), radius=16, fill='#283142')
draw.text((935, 824), '≈ 0.065 mm', fill=amber, font=font(53, True))
draw.text((935, 899), 'maximum source-point movement  |  12 positions total',
          fill=white, font=font(24))
draw.text((78, 1005), 'Visual-only repair. Source OBJ, face indices, skinning weights and muscle force paths unchanged; clinical anatomy remains open.',
          fill=muted, font=font(22))
canvas.save(HERE/'executive-biceps-tip-untangle.png', optimize=True)

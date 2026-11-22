"""Read each region's immutable density range from the native material law."""

import json, re
from pathlib import Path


def density_bounds(owner, material):
    macro = {
        15: "NUMI_HUMAN_SKIN_DERMIS_MATERIAL",
        16: "NUMI_HUMAN_SUBCUTANEOUS_ADIPOSE_MATERIAL",
    }[int(material)]
    cmd = json.loads((owner / "identities.json").read_text())["commands"][0]
    (path,) = [
        Path(arg.split("=", 1)[1].strip('"'))
        for arg in cmd
        if arg.startswith("-D" + macro + "=")
    ]
    text = re.sub(r"//[^\n]*", "", path.read_text())
    match = re.search(
        r"parameter\s+density\s*:\s*kg/m\^3\s*=\s*[\d.eE+-]+\s+in\s*\[\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\]",
        text,
    )
    if not match:
        raise ValueError("Unknown constitutive density declaration")
    return tuple(map(float, match.groups()))

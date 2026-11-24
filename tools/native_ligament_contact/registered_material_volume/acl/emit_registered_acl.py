import argparse
import hashlib
from pathlib import Path

# This source cohort has one registered bilateral ACL owner and one coupled bed owner.
base_hash = '70fce41cbeea274802ce9c5992b112d77cd5b06cd2cab7cde5730a3cc93aab5d'
result_hash = '44653bf12fcb4b8c6a27305b5863ef3fd9cbc7746e2c299819b7365fbb43b05a'
anchors = {'prepare.inc': '            source.materials.push_back(std::move(built.material));\n            source.objects.push_back(built.object);', 'verify_compiled.inc': '    std::uint32_t aclFirstNode=0u;'}


def install(base, dest):
    source = base.read_bytes()
    if hashlib.sha256(source).hexdigest() != base_hash:
        raise ValueError('The registered material seating source revision differs')
    for filename, anchor in anchors.items():
        needle = anchor.encode()
        if source.count(needle) != 1:
            raise ValueError('The native ACL insertion point changed')
        source = source.replace(needle, (Path(__file__).parent / filename).read_bytes() + needle)
    if hashlib.sha256(source).hexdigest() != result_hash:
        raise ValueError('The compiled native source could not be reproduced')
    with dest.open('xb') as stream:
        stream.write(source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Emit the exact native ACL source used by the registered M6 scene.')
    parser.add_argument('base', type=Path)
    parser.add_argument('dest', type=Path)
    args = parser.parse_args()
    install(args.base, args.dest)

"""Shared exact ray parity across open face partitions of closed surfaces."""
from fractions import Fraction as F
from itertools import product
import pytest
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.model import ImportError as HumanImportError

V = [(0,0,0),(4,0,0),(0,4,0),(0,0,4)]
FACES = [(0,2,1),(0,1,3),(0,3,2),(1,2,3)]


@pytest.mark.parametrize("chunk_size", [1,2,3,4])
def test_chunks_match_whole_index_including_ray_retries(chunk_size):
    records = ci._records(V, FACES)
    full = ci.prepare_point_location(records)
    chunks = tuple(ci.prepare_point_location(records[i:i+chunk_size])
                   for i in range(0, len(records), chunk_size))
    retries = 0
    for point in product((F(-1),F(0),F(1,4),F(1),F(4)), repeat=3):
        expected = ci.point_location_prepared_indexed(point, full)
        actual = ci.point_location_prepared_indexed_chunks(point, chunks)
        assert actual == expected
        retries += int(expected.get("attempts",0) > 1)
    assert retries > 0


def test_ambiguity_in_one_chunk_retries_the_entire_partition():
    # Outside point primary ray hits tetrahedron corner. Chunks resolve their
    # own first rays differently; only a common global retry preserves counts.
    records = ci._records(V, FACES)
    chunks = tuple(ci.prepare_point_location([record]) for record in records)
    point = (F(-1),)*3
    expected = ci.point_location_prepared_indexed(point, ci.prepare_point_location(records))
    assert expected["attempts"] > 1
    independent = [ci.point_location_prepared_indexed(point, chunk) for chunk in chunks]
    assert len({tuple(result["ray"]) for result in independent}) > 1
    assert sum(result["crossings"] for result in independent) % 2 == 1
    assert expected["location"] == "outside"
    assert ci.point_location_prepared_indexed_chunks(point, chunks) == expected


@pytest.mark.parametrize("chunks", [None, (), [], [object()], "not chunks"])
def test_invalid_chunk_snapshots_fail(chunks):
    with pytest.raises(HumanImportError, match="bounded prepared"):
        ci.point_location_prepared_indexed_chunks((F(0),)*3, chunks)


def test_chunk_count_is_bounded():
    prepared = ci.prepare_point_location(ci._records(V, FACES))
    with pytest.raises(HumanImportError, match="bounded prepared"):
        ci.point_location_prepared_indexed_chunks((F(0),)*3, [prepared]*65)

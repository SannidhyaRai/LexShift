"""M1 phase 2: gap encoding, variable-byte and Elias gamma codes, skip-pointer intersection. Hand-checked textbook values plus
random round trips; the index-level round trip runs on a small multi-zone index (the real one is checked by `python -m m1_index.compress`)."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("nltk")
try:
    from m1_index import compress as C
    from m1_index.index import IndexBuilder
    from m1_index.skips import MergeStats, intersect, skip_length
except (LookupError, RuntimeError, ImportError) as exc:
    pytest.skip(f"M1 cannot be imported here: {exc}", allow_module_level=True)

from common.schema import ZONES


# ------------------------------------------------------------------------------------------------ codes
def test_vb_textbook_examples():
    # IIR table 5.4: 824 -> 00000110 10111000, 5 -> 10000101, 214577 -> 00001101 00001100 10110001
    assert C.vb_encode([824]) == bytes([0b00000110, 0b10111000])
    assert C.vb_encode([5]) == bytes([0b10000101])
    assert C.vb_encode([214577]) == bytes([0b00001101, 0b00001100, 0b10110001])
    assert C.vb_decode(bytes([0b00000110, 0b10111000, 0b10000101])) == [824, 5]
    assert C.vb_encode([0]) == bytes([0x80])


def test_gamma_textbook_examples():
    # IIR table 5.5
    assert [C.gamma_code(n) for n in (1, 2, 3, 4, 9, 13, 24, 511, 1025)] == [
        "0", "100", "101", "11000", "1110001", "1110101", "111101000", "11111111011111111", "111111111100000000001"]
    for n in (1, 2, 3, 4, 9, 13, 24, 511, 1025, 2 ** 40):
        assert C.gamma_bits(n) == len(C.gamma_code(n))


def test_codes_reject_values_they_cannot_encode():
    with pytest.raises(ValueError):
        C.vb_encode([-1])
    with pytest.raises(ValueError):
        C.gamma_code(0)
    with pytest.raises(ValueError):
        C.vb_decode(bytes([0x01]))  # ends inside a number


def test_random_round_trips():
    rng = random.Random(7)
    for _ in range(300):
        xs = [rng.choice([1, 2, 3, rng.randint(1, 200), rng.randint(1, 10 ** 6), rng.randint(1, 2 ** 33)]) for _ in range(rng.randint(0, 60))]
        assert C.vb_decode(C.vb_encode(xs)) == xs
        data, nbits = C.gamma_encode(xs)
        assert nbits == sum(map(C.gamma_bits, xs)) and len(data) == (nbits + 7) // 8
        assert C.gamma_decode(data, nbits) == xs
        inc = sorted(set(xs))
        assert C.from_gaps(C.to_gaps(inc)) == inc


# ------------------------------------------------------------------------------------------------ the index as gap streams
def test_index_streams_round_trip_with_zones_restarting_positions():
    rng = random.Random(1)
    vocab = ["alpha", "beta", "gamma", "delta"]
    b = IndexBuilder()
    for d in range(40):
        zones = {}
        for z in rng.sample(list(ZONES), rng.randint(1, len(ZONES))):  # zones in a random order, some missing
            zones[z] = [rng.choice(vocab) for _ in range(rng.randint(0, 30))]
        b.add_tokens(f"d{d:02d}", {"year": 2020, "bench_size": 2}, zones)
    idx = b.finish()
    for t, term in enumerate(idx.terms):
        doc_gaps, pos_gaps = C.term_streams(idx, t)
        assert min(doc_gaps + pos_gaps) >= 1  # gamma-ready
        for dg, pg in ((C.vb_decode(C.vb_encode(doc_gaps)), C.vb_decode(C.vb_encode(pos_gaps))),
                       (C.gamma_decode(*C.gamma_encode(doc_gaps)), C.gamma_decode(*C.gamma_encode(pos_gaps)))):
            docs, positions = C.decode_term(idx, t, dg, pg)
            lo, hi = idx.t_start[t], idx.t_start[t + 1]
            assert docs == idx.p_doc[lo:hi].tolist(), term
            assert positions == idx.pos[idx.p_off[lo]:idx.p_off[hi]].tolist(), term


# ------------------------------------------------------------------------------------------------ skip pointers
def test_skip_length_is_sqrt():
    assert [skip_length(n) for n in (0, 1, 3, 4, 15, 16, 100, 4819)] == [0, 0, 0, 2, 3, 4, 10, 69]


def test_skip_intersection_equals_set_intersection():
    rng = random.Random(9)
    for _ in range(500):
        universe = rng.randint(1, 3000)
        a = sorted(rng.sample(range(universe), rng.randint(0, min(universe, 400))))
        b = sorted(rng.sample(range(universe), rng.randint(0, min(universe, 400))))
        want = sorted(set(a) & set(b))
        plain, skip = MergeStats(), MergeStats()
        assert intersect(a, 0, len(a), b, 0, len(b), False, plain) == want
        assert intersect(a, 0, len(a), b, 0, len(b), True, skip) == want
        assert plain.skips_followed == 0
        # slices of a larger array, as the index stores postings
        pad = [-5, -4] + a + [10 ** 9]
        assert intersect(pad, 2, 2 + len(a), b, 0, len(b), True) == want


def test_skips_save_comparisons_when_one_list_is_short_and_far_apart():
    long_list = list(range(0, 100000, 2))   # 50,000 entries, skip spacing 223
    short = [10, 50001, 99998]  # 50001 is odd: not in the long list
    plain, skip = MergeStats(), MergeStats()
    assert intersect(long_list, 0, len(long_list), short, 0, len(short), False, plain) == [10, 99998]
    assert intersect(long_list, 0, len(long_list), short, 0, len(short), True, skip) == [10, 99998]
    assert skip.skips_followed > 100
    assert skip.comparisons < plain.comparisons / 20

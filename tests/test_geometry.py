import numpy as np

from pathsnap.candidates import seg_dist


def one(px, py, ax, ay, bx, by):
    d, t = seg_dist(np.array([px]), np.array([py]), np.array([ax]), np.array([ay]), np.array([bx]), np.array([by]))
    return float(d[0]), float(t[0])


def test_point_above_middle_of_segment():
    d, t = one(0, 1, -1, 0, 1, 0)
    assert d == 1.0
    assert t == 0.5


def test_point_beyond_end_is_clamped_to_endpoint():
    d, t = one(3, 0, -1, 0, 1, 0)
    assert d == 2.0
    assert t == 1.0


def test_point_on_segment_has_zero_distance():
    d, _ = one(0.5, 0, 0, 0, 1, 0)
    assert d == 0.0


def test_zero_length_segment_does_not_crash():
    d, t = one(3, 4, 0, 0, 0, 0)
    assert d == 5.0
    assert t == 0.0

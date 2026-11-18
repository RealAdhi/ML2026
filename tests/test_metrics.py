import numpy as np

from pathsnap.decode_eval import seq_metrics

ARCS = np.array([[0, 1], [1, 2], [2, 3]])
ROUTE = np.array([0, 1, 2])


def test_perfect_prediction():
    obs = np.array([0, 1, 2])
    m = seq_metrics(np.array([0, 1, 2]), np.array([0, 1, 2]), obs, ARCS, ROUTE)
    assert m["acc"] == 1.0
    assert m["continuity"] == 1.0
    assert m["route_f1"] == 1.0


def test_wrong_middle_arc_breaks_accuracy_and_continuity():
    obs = np.array([0, 1, 2])
    m = seq_metrics(np.array([0, 2, 2]), np.array([0, 1, 2]), obs, ARCS, ROUTE)
    assert abs(m["acc"] - 2 / 3) < 1e-9
    assert m["continuity"] == 0.5


def test_staying_on_the_same_arc_is_continuous():
    obs = np.array([0, 1, 2, 3])
    m = seq_metrics(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), obs, ARCS, ROUTE)
    assert m["continuity"] == 1.0
    assert m["n_switch"] == 0

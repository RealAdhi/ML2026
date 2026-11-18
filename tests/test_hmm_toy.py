import numpy as np

from pathsnap.hmm import DecoderParams, Precomputed, viterbi


def toy():
    """Two GPS steps and two candidate arcs per step, with cross-connectivity only."""
    inf = np.inf
    rd = [np.array([[inf, 10.0], [10.0, inf]])]
    return Precomputed(obs=np.array([0, 1]), row_start=np.array([0, 2, 4]), rd=rd,
                       gc=np.array([10.0]), dt=np.array([1.0]), uturn=[np.zeros((2, 2), dtype=bool)],
                       arc=np.array([10, 11, 20, 21]), d=np.zeros(4))


def test_decoder_uses_road_connectivity_to_fix_a_weak_classifier_choice():
    pre = toy()
    p = np.array([0.90, 0.10, 0.55, 0.45])
    classifier_only = [10, 20]
    decoded = viterbi(pre, p, DecoderParams()).tolist()
    assert decoded != classifier_only
    assert decoded == [10, 21]


def test_confident_classifier_is_not_overruled_when_connectivity_agrees():
    pre = toy()
    p = np.array([0.05, 0.95, 0.95, 0.05])
    assert viterbi(pre, p, DecoderParams()).tolist() == [11, 20]

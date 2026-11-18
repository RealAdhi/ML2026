import os

import pandas as pd
import pytest

from pathsnap.candidates import FEATURES, MAX_CANDS, generate
from pathsnap.data import list_trips, load_trip

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "student_test")
pytestmark = pytest.mark.skipif(not os.path.isdir(DATA), reason="needs the student_test data folder")


@pytest.fixture(scope="module")
def first_trip():
    trip_id = list_trips(DATA)[0]
    return trip_id, load_trip(DATA, trip_id)


def test_trip_arrays_have_matching_lengths_and_increasing_timestamps(first_trip):
    _, trip = first_trip
    lengths = [len(getattr(trip, name)) for name in ("lon", "lat", "t", "px", "py")]
    assert len(set(lengths)) == 1
    assert (trip.t[1:] > trip.t[:-1]).all()


def test_route_can_be_withheld(first_trip):
    trip_id, _ = first_trip
    trip = load_trip(DATA, trip_id, with_route=False)
    assert trip.route is None


def test_arc_endpoints_are_valid_node_ids(first_trip):
    _, trip = first_trip
    assert trip.arcs.min() >= 0
    assert trip.arcs.max() < len(trip.node_x)


def test_unlabelled_generation_has_features_and_no_label(first_trip):
    _, trip = first_trip
    df, _ = generate(trip, with_labels=False)
    assert set(FEATURES) <= set(df.columns)
    assert "label" not in df.columns


def test_candidates_cover_every_observation_and_ranks_start_at_zero(first_trip):
    _, trip = first_trip
    df, _ = generate(trip, with_labels=False)
    assert df.obs.nunique() == len(trip.t)
    assert (df.groupby("obs").d_rank.min() == 0).all()


def test_candidate_distances_and_counts_are_bounded(first_trip):
    _, trip = first_trip
    df, _ = generate(trip, with_labels=False)
    assert (df.d >= 0).all()
    assert (df.groupby("obs").size() <= MAX_CANDS).all()


def test_candidate_generation_is_deterministic(first_trip):
    _, trip = first_trip
    first, _ = generate(trip, with_labels=False)
    second, _ = generate(trip, with_labels=False)
    pd.testing.assert_frame_equal(first, second)

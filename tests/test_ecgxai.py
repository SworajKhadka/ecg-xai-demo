"""Tests for the WFDB reader and the draft severity rules in ecgxai.py."""
import os

import pytest

import ecgxai as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "00001_lr")


def test_read_wfdb_shape():
    sig = E.read_wfdb(SAMPLE)
    assert sig.shape == (1000, 12)


def test_read_wfdb_first_value_in_mv():
    sig = E.read_wfdb(SAMPLE)
    assert sig[0, 0] == pytest.approx(-0.119, abs=1e-3)


@pytest.mark.parametrize(
    "codes, stadium, expected",
    [
        ({"NORM": 100}, None, 0),
        ({"CRBBB": 100}, None, 2),
        ({"INJAS": 100}, None, 3),
        ({"IMI": 100}, "Stadium III", 1),
    ],
)
def test_severity_grade(codes, stadium, expected):
    assert E.severity_grade(codes, stadium) == expected

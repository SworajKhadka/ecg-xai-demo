"""Shape tests for the three base models and an end-to-end ensemble check."""
import os

import numpy as np
import pytest
import torch

import ecgxai as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
SAMPLE = os.path.join(ROOT, "samples", "00001_lr")


@pytest.mark.parametrize("name", sorted(E.MODEL_ZOO))
def test_model_output_shapes(name):
    torch.manual_seed(0)
    model = E.MODEL_ZOO[name]().eval()
    with torch.no_grad():
        diag, sev = model(torch.randn(2, 12, E.WIN))
    assert diag.shape == (2, len(E.CLASSES)) == (2, 5)
    assert sev.shape == (2, 3)


def test_ensemble_probabilities_in_unit_interval():
    models, meta = E.load_models(MODEL_DIR)
    ens = E.Ensemble(models, meta)

    sig = E.fix_length(E.read_wfdb(SAMPLE))
    xn = E.normalise(E.bandpass(sig),
                     np.array(meta["norm_mean"], np.float32),
                     np.array(meta["norm_std"], np.float32))
    P, S = ens.base_probs(xn[None])
    prob, grades, grade_p, conf = ens.combine(P, S)

    assert prob.shape == (1, 5)
    for arr in (P, S, prob, grade_p, conf):
        assert np.all(np.isfinite(arr))
        assert np.all((arr >= 0) & (arr <= 1))
    assert grade_p.sum(-1) == pytest.approx(1.0, abs=1e-5)
    assert 0 <= int(grades[0]) <= 3

"""Tests for functions_cells_decode.py. Synthetic only; no data files needed."""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import functions_cells_decode as dec

GEOM = {1: (9, 6, 4), 2: (1, 6, 7), 3: (9, 6, 1)}   # I, A, B ports; I2 == B3, as in the data


def make_session(rng, ntr=160, nn=30):
    """Neurons that encode only the physical port (plus outcome); initiation is the port of I."""
    port = rng.normal(size=(10, nn)) * 5; outc = rng.normal(size=(2, nn)) * 5
    DM, fr = [], []
    for t, (I, A, B) in GEOM.items():
        for _ in range(ntr):
            ch, r = rng.integers(2), rng.integers(2)
            row = np.zeros(10); row[[1, 2, 5, 6, 7, 8]] = ch, r, t, A, B, I
            x = rng.normal(size=(nn, 63)) * .3
            x[:, 20:30] += port[I][:, None]
            x[:, 30:41] += port[A if ch else B][:, None]
            x[:, 41:55] += (port[A if ch else B] + outc[r])[:, None]
            DM.append(row); fr.append(x)
    return np.array(DM), np.array(fr)


@pytest.fixture(scope='module')
def sim():
    rng = np.random.default_rng(0)
    animals = [[make_session(rng) for _ in range(2)] for _ in range(3)]
    return ([[s[0] for s in a] for a in animals], [[s[1] for s in a] for a in animals])


def test_matrices_are_percent_rows(sim):
    xy, yx = dec.decoder_trial_states(sim, task_test=3)
    for m in (xy, yx):
        assert np.asarray(m).shape == (3, 7, 7)
        assert np.allclose(np.sum(m, 2), 100)


def test_port_code_lands_in_same_port_cells(sim):
    """If cells encode the port, misreads land where true and predicted share a port."""
    xy, yx = np.array(dec.decoder_trial_states(sim, task_test=3)).mean(1)
    assert xy[4, 0] > 95 and yx[0, 4] > 95          # B(T3)->I(T2) and I(T2)->B(T3)
    assert xy[0, 4] < 60 and yx[4, 0] < 60


def test_too_few_trials_excludes_session(sim):
    dm, fr = sim[0][0][0], sim[1][0][0]
    keep = ~((dm[:, 5] == 2) & (dm[:, 1] == 1) & (dm[:, 2] == 1)) | (np.cumsum((dm[:, 5] == 2) & (dm[:, 1] == 1) & (dm[:, 2] == 1)) <= 10)
    assert dec._session_patterns(fr[keep], dm[keep], 3) is None
    assert dec._session_patterns(fr, dm, 3) is not None


def test_animal_null_enumerates_35_splits():
    real, null = dec.animal_permutation_null(np.array([1., 2, 3]), np.array([10., 11, 12, 13]), 1)
    assert len(null) == 35 and np.isclose(real, null.max()) and np.isclose(real, null[0])


def test_port_null_uses_same_cells_as_real():
    rng = np.random.default_rng(1)
    m = [rng.random((3, n, 7, 7)) for n in (3, 3, 4, 4)]
    for same_port in (True, False):
        st = dec.decoding_statistics(*m, same_port=same_port)['Port']
        assert np.isclose(st['null'][0], st['diff'])


def test_port_cells_orientation_and_default():
    hp1, hp2 = np.zeros((2, 3, 3, 7, 7))
    hp1[2, :, 4, 0] = 80; hp2[2, :, 0, 4] = 60           # same-port cells only
    assert np.allclose(dec.decoding_scores(hp1, hp2)['Port'], 70)
    assert np.allclose(dec.decoding_scores(hp1, hp2, same_port=False)['Port'], 0)

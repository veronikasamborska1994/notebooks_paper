"""Synthetic tests for functions_cells_port.py."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import functions_cells_port as fp

PORTS = {1: (1, 6, 4), 2: (9, 6, 2), 3: (1, 6, 9)}          # I, A, B; I2 == B3, I3 == I1


def sessions(kind, n_sessions=3, seed=0, nn=15):
    """Neurons that code only the physical port or only the role of the current state."""
    rng = np.random.default_rng(seed); dms, frs = [], []
    for _ in range(n_sessions):
        code = rng.normal(size=(10, nn)) * 5; dm, fr = [], []
        for layout, (i, a, b) in PORTS.items():
            for t in range(80):
                choice = t % 2; d = np.zeros(10); d[[1, 2, 5, 6, 7, 8]] = choice, (t // 2) % 2, layout, a, b, i
                key = (i, a if choice else b) if kind == 'port' else (0, 1 if choice else 2)
                x = rng.normal(size=(nn, 63)) * .5
                x[:, :30] += code[key[0]][:, None]; x[:, 30:] += code[key[1]][:, None]
                dm.append(d); fr.append(x)
        dms.append(np.array(dm)); frs.append(np.array(fr))
    return dms, frs


def test_port_only_neurons_decode_as_the_same_port():
    same_port, same_role, _ = fp.animal_scores(*sessions('port'), [34, 35], 15, draws=3)[0]
    assert same_port > 95 and same_role < 5


def test_role_only_neurons_decode_as_the_same_role():
    same_port, same_role, control = fp.animal_scores(*sessions('role'), [34, 35], 15, draws=3)[0]
    assert same_role > 95 and same_port < 5 and control < 5


def test_noise_is_at_chance_and_result_is_reproducible():
    rng = np.random.default_rng(1); dms, frs = sessions('port')
    frs = [rng.normal(size=f.shape) for f in frs]
    s = fp.animal_scores(dms, frs, [34, 35], 15, draws=20)[0]
    assert np.all(np.abs(s - 100 / 3) < 15)
    assert np.array_equal(s, fp.animal_scores(dms, frs, [34, 35], 15, draws=20)[0])


def test_rewarded_only_uses_rewarded_choices():
    dms, frs = sessions('port')
    assert len(fp.session_states(frs[0], dms[0], [34, 35], True)['B3']) == 20
    assert len(fp.session_states(frs[0], dms[0], [34, 35], False)['B3']) == 40


def test_region_difference_p():
    p, pmax = fp.region_difference_p(np.array([[90., 91, 92, 1, 2, 3, 4], [5, 4, 3, 5, 4, 3, 4]]))
    assert np.isclose(p[0], 1 / 35) and p[1] > 0.2 and np.isclose(pmax, 1 / 35)

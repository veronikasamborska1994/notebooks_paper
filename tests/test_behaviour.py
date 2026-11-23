"""Synthetic tests for the Fig 1e and Fig 1f reversal analyses."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import functions_behaviour as fb


class Session:
    def __init__(self, **td):
        self.trial_data = td


class Experiment:
    """Every subject has the same list of sessions."""
    def __init__(self, sessions, n_subjects=2):
        self.sessions = sessions; self.subject_IDs = list(range(n_subjects))

    def get_sessions(self, subject_IDs=None):
        return self.sessions


def session_1e(n, crossings=(), reversals=(), n_forced=0, config=1):
    """Crossing c: countdown starts on trial c + 1. Reversal r: block changes on trial r + 1."""
    block = np.zeros(n, int); prt = np.zeros(n, int)
    for r in reversals:
        block[r + 1:] += 1
    for k, c in enumerate(crossings):
        prt[c + 1:(reversals[k] + 1 if k < len(reversals) else n)] = 5
    forced = np.r_[np.ones(n_forced, int), np.zeros(n, int)]
    pad = np.zeros(n_forced, int)
    return Session(**{'block': np.r_[pad, block], 'pre-reversal trials': np.r_[pad, prt],
                      'forced_trial': forced, 'n_trials': n + n_forced,
                      'configuration_i': np.full(n + n_forced, config)})


class Captured(Exception):
    pass


def trials_to_threshold(sessions):
    """First problem's trials-to-threshold as handed to the ANOVA."""
    original, seen = fb.pd.DataFrame.from_dict, {}

    def grab(data):
        seen['x'] = np.asarray(data['Data'])
        raise Captured
    fb.pd.DataFrame.from_dict = staticmethod(grab)
    try:
        fb.trials_till_reversal_plot(Experiment(sessions))
    except Captured:
        pass
    finally:
        fb.pd.DataFrame.from_dict = original
    return seen['x'][:10]


def test_carry_over_uses_last_crossing():
    # crossings at 10, 40 and 80; the session ends before the last reversal
    x = trials_to_threshold([session_1e(100, (10, 40, 80), (20, 50)), session_1e(60, (30,), (40,))])
    assert list(x[:4]) == [10, 20, 30, 30 + (100 - 80)]


def test_forced_trials_subtracted_once():
    x = trials_to_threshold([session_1e(50, n_forced=7), session_1e(60, (30,), (40,))])
    assert x[0] == 30 + 50


def test_fig1f_window_starts_on_first_new_state_trial():
    # perfect switching: choice == state on every trial, so P(new best) is exactly 0 before, 1 after
    n, revs = 120, (29, 59, 89)
    block = np.zeros(n, int)
    for r in revs:
        block[r + 1:] += 1
    state = block % 2
    s = Session(block=block, forced_trial=np.zeros(n, int), state=state, choices=state.copy())
    fb.plt.close('all')
    fb.meta_learning_reversals(Experiment([s]))
    for line in fb.plt.gca().get_lines()[:2]:
        assert np.allclose(line.get_ydata(), np.r_[np.zeros(10), np.ones(10)])
    fb.plt.close('all')

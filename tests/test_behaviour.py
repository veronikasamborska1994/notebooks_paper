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


class Ev:
    def __init__(self, name):
        self.name = name


def poke_session(n, wrong_trial, rev):
    """One out-of-sequence poke on trial `wrong_trial`; the block changes on trial rev + 1."""
    events = []
    for t in range(n):
        events += ['init_trial', 'choice_state', 'poke_1'] + (['poke_2'] if t == wrong_trial else [])
    block = np.zeros(n, int); block[rev + 1:] = 1
    s = Session(block=block, forced_trial=np.zeros(n, int), n_trials=n,
                configuration_i=np.full(n, 5), poke_A=np.full(n, 1), poke_B=np.full(n, 2))
    s.events = [Ev(e) for e in events]
    return s


def bad_pokes_first_problem(session):
    """Mean out-of-sequence pokes per reversal of the first problem (Fig 1g data)."""
    original, seen = fb.pd.DataFrame.from_dict, {}

    def grab(data):
        seen['x'] = np.asarray(data['Data'])
        raise Captured
    fb.pd.DataFrame.from_dict = staticmethod(grab)
    try:
        with np.errstate(all='ignore'):
            fb.out_of_sequence(Experiment([session]))
    except Captured:
        pass
    finally:
        fb.pd.DataFrame.from_dict = original
    return seen['x'][:2]


def test_poke_on_last_trial_of_old_block():
    # block 1 = trials 0-9 (10 trials), so one poke on trial 9 gives 0.1 and 0
    assert np.allclose(bad_pokes_first_problem(poke_session(20, 9, 9)), [0.1, 0])


def test_poke_on_first_trial_of_new_block():
    assert np.allclose(bad_pokes_first_problem(poke_session(20, 10, 9)), [0, 0.1])


def test_poke_on_last_trial_of_session_is_counted():
    assert np.allclose(bad_pokes_first_problem(poke_session(20, 19, 9)), [0, 0.1])


class RecCaptured(Exception):
    pass


def rec_pokes(wrong_trial):
    """Reversal means of the first problem (4 blocks of 5 trials per problem, 3 problems)."""
    from functions import functions_behaviour_rec as fr
    n = 70; tl = np.arange(n)
    block = tl // 5; task = np.minimum(tl // 20, 2) + 1
    events = []
    for t in range(n):
        a = {1: 'poke_1', 2: 'poke_3', 3: 'poke_5'}[task[t]]; b = {1: 'poke_2', 2: 'poke_4', 3: 'poke_6'}[task[t]]
        events += ['init_trial', 'choice_state', a] + ([b] if t == wrong_trial else [])
    s = Session(block=block, task=task, configuration_i=np.full(n, 9),
                poke_A=np.array([1, 3, 5])[task - 1], poke_B=np.array([2, 4, 6])[task - 1])
    s.events = [Ev(e) for e in events]
    seen, original = {}, fr.pd.DataFrame.from_dict

    def grab(data):
        seen['x'] = np.asarray(data['Data'])
        raise RecCaptured
    fr.pd.DataFrame.from_dict = staticmethod(grab)

    class Subject:
        def get_sessions(self):
            return [s] * 8                        # 8 sessions x 3 problems = the 24 expected
    try:
        fr.out_of_sequence_recordings([Subject(), Subject()])
    except RecCaptured:
        pass
    finally:
        fr.pd.DataFrame.from_dict = original
    return seen['x'][:4]


def test_recordings_poke_at_block_boundary():
    assert np.allclose(rec_pokes(4), [0.2, 0, 0, 0])      # last trial of block 1
    assert np.allclose(rec_pokes(5), [0, 0.2, 0, 0])      # first trial of block 2

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


def test_counts_restart_after_crossing_without_reversal():
    # crossings at 10, 40 and 80; the session ends before the last reversal, so the next count starts at 0
    x = trials_to_threshold([session_1e(100, (10, 40, 80), (20, 50)), session_1e(60, (30,), (40,))])
    assert list(x[:4]) == [11, 20, 30, 31]


def test_forced_trials_excluded_once():
    x = trials_to_threshold([session_1e(50, n_forced=7), session_1e(60, (30,), (40,))])
    assert x[0] == 50 + 31


def test_several_sessions_without_crossing_accumulate():
    x = trials_to_threshold([session_1e(50, n_forced=3), session_1e(20), session_1e(30, n_forced=2), session_1e(60, (30,), (40,))])
    assert x[0] == 50 + 20 + 30 + 31


def test_trials_after_last_reversal_carry_over():
    # 60 trials, reversal after trial 40: 19 trials of the new block in this session
    x = trials_to_threshold([session_1e(60, (30,), (40,)), session_1e(25), session_1e(60, (10,), (20,))])
    assert list(x[:2]) == [31, 19 + 25 + 11]


def test_counter_resets_on_new_problem():
    original, seen = fb.pd.DataFrame.from_dict, {}
    s = [session_1e(60, (30,), (40,)), session_1e(60, (10,), (20,), config=2)]

    def grab(data):
        seen['x'] = np.asarray(data['Data'])
        raise Captured
    fb.pd.DataFrame.from_dict = staticmethod(grab)
    try:
        fb.trials_till_reversal_plot(Experiment(s))
    except Captured:
        pass
    finally:
        fb.pd.DataFrame.from_dict = original
    assert seen['x'][0] == 31 and seen['x'][10] == 11


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


def rec_pokes(wrong_trial, pokes=('A', 'B'), only_A_B=True):
    """Reversal means of the first problem (4 blocks of 5 trials per problem, 3 problems).

    On `wrong_trial` the animal makes `pokes` once the choice state has started ('other' is a port of
    another problem); on every other trial it chooses A.
    """
    from functions import functions_behaviour_rec as fr
    n = 70; tl = np.arange(n)
    block = tl // 5; task = np.minimum(tl // 20, 2) + 1
    events = []
    for t in range(n):
        port = {'A': {1: 'poke_1', 2: 'poke_3', 3: 'poke_5'}[task[t]], 'B': {1: 'poke_2', 2: 'poke_4', 3: 'poke_6'}[task[t]],
                'other': 'poke_3' if task[t] == 1 else 'poke_1'}
        events += ['init_trial', 'choice_state'] + [port[k] for k in (pokes if t == wrong_trial else ('A',))]
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
        fr.out_of_sequence_recordings([Subject(), Subject()], only_A_B=only_A_B)
    except RecCaptured:
        pass
    finally:
        fr.pd.DataFrame.from_dict = original
    return seen['x'][:4]


def test_recordings_poke_at_block_boundary():
    assert np.allclose(rec_pokes(4), [0.2, 0, 0, 0])      # last trial of block 1
    assert np.allclose(rec_pokes(5), [0, 0.2, 0, 0])      # first trial of block 2


def test_recordings_trials_to_threshold_counts_first_block_fully():
    from functions import functions_behaviour_rec as fr
    n = 13 * 10; tl = np.arange(n)
    block = tl // 10                                        # reversal after every 10 trials
    prt = ((tl % 10) >= 7).astype(int) * 5                  # countdown starts on the 8th trial of each block
    s = Session(**{'block': block, 'pre-reversal trials': prt})
    s.events = [Ev(e) for _ in range(n) for e in ['init_trial']]
    seen, original = {}, fr.pd.DataFrame.from_dict

    def grab(data):
        seen['x'] = np.asarray(data['Data'])
        raise RecCaptured
    fr.pd.DataFrame.from_dict = staticmethod(grab)

    class Subject:
        def get_sessions(self):
            return [s] * 8
    try:
        fr.trials_till_reversal_plot_recordings([Subject(), Subject()])
    except RecCaptured:
        pass
    finally:
        fr.pd.DataFrame.from_dict = original
    assert np.allclose(seen['x'][:12], 7)                   # every block takes 7 trials, the first included


def test_rewarded_streak_resets_on_first_error():
    from functions.functions_behaviour_rec import rewarded_streak
    assert list(rewarded_streak([1, 1, 1, 0, 1, 1, 0, 0, 1])) == [0, 1, 2, 3, 0, 1, 2, 0, 0]


def test_fig1f_uses_sessions_with_a_single_reversal():
    # each session has one reversal; both directions are needed (A to B and B to A)
    def one_reversal(first_state):
        block = np.r_[np.zeros(30, int), np.ones(30, int)]
        state = (first_state + block) % 2
        return Session(block=block, forced_trial=np.zeros(60, int), state=state, choices=state.copy())
    fb.plt.close('all')
    fb.meta_learning_reversals(Experiment([one_reversal(0), one_reversal(1)]))
    for line in fb.plt.gca().get_lines()[:2]:
        assert np.allclose(line.get_ydata(), np.r_[np.zeros(10), np.ones(10)])
    fb.plt.close('all')

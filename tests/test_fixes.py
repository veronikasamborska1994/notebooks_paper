"""One test per fix that the other test files do not cover. Numbers refer to CORRECTIONS.md."""
import os, sys
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import functions_behaviour as fb, functions_behaviour_rec as fr
from functions import functions_cells_regress as regs, functions_cells_rsa as rsa, functions_cells_policy as pol
from functions import functions_cells_svd as svd, functions_cells_ext as ext, functions_cells_decode as dec
from test_neural_synthetic import data, region, N_HP, N_PFC
from test_behaviour import Session, Ev, rec_pokes

N_PERM = 5


@pytest.fixture(scope='module')
def permutation_figures():
    """Run the time-resolved permutation tests once; return the nulls handed to np.percentile and the legend labels."""
    def blocks(rng, DM, fr):
        DM[:, 4] = np.arange(len(DM)) // 15                  # block number, used to roll the firing rates
    hp, pfc = data(hp=blocks, pfc=blocks); shapes, original = [], np.percentile

    def record(a, *args, **kw):
        shapes.append(np.shape(a))
        return original(a, *args, **kw)
    np.percentile = record
    try:
        plt.close('all')
        regs.plot_simple_GLM(hp, pfc, n_perm=N_PERM)
        rsa.RSA_across_time_plot(hp, pfc, perm_n=N_PERM)
        pol.plot_policy_simple_GLM(hp, pfc, n_perm=N_PERM)
        pol.plot_correlations_slice(pfc, hp, perm_n=N_PERM)
    finally:
        np.percentile = original
    labels = [line.get_label() for n in plt.get_fignums() for ax in plt.figure(n).axes for line in ax.lines]
    plt.close('all')
    return shapes, labels


def test_1_maximum_over_time_is_taken_before_the_percentile(permutation_figures):
    shapes, _ = permutation_figures
    assert len(shapes) >= 4
    for shape in shapes:                                   # one maximum per permutation: no time axis is left
        assert shape[0] == N_PERM and 63 not in shape and 28 not in shape


def test_2_large_markers_are_labelled_p_01(permutation_figures):
    _, labels = permutation_figures
    assert any('< .01' in s for s in labels) and not any('.001' in s for s in labels)


def test_5_decoding_returns_both_directions_for_pfc():
    rng = np.random.default_rng(0)
    hp, pfc = region(rng, N_HP, ntr=180), region(rng, N_PFC, ntr=180)
    out = dec.permute_subjects(hp, pfc)
    xy, yx = dec.decoder_trial_states(pfc, task_test=3, H=1, perm=False)
    assert np.allclose(out[4][2], xy) and np.allclose(out[5][2], yx) and not np.allclose(xy, yx)


def test_6_cross_region_projection_uses_the_modes_of_the_same_problem():
    # with the same data as both regions, projecting onto the other region's temporal modes changes nothing
    hp, _ = data()
    plt.close('all'); svd.svd_between_brains(hp, hp)
    own, cross = [line.get_ydata() for line in plt.gca().lines[:2]]
    plt.close('all')
    assert np.allclose(own, cross)


def training_session(seed, config, n=80):
    rng = np.random.default_rng(seed)
    return Session(choices=rng.integers(0, 2, n).astype(float), outcomes=rng.integers(0, 2, n).astype(float),
                   configuration_i=np.full(n, config))


class Subjects:
    """A different list of sessions for every subject."""
    def __init__(self, per_subject):
        self.per_subject = per_subject; self.subject_IDs = list(range(len(per_subject)))

    def get_sessions(self, subject_IDs=None):
        return self.per_subject[subject_IDs[0]]


def test_7_policy_weights_of_a_problem_use_only_its_own_sessions():
    sessions = [training_session(c, c) for c in range(10)]                      # one session per problem
    in_order = fb.policy_training(Subjects([sessions]))[0]
    second_first = fb.policy_training(Subjects([[sessions[1], sessions[0]] + sessions[2:]]))[0]
    plt.close('all')
    assert np.allclose(in_order[1], second_first[0])                            # the second problem, fitted on its own session


def test_8_anova_rows_carry_their_own_subject_and_problem():
    exp = Subjects([[training_session(100 * s + c, c) for c in range(10)] for s in range(9)])
    coef = np.asarray(fb.policy_training(exp))             # subject x problem x weight
    seen, original = [], fb.pd.DataFrame.from_dict
    fb.pd.DataFrame.from_dict = staticmethod(lambda data: (seen.append(data), original(data))[1])
    try:
        fb.plot_policy_during_training(exp)
    finally:
        fb.pd.DataFrame.from_dict = original; plt.close('all')
    d = seen[0]                                            # weight of the previous choice (lag 1)
    assert np.allclose(d['Data'], coef[np.asarray(d['Sub_id']), np.asarray(d['cond']).astype(int), 11])


def test_9_error_band_of_the_choice_history_weights_is_symmetric():
    hp, pfc = data(); bands, original = [], plt.fill_between
    plt.fill_between = lambda x, upper, lower, **k: bands.append((upper, lower))
    try:
        plt.close('all'); pol.plot_policy_during_recordings(pfc, hp, n=5)
    finally:
        plt.fill_between = original
    line = plt.gcf().axes[0].lines[0].get_ydata(); upper, lower = bands[0]
    plt.close('all')
    assert np.allclose(upper - line, line - lower)


def test_10_port_correction_is_applied_to_every_layout():
    # A is the port used in all three layouts (port 1). The task file calls it B in layouts 2 and 3, so both need relabelling.
    # Layout 3: the light is on the task file's A (port 4) while port 1 is good, and the animal pokes port 1 first.
    n = 78; t = np.arange(n); task = np.minimum(t // 26, 2) + 1
    state = np.where(task == 3, 0, 1)
    s = Session(choices=state, state=state, block=t // 6, task=task, outcomes=np.ones(n), configuration_i=np.full(n, 9),
                poke_A=np.array([1, 3, 4])[task - 1], poke_B=np.array([2, 1, 1])[task - 1])
    s.events = [Ev(e) for k in range(n) for e in ('init_trial', 'a_forced_state', 'poke_1')]

    class Subject:
        def get_sessions(self):
            return [s]
    a_when_b_forced, _, a_when_a_forced, *_ = fr.forced_trials_([Subject()])
    assert a_when_b_forced[0] == 1 and a_when_a_forced[0] == 0


def test_11_other_ports_count_only_after_the_choice():
    assert np.allclose(rec_pokes(2, ('other', 'A'), only_A_B=False), 0)              # poked before choosing: in sequence
    assert np.allclose(rec_pokes(2, ('A', 'other'), only_A_B=False), [0.2, 0, 0, 0])   # poked after choosing A


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_12_unrewarded_trials_exclude_rewarded_ones():
    def rewarded_b_in_layout_2(rng, DM, fr):
        fr[:] = 0
        fr[(DM[:, 5] == 2) & (DM[:, 1] == 0) & (DM[:, 2] == 1)] = 1
    hp, _ = data(hp=rewarded_b_in_layout_2)
    plt.close('all'); ext.heatplot_sort_init_B(hp, ch=1)   # A choices: the planted activity must not appear
    layout_2 = np.asarray(plt.gcf().axes[0].images[0].get_array())
    plt.close('all')
    assert np.all(layout_2 == 0)


def test_15_a_poke_after_b_choice_is_counted_in_the_first_problem():
    assert np.allclose(rec_pokes(2, ('B', 'A')), [0.2, 0, 0, 0])


def test_18_error_bars_are_computed_for_each_problem():
    # 10 problems x 10 blocks of 3 trials; the second subject makes one out-of-sequence poke per trial in problem 1
    def session(config, wrong):
        n = 30
        s = Session(n_trials=n, block=np.arange(n) // 3, forced_trial=np.zeros(n, int), configuration_i=np.full(n, config),
                    poke_A=np.full(n, 1), poke_B=np.full(n, 2))
        s.events = [Ev(e) for _ in range(n) for e in ['init_trial', 'choice_state', 'poke_1'] + (['poke_2'] if wrong else [])]
        return s
    exp = Subjects([[session(c, False) for c in range(1, 11)], [session(c, c == 1) for c in range(1, 11)]])
    exp.get_sessions = lambda subject_ID: exp.per_subject[subject_ID]

    class Anova:
        def __init__(self, *a, **k):
            self.anova_table = pd.DataFrame({k: [0.0, 0.0] for k in ('Num DF', 'Den DF', 'F Value', 'Pr > F')})

        def fit(self):
            return self

    class Captured(Exception):
        pass
    seen = {}

    def grab(**k):
        seen.update(k)
        raise Captured
    originals = fb.AnovaRM, plt.errorbar
    fb.AnovaRM, plt.errorbar = Anova, grab
    try:
        fb.out_of_sequence(exp)
    except Captured:
        pass
    finally:
        fb.AnovaRM, plt.errorbar = originals; plt.close('all')
    assert np.allclose(seen['y'], [0.5] + [0] * 9)
    assert np.allclose(seen['yerr'], [0.5 / np.sqrt(2)] + [0] * 9)

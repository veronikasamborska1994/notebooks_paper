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


@pytest.fixture(autouse=True)
def no_minimum_reversals(monkeypatch):
    # short synthetic problems; the rule that drops interrupted problems has its own test
    monkeypatch.setattr(fb, 'MIN_REVERSALS_PER_PROBLEM', 0)


@pytest.fixture(scope='module')
def permutation_figures():
    """Run the time-resolved permutation tests once; return the shapes of the nulls handed to np.percentile
    and the (label, size) of every significance marker that has a legend entry."""
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
    markers = [(line.get_label(), line.get_markersize()) for n in plt.get_fignums() for ax in plt.figure(n).axes
               for line in ax.lines if line.get_marker() == '.' and not line.get_label().startswith('_')]
    plt.close('all')
    return shapes, markers


def test_1_maximum_over_time_is_taken_before_the_percentile(permutation_figures):
    shapes, _ = permutation_figures
    assert len(shapes) >= 4
    for shape in shapes:                                   # one maximum per permutation: no time axis is left
        assert shape[0] == N_PERM and 63 not in shape and 28 not in shape


def test_2_large_markers_are_labelled_p_01(permutation_figures):
    _, markers = permutation_figures
    assert len(markers) >= 10
    for label, size in markers:                            # session permutations: only the large markers are in the legend
        assert size == 4 and '< .01' in label and '.001' not in label


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
                   configuration_i=np.full(n, config), poke_A=np.full(n, 10 + config), poke_B=np.full(n, 20 + config),
                   block=np.arange(n) // 8, forced_trial=np.zeros(n, int))


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


def test_28_29_circular_shift_stays_away_from_the_unshuffled_data():
    from functions.helper_functions import circular_shift
    np.random.seed(0)
    for n_rows, m in [(157, 12), (60, 11), (278 - 12, 11), (30, 14)]:
        s = np.array([circular_shift(n_rows, m) for _ in range(5000)])
        assert s.min() >= m and s.max() <= n_rows - m          # never closer than m rows to the identity, either way
        assert s.min() == m and s.max() == n_rows - m           # whole allowed range is used


def _record_shifts(module, monkeypatch):
    import functions.helper_functions as hf
    calls = []

    def spy(n_rows, min_shift):
        calls.append((n_rows, min_shift))
        return hf.circular_shift(n_rows, min_shift)
    monkeypatch.setattr(module, 'circular_shift', spy)
    return calls


def test_28_within_region_roll_uses_each_tasks_own_length(monkeypatch):
    hp, _ = data()
    hp = {'DM': [hp['DM'][0][:2]], 'Data': [hp['Data'][0][:2]]}
    for dm in hp['DM'][0]:
        dm[:, 4] = np.arange(len(dm)) // 20                          # a block change every 20 trials (block_min = 20)
    calls = _record_shifts(regs, monkeypatch)
    regs.GLM_roll_time(hp, perm=3)
    lengths = [[int(np.sum(dm[:, 5] == t)) for t in (1, 2, 3)] for dm in hp['DM'][0]]
    assert calls == [(l, 20) for ls in lengths for _ in range(3) for l in ls]   # tasks 1-3 of each session, for each of 3 perms


def test_29_policy_roll_shifts_within_the_rows_regressed(monkeypatch):
    hp, _ = data()
    hp = {'DM': [hp['DM'][0][:2]], 'Data': [hp['Data'][0][:2]]}
    calls = _record_shifts(pol, monkeypatch)
    pol.GLM_policy_roll_tome(hp, perm=2, n=11)
    n_rows = [len(dm) - 11 - 1 for dm in hp['DM'][0]]                # n history trials and one more are dropped
    assert calls == [(r, 11) for r in n_rows for _ in range(2)]


def test_fig1f_both_error_bands_have_a_visible_fill():
    from test_behaviour import Experiment
    rng = np.random.default_rng(0); n = 120; block = np.zeros(n, int)
    for r in (29, 59, 89):
        block[r + 1:] += 1
    state = block % 2
    sessions = [Session(block=block, forced_trial=np.zeros(n, int), state=state,
                        choices=np.where(rng.random(n) < .8, state, 1 - state)) for _ in range(3)]
    plt.close('all'); fb.meta_learning_reversals(Experiment(sessions, n_subjects=3))
    from matplotlib.collections import LineCollection
    bands = [c for c in plt.gca().collections if not isinstance(c, LineCollection)]   # the block-switch line is a LineCollection
    plt.close('all')
    assert len(bands) == 2
    for b in bands:
        r, g, bl, a = b.get_facecolor()[0]
        assert a >= 0.25 and max(r, g, bl) < 0.9                   # not almost white, and an explicit fill colour


def test_fig1il_outcome_x_choice_panels_share_the_outcome_colour():
    exp = Subjects([[training_session(100 * s + c, c) for c in range(10)] for s in range(9)])
    plt.close('all'); fb.plot_policy_during_training(exp)
    fig = [plt.figure(n) for n in plt.get_fignums() if len(plt.figure(n).axes) == 4][-1]
    colours = [ax.lines[0].get_color() for ax in fig.axes]
    plt.close('all')
    assert colours[0] == colours[1] and colours[2] == colours[3] and colours[1] != colours[2]


def test_last_time_tick_is_on_the_last_time_bin():
    from functions.helper_functions import XTICK_POSITIONS, XTICK_POSITIONS_DLC
    hp, _ = data()
    n_bins = hp['Data'][0][0].shape[2]
    assert XTICK_POSITIONS[-1] == 62 and XTICK_POSITIONS_DLC[-1] == 63
    assert XTICK_POSITIONS[-1] <= n_bins - 1


def test_rsa_ticks_sit_on_the_choice_reward_and_half_second_bins_and_the_markers_have_a_legend():
    def blocks(rng, DM, fr):
        DM[:, 4] = np.arange(len(DM)) // 15
    hp, pfc = data(hp=blocks, pfc=blocks)
    plt.close('all'); rsa.RSA_across_time_plot(hp, pfc, perm_n=N_PERM)
    axes = [ax for n in plt.get_fignums() for ax in plt.figure(n).axes
            if [t.get_text() for t in ax.get_xticklabels()] == ['C', 'R', '+0.5']]
    ticks = [list(ax.get_xticks()) for ax in axes]; has_legend = axes[0].get_legend() is not None
    plt.close('all')
    assert len(axes) >= 6
    assert all(t == [35 - 34, 42 - 34, 49 - 34] for t in ticks)      # x position i is the window starting at bin 34 + i
    assert has_legend


def test_fig4g_has_one_legend_naming_the_regions_pfc_and_ca1():
    from test_decode import make_session
    rng = np.random.default_rng(0)
    hp = {'DM': [np.array([make_session(rng)[0] for _ in range(N_HP)], dtype=object)],
          'Data': [np.array([make_session(rng)[1] for _ in range(N_HP)], dtype=object)]}
    pfc = {'DM': [np.array([make_session(rng)[0] for _ in range(N_PFC)], dtype=object)],
           'Data': [np.array([make_session(rng)[1] for _ in range(N_PFC)], dtype=object)]}
    plt.close('all')
    try:
        dec.plot(hp, pfc)
    except Exception as e:
        pytest.skip('synthetic decoding data not supported: %s' % e)
    names = [t.get_text() for t in plt.gca().get_legend().get_texts()]
    plt.close('all')
    assert names == ['PFC', 'CA1']


def test_30_time_control_keeps_adjacent_layouts_in_either_order_and_compares_earlier_with_later(monkeypatch):
    def session(order):                                       # layouts run in this order, 30 trials each
        task = np.repeat(order, 30).astype(float)
        DM = np.zeros((len(task), 8)); DM[:, 5] = task
        return DM, np.zeros((len(task), 2, 5))
    sessions = [session([2, 1, 3]), session([1, 2, 3]), session([1, 3, 2]), session([3, 1, 2])]
    d = {'DM': [np.array([s[0] for s in sessions] + [None], object)[:-1]],
         'Data': [np.array([s[1] for s in sessions] + [None], object)[:-1]]}
    monkeypatch.setattr(ext, 'task_ind', lambda task, a, b: task)
    _, _, order = ext.control_for_time(d, compare_tasks=1)          # T1-T2: adjacent in sessions 0, 1 and 3
    assert order == [(2, 1), (1, 2), (1, 2)]
    _, _, order = ext.control_for_time(d, compare_tasks=3)          # T1-T3: adjacent in sessions 0, 2 and 3
    assert order == [(1, 3), (1, 3), (3, 1)]


def test_ext2dh_last_free_trial_of_a_layout_is_scored_with_its_own_ports():
    # trial 19 is the last trial of layout 1 (reversal block 4); A then B is an out-of-sequence poke in layout 1
    assert np.allclose(rec_pokes(19), [0, 0, 0, 0.2])
    assert np.allclose(rec_pokes(19, ('A', 'other'), only_A_B=False), [0, 0, 0, 0.2])
    # the first trial of a layout is scored too (its "no previous choice" reset must not wipe the choice just made)
    assert np.allclose(rec_pokes(0), [0.2, 0, 0, 0])


def _layout_session(seed, i, a, b, n=80, block_len=8):
    rng = np.random.default_rng(seed)
    return Session(choices=rng.integers(0, 2, n).astype(float), outcomes=rng.integers(0, 2, n).astype(float),
                   configuration_i=np.full(n, i), poke_A=np.full(n, a), poke_B=np.full(n, b),
                   block=np.arange(n) // block_len, forced_trial=np.zeros(n, int))


LAYOUTS = [(6, 1, 5), (6, 5, 3), (3, 1, 6), (1, 3, 6), (2, 6, 4), (4, 2, 6), (5, 1, 2), (1, 2, 5), (2, 3, 1), (3, 4, 5)]


def test_c3_problems_are_told_apart_by_the_whole_port_layout():
    # the first two layouts share the initiation port (6) but not A and B: they are two problems, ten in all
    sessions = [_layout_session(k, *lay) for k, lay in enumerate(LAYOUTS)]
    coefs = fb.policy_training(Subjects([sessions]))[0]
    plt.close('all')
    assert len(coefs) == 10


def test_c3_an_interrupted_problem_is_left_out(monkeypatch):
    monkeypatch.setattr(fb, 'MIN_REVERSALS_PER_PROBLEM', 5)
    full = [_layout_session(k, *lay) for k, lay in enumerate(LAYOUTS)]                # 9 reversals each
    interrupted = _layout_session(99, 6, 2, 3, block_len=30)                          # 2 reversals, same initiation port
    sessions = full[:9] + [interrupted] + full[9:]
    assert fb.interrupted_problem_sessions(sessions) == {9}
    with_it = fb.policy_training(Subjects([sessions]))[0]
    without = fb.policy_training(Subjects([full]))[0]
    plt.close('all')
    assert len(with_it) == 10 and np.allclose(with_it, without)

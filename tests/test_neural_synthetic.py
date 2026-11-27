"""Known-answer tests for the RSA, SVD and policy analyses on synthetic sessions."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import functions_cells_rsa as rsa
from functions import functions_cells_svd as svd
from functions import functions_cells_policy as pol
from functions import RSAs
import matplotlib.pyplot as plt

N_HP, N_PFC = 15 + 7 + 2, 9 + 16 + 13 + 2          # sessions per region (last animal is the remainder)
PORTS = {1: (9, 6, 4), 2: (1, 6, 7), 3: (9, 6, 1)}  # I, A, B; I2 == B3


def session(rng, ntr=120, nn=4, nt=63, signal=None):
    DM = np.zeros((ntr, 10)); DM[:, 5] = np.repeat([1, 2, 3], ntr // 3)
    half = ntr // 6                                  # balanced choice x reward in each half of each task
    for h in range(6):
        k = rng.permutation(half)
        DM[h * half:(h + 1) * half, 1] = k % 2; DM[h * half:(h + 1) * half, 2] = (k // 2) % 2
    for t, (i, a, b) in PORTS.items():
        DM[DM[:, 5] == t, 6:9] = a, b, i
    fr = rng.normal(size=(ntr, nn, nt))
    if signal is not None:
        signal(rng, DM, fr)
    return DM, fr


def region(rng, n, **k):
    out = [session(rng, **k) for _ in range(n)]
    dm = np.empty(n, object); fr = np.empty(n, object)
    for i, (d, f) in enumerate(out):
        dm[i], fr[i] = d, f
    return {'DM': [dm], 'Data': [fr]}


def data(seed=0, hp=None, pfc=None):
    rng = np.random.default_rng(seed)
    return region(rng, N_HP, signal=hp), region(rng, N_PFC, signal=pfc)


def outcome_after_40(rng, DM, fr):
    w = rng.normal(size=fr.shape[1]) * 5
    fr[:, :, 42:] += DM[:, 2][:, None, None] * w[None, :, None]


def test_rsa_planted_outcome_is_recovered_at_the_right_time():
    hp, pfc = data(pfc=outcome_after_40)
    X = rsa._build_rdm_X(False)
    cond, init = rsa._extract_all_times(pfc)
    cpd = np.array([rsa._cpd_from_matrix(rsa._build_matrix(cond, init, t, t + 1), X)[2] for t in range(34, 62)])
    assert cpd[8:].min() > 0.15 and cpd[:8].max() < 0.05         # signal starts at time bin 42 = index 8


def test_rsa_fast_and_reference_extraction_agree():
    hp, _ = data()
    cond, init = rsa._extract_all_times(hp)
    assert np.allclose(rsa._build_matrix(cond, init, 40, 43), rsa.extract_trials(hp, 40, 43))


def test_rsa_port_rdm_matches_port_geometry():
    order = [('A', 1)] * 2 + [('A', 2)] * 2 + [('A', 3)] * 2 + [('I', 1), ('I', 3), ('I', 2)] \
        + [('B', 3)] * 2 + [('B', 2)] * 2 + [('B', 1)] * 2
    port = [PORTS[t]['IAB'.index(k)] for k, t in order]
    expected = np.array([[a == b for b in port] for a in port])
    assert np.array_equal(np.asarray(RSAs.RSA_physical_rdm()), expected)


def test_svd_animal_null_identity_split_equals_real():
    hp, pfc = data()
    inds = np.arange(24, 30)
    for d, axis in ((True, 0), (True, 1), (False, 0)):
        real = svd.real_diff(hp, diagonal=d, axis=axis, inds=inds)[0] - svd.real_diff(pfc, diagonal=d, axis=axis, inds=inds)[0]
        null = [a - b for a, b in svd.permute(pfc, hp, diagonal=d, axis=axis, inds=inds, animal_perm=True)]
        assert len(null) == 35 and np.isclose(null[-1], real)    # last split: PFC = animals 3-6


def test_svd_finds_no_gap_when_tasks_share_structure():
    def shared(rng, DM, fr):
        fr += 8 * rng.normal(size=(fr.shape[1], fr.shape[2]))[None]
    hp, _ = data(hp=shared)
    same = svd.real_diff(hp, diagonal=True, axis=0, inds=np.arange(24, 30))[0]

    def differs(rng, DM, fr):
        for t in (1, 2, 3):
            fr[DM[:, 5] == t] += 8 * rng.normal(size=(fr.shape[1], fr.shape[2]))[None]
    hp2, _ = data(hp=differs)
    differ = svd.real_diff(hp2, diagonal=True, axis=0, inds=np.arange(24, 30))[0]
    assert abs(same) < 0.02 and differ > 5 * abs(same)


def test_policy_diagonal_null_identity_split_equals_real():
    hp, pfc = data()
    a, b, _, _ = pol.permute_sessions_diagonal(hp, pfc, animal_perm=True)
    r_hp = pol.correlations_ab(hp); r_pfc = pol.correlations_ab(pfc)
    assert np.isclose(a[-1], r_pfc[2] - r_hp[2]) and np.isclose(b[-1], r_pfc[3] - r_hp[3])


def test_policy_weights_recover_choice_history():
    rng = np.random.default_rng(3); n = 3000
    c = np.zeros(n); r = rng.integers(0, 2, n)
    for t in range(1, n):
        c[t] = rng.random() < (0.9 if c[t - 1] else 0.1)
    DM = np.zeros((n, 10)); DM[:, 1] = c; DM[:, 2] = r
    beta = pol.policy_for_GLM({'DM': [[DM]]}, 5)
    assert beta[5] > 2 and np.abs(beta[6:10]).max() < 1            # lag 1 choice dominates


def history_session(seed=4, ntr=600):
    """Choices that depend on the previous choice and on whether it was rewarded."""
    rng = np.random.default_rng(seed); c = np.zeros(ntr); r = rng.integers(0, 2, ntr).astype(float)
    for t in range(1, ntr):
        logit = 2 * (c[t - 1] - 0.5) + 4 * (c[t - 1] - 0.5) * (r[t - 1] - 0.5)
        c[t] = rng.random() < 1 / (1 + np.exp(-logit))
    DM = np.zeros((ntr, 10)); DM[:, 1] = c; DM[:, 2] = r; DM[:, 5] = np.repeat([1, 2, 3], ntr // 3)
    for t, (i, a, b) in PORTS.items():
        DM[DM[:, 5] == t, 6:9] = a, b, i
    return DM


def test_applied_policy_is_the_fitted_decision_function():
    from sklearn.linear_model import LogisticRegression
    DM = history_session(); n = 5; c, r = DM[:, 1], DM[:, 2]
    X = pol._policy_design(c, r, n)
    policy = X @ pol.policy_for_GLM({'DM': [[DM]]}, n)
    fit = LogisticRegression(fit_intercept=False).fit(X, c[n:])
    assert np.allclose(policy, fit.decision_function(X))
    loglik = np.mean(c[n:] * policy - np.log1p(np.exp(policy)))
    assert loglik > np.log(0.5) and np.corrcoef(policy, c[n:])[0, 1] > 0.3


def test_policy_regressor_inside_the_glm_matches_the_fitted_policy():
    """A neuron that follows the fitted policy must load on it with weight 1 in policy_A_B."""
    DM = history_session(); n = 5
    beta = pol.policy_for_GLM({'DM': [[DM]]}, n)
    fr = np.zeros((len(DM), 1, 63))
    for t in (1, 2, 3):
        k = np.where(DM[:, 5] == t)[0]
        fr[k[n:], 0, :] = (pol._policy_design(DM[k, 1], DM[k, 2], n) @ beta)[:, None]
    for C in pol.policy_A_B({'DM': [[DM]], 'Data': [[fr]]}, n=n, perm=False):
        assert np.allclose(C[1], 1)


def _coefs(shared_a, shared_b, seed=5, n_neurons=400, nt=63):
    """Policy coefficients per task (predictor x neuron x time); shared_* sets the cross-task correlation."""
    rng = np.random.default_rng(seed)
    out = []
    for shared in (shared_a, shared_b):
        common = rng.normal(size=(n_neurons, 1))
        out += [np.stack([np.zeros((n_neurons, nt)),
                          np.sqrt(shared) * common + np.sqrt(1 - shared) * rng.normal(size=(n_neurons, nt))]) for _ in range(3)]
    return out                                             # C_1_a, C_2_a, C_3_a, C_1_b, C_2_b, C_3_b


def test_policy_slices_are_pfc_minus_ca1_for_a_and_for_b():
    """The slice statistic is PFC - CA1, separately for A (index 0) and B (index 1) choices, at the three anchors."""
    ca1, pfc = _coefs(0.8, 0.1), _coefs(0.5, 0.5)          # cross-problem correlation on A and on B choices
    pfc_ca1 = pol._compute_perm_result(*ca1, *pfc, 1, [24, 35, 42])
    assert pfc_ca1.shape == (2, 3, 63)
    assert np.all(pfc_ca1[0] < -0.2) and np.all(pfc_ca1[1] > 0.2)
    assert np.allclose(pol._compute_perm_result(*pfc, *ca1, 1, [24, 35, 42]), -pfc_ca1)


def test_policy_slice_null_identity_split_equals_real_split():
    hp, pfc = data()
    pfc_ca1 = pol.permute_policy_A_B(hp, pfc, perm_n=1, animal_perm=True)
    assert len(pfc_ca1) == 35
    real = pol._compute_perm_result(*pol.policy_A_B(hp, n=11, perm=False), *pol.policy_A_B(pfc, n=11, perm=False), 1, [24, 35, 42])
    assert np.allclose(pfc_ca1[-1], real)


def halves(structures, seed=0, n_neurons=30, n_time=8, drift=0.0):
    """Six halves (1a, 1b, 2a, 2b, 3a, 3b), one structure per problem plus a stationary random-walk drift:
    each half keeps `1 - drift` of the previous half's pattern (so only the time gap sets the similarity of two halves)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(3, n_neurons, n_time))
    walk = rng.normal(size=(n_neurons, n_time)); out = []
    for k in range(6):
        out.append(base[structures[k // 2]] + 2 * walk + 0.1 * rng.normal(size=(n_neurons, n_time)))
        walk = (1 - drift) * walk + np.sqrt(1 - (1 - drift) ** 2) * rng.normal(size=walk.shape)
    return out


def cumulative(data, modes_from, diagonal, axis):
    u, _, vh = np.linalg.svd(modes_from, full_matrices=False)
    proj = u.T @ data @ vh.T
    w = np.sum(proj ** 2, axis=axis) if diagonal else np.abs(np.diag(proj))
    return np.cumsum(w) / data.shape[0]


def reference(f, diagonal, axis):
    """Independent definition: every pair of halves in both directions, the two cumulative curves averaged.
    Within 1a/1b, 2a/2b, 3a/3b; between 1b/2a, 2b/3a."""
    f1, f1b, f2, f2b, f3, f3b = f
    both = lambda a, b: (cumulative(a, b, diagonal, axis) + cumulative(b, a, diagonal, axis)) / 2
    within = np.mean([both(f1, f1b), both(f2, f2b), both(f3, f3b)], 0)
    between = np.mean([both(f1b, f2), both(f2b, f3)], 0)
    if diagonal:
        within, between = within / within[-1], between / between[-1]
    return within, between


def published_pairing_gap(f, diagonal, axis):
    """The pairing used in the 2022 code, for comparison: 1a/1b and 2a/2b within; 2a and 3a both onto the modes of 1b."""
    f1, f1b, f2, f2b, f3, f3b = f
    within = np.mean([cumulative(f1, f1b, diagonal, axis), cumulative(f2, f2b, diagonal, axis)], 0)
    between = np.mean([cumulative(f2, f1b, diagonal, axis), cumulative(f3, f1b, diagonal, axis)], 0)
    if diagonal:
        within, between = within / within[-1], between / between[-1]
    return (np.trapezoid(within) - np.trapezoid(between)) / within.shape[0]


def test_svd_pairs_are_adjacent_half_steps_in_both_directions():
    f = halves([0, 1, 2])
    for diagonal, axis in ((True, 0), (True, 1), (False, 0)):
        within, between = reference(f, diagonal, axis)
        _, got_between, got_within = svd.svd(*f, diagonal=diagonal, axis=axis)
        assert np.allclose(got_within, within) and np.allclose(got_between, between)


def test_svd_uses_every_half():
    f = halves([0, 1, 2])
    base = svd.svd(*f, diagonal=True, axis=0)[0]
    for k in range(6):
        changed = list(f); changed[k] = np.random.default_rng(1).normal(size=f[k].shape)
        assert not np.isclose(svd.svd(*changed, diagonal=True, axis=0)[0], base)


def test_svd_is_invariant_to_the_direction_of_time():
    """Reversing the order of the halves (3b, 3a, 2b, 2a, 1b, 1a) gives the same curves."""
    f = halves([0, 1, 2])
    rev = [f[5], f[4], f[3], f[2], f[1], f[0]]
    for diagonal, axis in ((True, 0), (True, 1), (False, 0)):
        a = svd.svd(*f, diagonal=diagonal, axis=axis); b = svd.svd(*rev, diagonal=diagonal, axis=axis)
        assert np.allclose(a[1], b[1]) and np.allclose(a[2], b[2])


def test_svd_no_gap_from_drift_alone():
    """The same structure in every problem and a steady drift: every pair is one half-step apart, so no gap is expected,
    whereas the 2022 pairing (1b onto 3a is three half-steps) finds one."""
    gaps = {'now': [], 'published': []}
    for seed in range(8):
        f = halves([0, 0, 0], drift=0.5, seed=seed)
        for diagonal, axis in ((True, 0), (True, 1)):
            gaps['now'].append(svd.svd(*f, diagonal=diagonal, axis=axis)[0])
            gaps['published'].append(published_pairing_gap(f, diagonal, axis))
    assert abs(np.mean(gaps['now'])) < 0.5 * abs(np.mean(gaps['published']))


def test_svd_gap_when_the_problem_changes():
    f = halves([0, 1, 2])
    for diagonal, axis in ((True, 0), (True, 1), (False, 0)):
        assert svd.svd(*f, diagonal=diagonal, axis=axis)[0] > 0.02


def test_svd_between_brains_cross_region_differs_when_structure_differs():
    hp, pfc = data()
    plt.close('all'); svd.svd_between_brains(hp, pfc)
    own_hp, cross_hp, own_pfc, cross_pfc = [line.get_ydata() for line in plt.gca().lines[:4]]
    plt.close('all')
    m = min(len(own_hp), len(cross_hp))  # the number of modes is set by the region the modes come from
    assert not np.allclose(own_hp[:m], cross_hp[:m]) and not np.allclose(own_pfc[:m], cross_pfc[:m])


def test_svd_panel_is_reproducible_and_prints_its_p_value(capsys):
    hp, pfc = data()
    inds = np.arange(24, 30)
    results = []
    for _ in range(2):
        plt.figure(); np.random.seed(5)
        svd.run_permutations_and_plot(hp, pfc, d=True, p=20, axis=0, inds=inds, seed=7)
        plt.close('all'); results.append(capsys.readouterr().out)
    assert results[0] == results[1] and 'P = ' in results[0]

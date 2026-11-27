"""Helper functions shared across analysis modules.

Contains:
- animal_exp_permute: split sessions by animal for permutation tests
- _CPD / _cpd: coefficient of partial determination (multi-output / single-output)
- exp_mov_ave: exponential moving average
- task_ind: consistent task ID assignment based on poke locations
- exponential: simple exponential function for curve fitting
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# Sessions per animal for each brain region
HP_SESSIONS_PER_ANIMAL = (15, 7)       # 3 CA1 animals (last group is remainder)
PFC_SESSIONS_PER_ANIMAL = (9, 16, 13)  # 4 PFC animals (last group is remainder)

# Standard x-axis tick positions and labels for trial-aligned plots
XTICK_POSITIONS = [0, 12, 24, 35, 42, 49, 62]      # last tick on the last time bin (63 bins, 0-62)
XTICK_POSITIONS_DLC = [0, 12.5, 25, 38, 45, 51, 63]  # 64 bins, 0-63
XTICK_LABELS = ['-1', '-0.5', 'Init', 'Ch', 'R', '+0.5', '+1']

# Key event time indices within trial
TIME_INDICES = [24, 35, 42]       # init, choice, outcome
TIME_INDICES_DLC = [25, 38, 45]

# NOTE: DLC tick positions vary by function due to different trial lengths.
# Some functions (plot_policy_simple_GLM, plot_diagonal_sums) use 63 as the
# last DLC tick instead of 64. Do NOT blindly apply XTICK_POSITIONS_DLC
# everywhere — check the original values in each function.


def animal_exp_permute(all_sessions_HP, all_sessions_PFC):
    """Split HP and PFC session data by animal for permutation testing.

    Splits the neural recording sessions from hippocampus (HP) and prefrontal
    cortex (PFC) into per-animal groups (3 HP animals, 4 PFC animals) and
    returns matched lists of design matrices and firing rate arrays.

    Parameters
    ----------
    all_sessions_HP : dict
        HP data dict with keys 'DM' (design matrices) and 'Data' (firing rates).
    all_sessions_PFC : dict
        PFC data dict with same structure.

    Returns
    -------
    all_subjects_DM : list of ndarray
        Design matrices grouped by animal (7 groups: 3 HP + 4 PFC).
    all_subjects_fr : list of ndarray
        Firing rates grouped by animal (same order).
    """
    HP_1, HP_2 = HP_SESSIONS_PER_ANIMAL
    PFC_1, PFC_2, PFC_3 = PFC_SESSIONS_PER_ANIMAL

    # Design matrices per animal
    HP_all_sessions_1 = all_sessions_HP['DM'][0][:HP_1]
    HP_all_sessions_2 = all_sessions_HP['DM'][0][HP_1:HP_1 + HP_2]
    HP_all_sessions_3 = all_sessions_HP['DM'][0][HP_1 + HP_2:]

    PFC_all_sessions_1 = all_sessions_PFC['DM'][0][:PFC_1]
    PFC_all_sessions_2 = all_sessions_PFC['DM'][0][PFC_1:PFC_1 + PFC_2]
    PFC_all_sessions_3 = all_sessions_PFC['DM'][0][PFC_1 + PFC_2:PFC_1 + PFC_2 + PFC_3]
    PFC_all_sessions_4 = all_sessions_PFC['DM'][0][PFC_1 + PFC_2 + PFC_3:]

    # Firing rates per animal
    HP_all_sessions_1_fr = all_sessions_HP['Data'][0][:HP_1]
    HP_all_sessions_2_fr = all_sessions_HP['Data'][0][HP_1:HP_1 + HP_2]
    HP_all_sessions_3_fr = all_sessions_HP['Data'][0][HP_1 + HP_2:]

    PFC_all_sessions_1_fr = all_sessions_PFC['Data'][0][:PFC_1]
    PFC_all_sessions_2_fr = all_sessions_PFC['Data'][0][PFC_1:PFC_1 + PFC_2]
    PFC_all_sessions_3_fr = all_sessions_PFC['Data'][0][PFC_1 + PFC_2:PFC_1 + PFC_2 + PFC_3]
    PFC_all_sessions_4_fr = all_sessions_PFC['Data'][0][PFC_1 + PFC_2 + PFC_3:]

    all_subjects_DM = [HP_all_sessions_1, HP_all_sessions_2, HP_all_sessions_3,
                       PFC_all_sessions_1, PFC_all_sessions_2, PFC_all_sessions_3, PFC_all_sessions_4]
    all_subjects_fr = [HP_all_sessions_1_fr, HP_all_sessions_2_fr, HP_all_sessions_3_fr,
                       PFC_all_sessions_1_fr, PFC_all_sessions_2_fr, PFC_all_sessions_3_fr, PFC_all_sessions_4_fr]

    return all_subjects_DM, all_subjects_fr


def exact_perm_threshold(null, alpha=0.05, axis=0):
    """Threshold for an exact permutation test on a small, fully enumerated null.

    With N enumerated splits (the real split is one of them) the exact
    one-sided P value of the real statistic is rank / N, where rank counts the
    null values that are >= the real one. ``real > threshold`` is True exactly
    when that P value is <= alpha. An interpolated np.percentile is not exact:
    with the 35 animal splits it lets a statistic ranked 2nd (P = 2/35 = 0.057)
    pass the 95th percentile, and lets the largest one pass the 99th percentile
    although the smallest possible P is 1/35 = 0.029.

    Parameters
    ----------
    null : array-like
        Null statistics, permutations along ``axis``.
    alpha : float
        Significance level.
    axis : int
        Axis that indexes permutations.

    Returns
    -------
    threshold : ndarray (or 0-d array)
        The (k+1)-th largest null value, k = floor(alpha * N). np.inf when
        k == 0, i.e. when alpha cannot be reached with N permutations.
    """
    null = np.asarray(null, dtype=float)
    n = null.shape[axis]
    k = int(np.floor(alpha * n + 1e-9))
    if k < 1:
        return np.full(np.delete(null.shape, axis).astype(int), np.inf)
    return np.take(np.sort(null, axis=axis), n - 1 - k, axis=axis)


def exact_perm_p(real, null, tol=1e-9):
    """Exact one-sided permutation P value: fraction of null values >= real.

    ``tol`` makes the count robust to floating-point noise and counts ties
    against the real value (decoding scores are discrete, so ties do occur).
    The real split is part of the enumerated null, so P is at least 1/N.
    """
    null = np.asarray(null, dtype=float)
    rank = max(int(np.sum(null >= real - tol)), 1)
    return rank / len(null), rank


def _CPD(X, y):
    """Coefficient of partial determination for multi-output regression.

    Uses sklearn LinearRegression to compute CPD for each predictor in X,
    measuring how much each predictor uniquely contributes to explaining
    variance in y beyond the other predictors.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_predictors)
        Design matrix (no intercept fitted internally).
    y : ndarray, shape (n_trials, n_outputs)
        Response matrix (e.g., firing rates across neurons/timepoints).

    Returns
    -------
    cpd : ndarray, shape (n_outputs, n_predictors)
        CPD values. NaN if the reduced model explains all variance (SSE = 0).
    """
    ols = LinearRegression(copy_X=True, fit_intercept=False)
    ols.fit(X, y)
    sse = np.sum((ols.predict(X) - y) ** 2, axis=0)
    cpd = np.zeros([y.shape[1], X.shape[1]])
    for i in range(X.shape[1]):
        X_i = np.delete(X, i, axis=1)
        ols.fit(X_i, y)
        sse_X_i = np.sum((ols.predict(X_i) - y) ** 2, axis=0)
        if len(np.where(sse_X_i == 0)[0]) > 0:
            cpd[:, i] = np.nan
        else:
            cpd[:, i] = (sse_X_i - sse) / sse_X_i
    return cpd


def precompute_CPD(X):
    """Precompute residual-making matrices for fast repeated CPD with the same X.

    For roll-permutation tests where X stays fixed and only y changes,
    this avoids recomputing pseudoinverses on every call.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_predictors)

    Returns
    -------
    M_full : ndarray, shape (n_trials, n_trials)
        Residual maker for the full model.
    Ms : list of ndarray
        Residual makers for each leave-one-predictor-out reduced model.
    """
    I = np.eye(X.shape[0])
    M_full = I - X @ np.linalg.pinv(X)
    Ms = []
    for i in range(X.shape[1]):
        X_i = np.delete(X, i, axis=1)
        Ms.append(I - X_i @ np.linalg.pinv(X_i))
    return M_full, Ms


def CPD_from_precomputed(M_full, Ms, y):
    """Compute CPD using precomputed residual-making matrices.

    Parameters
    ----------
    M_full : ndarray from precompute_CPD
    Ms : list of ndarray from precompute_CPD
    y : ndarray, shape (n_trials, n_outputs)

    Returns
    -------
    cpd : ndarray, shape (n_outputs, n_predictors)
    """
    sse = np.sum((M_full @ y) ** 2, axis=0)
    cpd = np.zeros([y.shape[1], len(Ms)])
    for i, M_i in enumerate(Ms):
        sse_i = np.sum((M_i @ y) ** 2, axis=0)
        zero_mask = sse_i == 0
        if np.any(zero_mask):
            cpd[:, i] = np.nan
        else:
            cpd[:, i] = (sse_i - sse) / sse_i
    return cpd


def _cpd(X, y):
    """Coefficient of partial determination for single-output regression.

    Uses pseudoinverse for fast computation. Suitable when y is 1D.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_predictors)
        Design matrix.
    y : ndarray, shape (n_trials,)
        Response vector.

    Returns
    -------
    cpd : ndarray, shape (n_predictors,)
        CPD value for each predictor.
    """
    pdes = np.linalg.pinv(X)
    pe = np.matmul(pdes, y)
    Y_predict = np.matmul(X, pe)
    sse = np.sum((Y_predict - y) ** 2, axis=0)
    cpd = np.zeros([X.shape[1]])
    for i in range(X.shape[1]):
        X_i = np.delete(X, i, axis=1)
        pdes_i = np.linalg.pinv(X_i)
        pe_i = np.matmul(pdes_i, y)
        Y_predict_i = np.matmul(X_i, pe_i)
        sse_X_i = np.sum((Y_predict_i - y) ** 2, axis=0)
        cpd[i] = (sse_X_i - sse) / sse_X_i
    return cpd


def exp_mov_ave(data, tau=8, initValue=0.5, alpha=None):
    """Exponential moving average for 1D data.

    Parameters
    ----------
    data : array-like
        Input time series.
    tau : float
        Time constant (used if alpha is None). Default 8.
    initValue : float
        Initial value of the moving average. Default 0.5.
    alpha : float or None
        Learning rate. If None, computed as 1 - exp(-1/tau).

    Returns
    -------
    mov_ave : ndarray
        Smoothed time series (same length as data).
    """
    if not alpha:
        alpha = 1. - np.exp(-1. / tau)
    mov_ave = np.zeros(np.size(data) + 1)
    mov_ave[0] = initValue
    for i, x in enumerate(data):
        mov_ave[i + 1] = (1. - alpha) * mov_ave[i] + alpha * x
    return mov_ave[1:]


def task_ind(task, a_pokes, b_pokes):
    """Assign consistent task IDs based on poke port geometry.

    Task 1: A and B at left-right extremes (b_pokes == 10 - a_pokes).
    Task 2: B is one of the diagonal ports (2, 3, 7, or 8).
    Task 3: B is top or bottom (1 or 9).

    Parameters
    ----------
    task : ndarray
        Task labels (not directly used; length determines output size).
    a_pokes : ndarray
        A-poke port numbers.
    b_pokes : ndarray
        B-poke port numbers.

    Returns
    -------
    taskid : ndarray
        Task IDs (1, 2, or 3) for each trial.
    """
    taskid = np.zeros(len(task))
    taskid[b_pokes == 10 - a_pokes] = 1
    taskid[np.isin(b_pokes, [2, 3, 7, 8])] = 2
    taskid[np.isin(b_pokes, [1, 9])] = 3
    return taskid


def exponential(x, a, k, b):
    """Exponential function: a * exp(k * x) + b.

    For use with scipy.optimize.curve_fit.
    """
    return a * np.exp(k * x) + b


def circular_shift(n_rows, min_shift):
    """Random circular shift for a roll-permutation null: at least `min_shift` rows away from the unshuffled data
    in either direction, i.e. uniform on [min_shift, n_rows - min_shift]. `n_rows` must be the length of the array
    that is rolled (bugs 28 and 29: shifts drawn up to a longer length wrapped round to the identity or near it)."""
    if n_rows - 2 * min_shift < 0:
        return np.random.randint(1, n_rows)
    return np.random.randint(min_shift, n_rows - min_shift + 1)

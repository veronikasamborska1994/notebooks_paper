"""Representational Similarity Analysis (RSA) across time with permutation testing.

Computes how well different representational dissimilarity matrices (RDMs)
explain neural population similarity structure at each timepoint. Tests
whether representations are task-general using session-shuffle and
animal-shuffle permutations.

Key functions:
- extract_trials: extract trial-averaged firing rates by condition
- regression_RSA: fit RDMs to neural RDMs using CPD
- RSA_across_time_plot: main plotting function (Figure 4C-E)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from palettable import wesanderson as wes
from functions import RSAs as rs
from itertools import combinations
from math import factorial
import palettable
from tqdm.auto import trange
from functions.helper_functions import task_ind, _cpd, animal_exp_permute


# ---------------------------------------------------------------------------
# Condition specs for the 15-row RSA matrix
# ---------------------------------------------------------------------------
# (task_id, choice_val, reward_val) for the 12 choice/reward conditions
_COND_SPECS = [
    (1, 1, 1), (1, 1, 0), (2, 1, 1), (2, 1, 0), (3, 1, 1), (3, 1, 0),  # a_1_r .. a_3_nr
    (3, 0, 1), (3, 0, 0), (2, 0, 1), (2, 0, 0), (1, 0, 1), (1, 0, 0),  # b_3_r .. b_1_nr
]
_INIT_TASKS = [1, 3, 2]  # row order: i_1, i_3, i_2
_INIT_START, _INIT_STOP = 24, 26


def extract_trials(data, t_start, t_end, perm_n = False, animal_perm = False):
    '''This function arranges data in a format used in later RSA analysis - finds A reward, A non-rewarded,
    B reward, B non-rewarded, Initiation average firing rates for each neuron.
    t_start and t_end are used to extract firing rates in time windows around and after choice.
    perm_n here is used solely to access data.'''
    if perm_n: # if it's a permutation test access data this way -->
        dm = data[0]
        firing = data[1]
        if animal_perm == True: # if it's an animal based permutation test
            dm = np.concatenate(dm,0)
            firing = np.concatenate(firing,0)

    else:
        dm = data['DM'][0]
        firing = data['Data'][0]


    neurons = 0
    for s in firing:
        neurons += s.shape[1]

    a_1_nr = np.zeros((neurons)); a_2_nr = np.zeros((neurons)); a_3_nr = np.zeros((neurons))
    a_1_r = np.zeros((neurons)); a_2_r = np.zeros((neurons)); a_3_r = np.zeros((neurons))
    b_1_nr = np.zeros((neurons)); b_2_nr = np.zeros((neurons)); b_3_nr = np.zeros((neurons))
    b_1_r = np.zeros((neurons)); b_2_r = np.zeros((neurons)); b_3_r = np.zeros((neurons))
    i_1 = np.zeros((neurons)); i_2 = np.zeros((neurons)); i_3 = np.zeros((neurons))
    init_start = 24;  init_stop = 26 # initiation is held the same whereas t_start and t_end allow to select for windows after choice

    n_neurons_cum = 0;
    for  s, sess in enumerate(dm):
        DM = dm[s]
        firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        n_neurons_cum += n_neurons
        # trial information
        choices = DM[:,1]; task =  DM[:,5]; a_pokes = DM[:,6]; b_pokes = DM[:,7]; reward = DM[:,2]
        taskid = task_ind(task, a_pokes, b_pokes)
        # task IDs
        task_1 = np.where(taskid == 1)[0]; task_2 = np.where(taskid == 2)[0]; task_3 = np.where(taskid == 3)[0]
        # find trial indicies for A rewarded choices
        task_1_a_r = np.where((choices == 1) & (reward == 1) & (taskid == 1))[0]
        task_2_a_r = np.where((choices == 1) & (reward == 1) & (taskid == 2))[0]
        task_3_a_r = np.where((choices == 1) & (reward == 1) & (taskid == 3))[0]
        # find trial indicies for A non-rewarded choices
        task_1_a_nr = np.where((choices == 1) & (reward == 0) & (taskid == 1))[0]
        task_2_a_nr = np.where((choices == 1) & (reward == 0) & (taskid == 2))[0]
        task_3_a_nr = np.where((choices == 1) & (reward == 0) & (taskid == 3))[0]
        # find trial indicies for B rewarded choices
        task_1_b_r = np.where((choices == 0) & (reward == 1) & (taskid == 1))[0]
        task_2_b_r = np.where((choices == 0) & (reward == 1) & (taskid == 2))[0]
        task_3_b_r = np.where((choices == 0) & (reward == 1) & (taskid == 3))[0]
       # find trial indicies for B non-rewarded choices
        task_1_b_nr = np.where((choices == 0) & (reward == 0) & (taskid == 1))[0]
        task_2_b_nr = np.where((choices == 0) & (reward == 0) & (taskid == 2))[0]
        task_3_b_nr = np.where((choices == 0) & (reward == 0) & (taskid == 3))[0]

        # store mean firing rates for A reward, A no reward, B reward, B no reward, Initiation in all tasks
        a_1_r[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_1_a_r,:, t_start:t_end],2),0)
        a_2_r[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_2_a_r,:, t_start:t_end],2),0)
        a_3_r[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_3_a_r,:, t_start:t_end],2),0)

        a_1_nr[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_1_a_nr,:, t_start:t_end],2),0)
        a_2_nr[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_2_a_nr,:, t_start:t_end],2),0)
        a_3_nr[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_3_a_nr,:, t_start:t_end],2),0)

        b_1_r[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_1_b_r,:, t_start:t_end],2),0)
        b_2_r[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_2_b_r,:, t_start:t_end],2),0)
        b_3_r[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_3_b_r,:, t_start:t_end],2),0)

        b_1_nr[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_1_b_nr,:, t_start:t_end],2),0)
        b_2_nr[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_2_b_nr,:, t_start:t_end],2),0)
        b_3_nr[n_neurons_cum-n_neurons:n_neurons_cum] =  np.mean(np.mean(firing_rates[task_3_b_nr,:, t_start:t_end],2),0)

        i_1[n_neurons_cum-n_neurons:n_neurons_cum] = np.mean(np.mean(firing_rates[task_1,:, init_start:init_stop],2),0)
        i_2[n_neurons_cum-n_neurons:n_neurons_cum] = np.mean(np.mean(firing_rates[task_2,:, init_start:init_stop],2),0)
        i_3[n_neurons_cum-n_neurons:n_neurons_cum] = np.mean(np.mean(firing_rates[task_3,:, init_start:init_stop],2),0)

    matrix_for_correlations = np.vstack([a_1_r,a_1_nr,a_2_r,a_2_nr,a_3_r,a_3_nr,\
                                        i_1,i_3,i_2, b_3_r, b_3_nr, b_2_r, b_2_nr, b_1_r, b_1_nr])

    return matrix_for_correlations


# ---------------------------------------------------------------------------
# Optimised helpers: extract all timepoints at once, then slice
# ---------------------------------------------------------------------------

def _extract_all_times(data, perm_n=False, animal_perm=False):
    """Extract condition-averaged firing rates for ALL timepoints at once.

    Returns
    -------
    cond_all : ndarray, shape (12, n_neurons, n_timepoints)
        Trial-averaged rates for 12 choice × reward × task conditions.
    init_all : ndarray, shape (3, n_neurons)
        Initiation-period averages for tasks 1, 3, 2 (fixed window 24:26).
    """
    if perm_n:
        dm = data[0]; firing = data[1]
        if animal_perm:
            dm = np.concatenate(dm, 0); firing = np.concatenate(firing, 0)
    else:
        dm = data['DM'][0]; firing = data['Data'][0]

    neurons = sum(s.shape[1] for s in firing)
    n_tp = firing[0].shape[2]

    cond_all = np.zeros((12, neurons, n_tp))
    init_all = np.zeros((3, neurons))

    n_neurons_cum = 0
    for s in range(len(dm)):
        firing_rates = firing[s]; DM = dm[s]
        n_trials, n_neurons, _ = firing_rates.shape
        sl = slice(n_neurons_cum, n_neurons_cum + n_neurons)

        choices = DM[:,1]; task = DM[:,5]; a_pokes = DM[:,6]; b_pokes = DM[:,7]; reward = DM[:,2]
        taskid = task_ind(task, a_pokes, b_pokes)

        for c, (t, ch, rw) in enumerate(_COND_SPECS):
            trials = np.where((choices == ch) & (reward == rw) & (taskid == t))[0]
            if len(trials) > 0:
                cond_all[c, sl, :] = np.mean(firing_rates[trials], 0)

        for t_idx, t in enumerate(_INIT_TASKS):
            trials = np.where(taskid == t)[0]
            if len(trials) > 0:
                init_all[t_idx, sl] = np.mean(np.mean(firing_rates[trials, :, _INIT_START:_INIT_STOP], 2), 0)

        n_neurons_cum += n_neurons

    return cond_all, init_all


def _build_matrix(cond_all, init_all, t_start, t_end):
    """Build 15-condition × neurons matrix from precomputed all-times data."""
    cond_t = np.mean(cond_all[:, :, t_start:t_end], 2)  # (12, neurons)
    return np.vstack([cond_t[:6], init_all, cond_t[6:]])


def _cpd_from_matrix(matrix, X):
    """Compute CPD from a condition × neuron matrix."""
    corr = np.corrcoef(matrix)
    np.fill_diagonal(corr, np.nan)
    y = corr.flatten()
    y = y[~np.isnan(y)]
    return _cpd(X, y)


def _build_rdm_X(dlc):
    """Build the RDM design matrix (constant across time and permutations)."""
    physical_rsa = np.asarray(rs.RSA_physical_rdm(), dtype=np.float32)
    np.fill_diagonal(physical_rsa, np.nan)
    physical_rsa = physical_rsa.flatten(); physical_rsa = physical_rsa[~np.isnan(physical_rsa)]

    choice_ab_rsa = np.asarray(rs.RSA_a_b_rdm(), dtype=np.float32)
    np.fill_diagonal(choice_ab_rsa, np.nan)
    choice_ab_rsa = choice_ab_rsa.flatten(); choice_ab_rsa = choice_ab_rsa[~np.isnan(choice_ab_rsa)]

    reward_no_reward = np.asarray(rs.reward_rdm(), dtype=np.float32)
    np.fill_diagonal(reward_no_reward, np.nan)
    reward_no_reward = reward_no_reward.flatten(); reward_no_reward = reward_no_reward[~np.isnan(reward_no_reward)]

    reward_at_choices = np.asarray(rs.reward_choice_space(), dtype=np.float32)
    np.fill_diagonal(reward_at_choices, np.nan)
    reward_at_choices = reward_at_choices.flatten(); reward_at_choices = reward_at_choices[~np.isnan(reward_at_choices)]

    choice_initiation_rsa = np.asarray(rs.choice_vs_initiation(), dtype=np.float32)
    np.fill_diagonal(choice_initiation_rsa, np.nan)
    choice_initiation_rsa = choice_initiation_rsa.flatten(); choice_initiation_rsa = choice_initiation_rsa[~np.isnan(choice_initiation_rsa)]

    a_specific = np.asarray(rs.a_bs_task_specific(), dtype=np.float32); a_specific[6:,6:] = 0
    np.fill_diagonal(a_specific, np.nan); a_specific = a_specific.flatten(); a_specific = a_specific[~np.isnan(a_specific)]
    ones = np.ones(len(choice_ab_rsa))

    if not dlc:
        return np.vstack([physical_rsa, choice_ab_rsa, reward_no_reward, reward_at_choices, choice_initiation_rsa, a_specific, ones]).T
    else:
        return np.vstack([choice_ab_rsa, reward_no_reward, reward_at_choices, choice_initiation_rsa, a_specific, ones]).T


def _one_perm_rsa(all_subjects_DM, all_subjects_fr, n_hp_sessions, X, t_start, t_end, perm_n, animal_perm):
    '''Single permutation iteration for regression_RSA (session-shuffle branch).'''
    n_sessions = np.arange(len(all_subjects_DM))
    np.random.shuffle(n_sessions)
    indices_HP = n_sessions[:n_hp_sessions]; indices_PFC = n_sessions[n_hp_sessions:]
    data_PFC = [all_subjects_DM[indices_PFC], all_subjects_fr[indices_PFC]]
    data_HP  = [all_subjects_DM[indices_HP],  all_subjects_fr[indices_HP]]

    m_pfc = extract_trials(data_PFC, t_start, t_end, perm_n=perm_n, animal_perm=animal_perm)
    m_pfc = np.corrcoef(m_pfc); np.fill_diagonal(m_pfc, np.nan); m_pfc = m_pfc.flatten()
    y_pfc = m_pfc[~np.isnan(m_pfc)]

    m_hp = extract_trials(data_HP, t_start, t_end, perm_n=perm_n, animal_perm=animal_perm)
    m_hp = np.corrcoef(m_hp); np.fill_diagonal(m_hp, np.nan); m_hp = m_hp.flatten()
    y_hp = m_hp[~np.isnan(m_hp)]

    return _cpd(X, y_pfc), _cpd(X, y_hp)


def regression_RSA(HP, PFC, t_start, t_end, perm_n=False, animal_perm=False, dlc=False):
    """Fit RDMs to neural correlation matrices using CPD regression (Fig 4C-E).

    Computes neural RDMs (Pearson correlation between condition-averaged
    firing rates) separately for HP and PFC, then regresses model RDMs
    against them. Optionally runs session-shuffle or animal-shuffle
    permutation tests.

    Parameters
    ----------
    HP, PFC : dict or list
        Neural data (raw dict or permuted list).
    t_start, t_end : int
        Time indices for the analysis window.
    perm_n : int or bool
        Number of session-shuffle permutations. False for no permutation.
    animal_perm : bool
        If True, use animal-shuffle permutations instead.
    dlc : bool
        If True, exclude the port RDM (already regressed out via DLC).

    Returns
    -------
    None (plots figures directly).
    """
    # extract firing rates for HP data and compute a correlation matrix
    matrix_for_correlations_HP = extract_trials(HP, t_start, t_end, perm_n = False, animal_perm = animal_perm)
    correlation_m = np.corrcoef(matrix_for_correlations_HP)
    np.fill_diagonal(correlation_m,np.nan); correlation_m_f = correlation_m.flatten()
    correlation_m_f = correlation_m_f[~np.isnan(correlation_m_f)]

    # extract firing rates for PFC data and compute a correlation matrix
    matrix_for_correlations_PFC = extract_trials(PFC, t_start, t_end, perm_n = False,  animal_perm = animal_perm)
    correlation_m_PFC = np.corrcoef(matrix_for_correlations_PFC)
    np.fill_diagonal(correlation_m_PFC,np.nan); correlation_m_PFC_f = correlation_m_PFC.flatten()
    correlation_m_PFC_f = correlation_m_PFC_f[~np.isnan(correlation_m_PFC_f)]

    X = _build_rdm_X(dlc)
    cpd_HP = _cpd(X, correlation_m_f); cpd_PFC = _cpd(X, correlation_m_PFC_f)

    pfc_perm = None; hp_perm = None
    if perm_n: # if it's a permutation test
        all_subjects_DM = np.concatenate((HP['DM'][0],PFC['DM'][0]),0);  all_subjects_fr = np.concatenate((HP['Data'][0],PFC['Data'][0]),0)
        n_hp_sessions = len(HP['DM'][0])
        pfc_perm = np.zeros((int(perm_n),X.shape[1])); hp_perm = np.zeros((int(perm_n),X.shape[1]))
        if animal_perm == False:
            results = [_one_perm_rsa(all_subjects_DM, all_subjects_fr, n_hp_sessions, X, t_start, t_end, perm_n, animal_perm)
                       for _ in trange(int(perm_n), desc='RSA perms', leave=False)]
            for i, (cpd_PFC_perm, cpd_HP_perm) in enumerate(results):
                pfc_perm[i,:] = cpd_PFC_perm; hp_perm[i,:] = cpd_HP_perm
        else:
            animals_PFC = [0,1,2,3];  animals_HP = [4,5,6]
            m, n = len(animals_PFC), len(animals_HP)
            num_rounds = factorial(m + n) / (factorial(m)*factorial(n))
            pfc_perm = np.zeros((int(num_rounds),X.shape[1])); hp_perm = np.zeros((int(num_rounds),X.shape[1])) # array to store permuted cpds
            all_subjects_DM, all_subjects_fr  = animal_exp_permute(HP,PFC)
            nn = 0
            for indices_PFC in combinations(range(m + n), m):
                indices_HP = [i for i in range(m + n) if i not in indices_PFC]
                DM_PFC = np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_PFC)]
                Data_PFC = np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_PFC)]; data_PFC = [DM_PFC,Data_PFC]

                DM_HP = np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_HP)]
                Data_HP = np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_HP)]
                data_HP = [DM_HP,Data_HP]

                matrix_for_correlations_perm_PFC = extract_trials(data_PFC, t_start, t_end, perm_n  = perm_n,  animal_perm = animal_perm)
                matrix_for_correlations_perm_PFC = np.corrcoef(matrix_for_correlations_perm_PFC)
                np.fill_diagonal(matrix_for_correlations_perm_PFC,np.nan)
                matrix_for_correlations_perm_PFC= matrix_for_correlations_perm_PFC.flatten()
                y_perm_PFC = matrix_for_correlations_perm_PFC[~np.isnan(matrix_for_correlations_perm_PFC)]
                cpd_PFC_perm = _cpd(X,y_perm_PFC)

                matrix_for_correlations_perm_HP = extract_trials(data_HP, t_start, t_end, perm_n  = perm_n,  animal_perm = animal_perm)
                matrix_for_correlations_perm_HP  = np.corrcoef(matrix_for_correlations_perm_HP)
                np.fill_diagonal(matrix_for_correlations_perm_HP,np.nan)
                matrix_for_correlations_perm_HP= matrix_for_correlations_perm_HP.flatten()
                y_perm_HP = matrix_for_correlations_perm_HP[~np.isnan(matrix_for_correlations_perm_HP)]
                cpd_HP_perm = _cpd(X,y_perm_HP)
                pfc_perm[nn,:] = cpd_PFC_perm;  hp_perm[nn,:]  = cpd_HP_perm
                nn += 1

    return cpd_PFC, cpd_HP, matrix_for_correlations_PFC, matrix_for_correlations_HP, pfc_perm, hp_perm


def RSA_across_time_plot(HP, PFC, perm_n = 5, animal_perm = False, dlc = False):
    if dlc == False:
        cue = np.arange(34,62);  reward = np.arange(35,63)
        prtd = 0; ab = 1; re = 2;  re_ab = 3; ch_i = 4; tsp = 5
    else:
        cue = np.arange(37,61);  reward = np.arange(38,62)
        ab = 0; re = 1;  re_ab = 2; ch_i = 3; tsp = 4

    X = _build_rdm_X(dlc)
    n_times = len(cue)
    n_features = X.shape[1]

    # --- Real data: extract all timepoints once, then slice ---
    hp_cond, hp_init = _extract_all_times(HP)
    pfc_cond, pfc_init = _extract_all_times(PFC)

    corr_HP = []; corr_PFC = []
    cpd_PFC_list = []; cpd_HP_list = []
    diff_space = []; diff_a_b = []; diff_rew = []; diff_rew_ab = []; diff_ch_init = []; diff_a_spec = []

    for start, end in zip(cue, reward):
        mat_hp = _build_matrix(hp_cond, hp_init, start, end)
        mat_pfc = _build_matrix(pfc_cond, pfc_init, start, end)
        corr_HP.append(mat_hp); corr_PFC.append(mat_pfc)

        cpd_HP = _cpd_from_matrix(mat_hp, X)
        cpd_PFC = _cpd_from_matrix(mat_pfc, X)
        cpd_HP_list.append(cpd_HP); cpd_PFC_list.append(cpd_PFC)

        if not dlc:
            diff_space.append(cpd_HP[prtd] - cpd_PFC[prtd])
        diff_a_b.append(cpd_HP[ab] - cpd_PFC[ab])
        diff_rew.append(cpd_PFC[re] - cpd_HP[re])
        diff_rew_ab.append(cpd_HP[re_ab] - cpd_PFC[re_ab])
        diff_ch_init.append(cpd_PFC[ch_i] - cpd_HP[ch_i])
        diff_a_spec.append(cpd_HP[tsp] - cpd_PFC[tsp])

    cpd_PFC_list = np.asarray(cpd_PFC_list); cpd_HP_list = np.asarray(cpd_HP_list)

    # --- Permutations: one shuffle → all timepoints (28x faster) ---
    all_hp_perm = np.zeros((n_times, int(perm_n), n_features))
    all_pfc_perm = np.zeros((n_times, int(perm_n), n_features))

    if perm_n:
        if not animal_perm:
            all_subjects_DM = np.concatenate((HP['DM'][0], PFC['DM'][0]), 0)
            all_subjects_fr = np.concatenate((HP['Data'][0], PFC['Data'][0]), 0)
            n_hp = len(HP['DM'][0])

            for p in trange(int(perm_n), desc='RSA perms', leave=False):
                n_sessions = np.arange(len(all_subjects_DM))
                np.random.shuffle(n_sessions)
                hp_idx = n_sessions[:n_hp]; pfc_idx = n_sessions[n_hp:]

                hp_data = [all_subjects_DM[hp_idx], all_subjects_fr[hp_idx]]
                pfc_data = [all_subjects_DM[pfc_idx], all_subjects_fr[pfc_idx]]

                hp_cond_p, hp_init_p = _extract_all_times(hp_data, perm_n=True)
                pfc_cond_p, pfc_init_p = _extract_all_times(pfc_data, perm_n=True)

                for t_idx, (start, end) in enumerate(zip(cue, reward)):
                    mat_hp = _build_matrix(hp_cond_p, hp_init_p, start, end)
                    mat_pfc = _build_matrix(pfc_cond_p, pfc_init_p, start, end)
                    all_hp_perm[t_idx, p, :] = _cpd_from_matrix(mat_hp, X)
                    all_pfc_perm[t_idx, p, :] = _cpd_from_matrix(mat_pfc, X)

        else:
            # Animal-level permutation: enumerate all C(7,4) = 35 allocations
            all_subjects_DM, all_subjects_fr = animal_exp_permute(HP, PFC)
            animals_PFC = [0,1,2,3]; animals_HP = [4,5,6]
            m, n_animals = len(animals_PFC), len(animals_HP)
            num_rounds = int(factorial(m + n_animals) / (factorial(m) * factorial(n_animals)))
            all_hp_perm = np.zeros((n_times, num_rounds, n_features))
            all_pfc_perm = np.zeros((n_times, num_rounds, n_features))

            for nn, indices_PFC in enumerate(combinations(range(m + n_animals), m)):
                indices_HP = [i for i in range(m + n_animals) if i not in indices_PFC]
                DM_PFC = np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_PFC)]
                Data_PFC = np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_PFC)]
                DM_HP = np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_HP)]
                Data_HP = np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_HP)]

                hp_cond_p, hp_init_p = _extract_all_times([DM_HP, Data_HP], perm_n=True, animal_perm=True)
                pfc_cond_p, pfc_init_p = _extract_all_times([DM_PFC, Data_PFC], perm_n=True, animal_perm=True)

                for t_idx, (start, end) in enumerate(zip(cue, reward)):
                    mat_hp = _build_matrix(hp_cond_p, hp_init_p, start, end)
                    mat_pfc = _build_matrix(pfc_cond_p, pfc_init_p, start, end)
                    all_hp_perm[t_idx, nn, :] = _cpd_from_matrix(mat_hp, X)
                    all_pfc_perm[t_idx, nn, :] = _cpd_from_matrix(mat_pfc, X)

    # --- Max-statistic multiple comparison correction ---
    signed_diffs = all_hp_perm - all_pfc_perm  # (n_times, perm_n, n_features)
    signed_diffs[:, :, re] = all_pfc_perm[:, :, re] - all_hp_perm[:, :, re]
    signed_diffs[:, :, ch_i] = all_pfc_perm[:, :, ch_i] - all_hp_perm[:, :, ch_i]
    max_per_perm = np.max(signed_diffs, axis=0)  # (perm_n, n_features) - max over time per permutation
    cpd_p_val_mult_compar_95 = np.percentile(max_per_perm, 95, axis=0)
    cpd_p_val_mult_compar_99 = np.percentile(max_per_perm, 99, axis=0)  # P < 0.01

    if not dlc:
        space_sig = np.where(diff_space > cpd_p_val_mult_compar_95[prtd])[0]
        space_sig_99 = np.where(diff_space > cpd_p_val_mult_compar_99[prtd])[0]

    a_b_sig = np.where(diff_a_b > cpd_p_val_mult_compar_95[ab])[0];  a_b_sig_99 = np.where(diff_a_b > cpd_p_val_mult_compar_99[ab])[0]
    rew_sig = np.where(diff_rew > cpd_p_val_mult_compar_95[re])[0]; rew_ab_sig = np.where(diff_rew_ab > cpd_p_val_mult_compar_95[re_ab])[0]
    ch_init_sig = np.where(diff_ch_init > cpd_p_val_mult_compar_95[ch_i])[0];  a_spec_sig = np.where(diff_a_spec > cpd_p_val_mult_compar_95[tsp])[0]
    rew_sig_99 = np.where(diff_rew > cpd_p_val_mult_compar_99[re])[0]; rew_ab_sig_99 = np.where(diff_rew_ab > cpd_p_val_mult_compar_99[re_ab])[0]
    ch_init_sig_99 = np.where(diff_ch_init > cpd_p_val_mult_compar_99[ch_i])[0]; a_spec_sig_99 = np.where(diff_a_spec > cpd_p_val_mult_compar_99[tsp])[0]

    # --- Plotting ---
    inds = [1,8,int(len(cue)/2),len(cue)-1]; isl = wes.Royal2_5.mpl_colors; k = 'grey'
    if dlc == False:
        sbplt = 7
        sign_95 = [space_sig, a_b_sig, rew_sig, rew_ab_sig, ch_init_sig, a_spec_sig]
        sign_99 = [space_sig_99, a_b_sig_99, rew_sig_99, rew_ab_sig_99, ch_init_sig_99, a_spec_sig_99]
        pred_str = ['Port', 'A vs B choice', 'Outcome', 'Outcome at A vs B', 'Choice vs Initiation', 'Problem Specific A choice']
    else:
        sbplt = 7
        sign_95 = [a_b_sig, rew_sig, rew_ab_sig, ch_init_sig, a_spec_sig]
        sign_99 = [a_b_sig_99, rew_sig_99, rew_ab_sig_99, ch_init_sig_99, a_spec_sig_99]
        pred_str = ['A vs B choice', 'Outcome', 'Outcome at A vs B', 'Choice vs Initiation', 'Problem Specific A choice']
    plt.figure(figsize = (15,2.5))
    offset = [6,3,4,3,5,1]
    for pred in np.arange(len(sign_95)):
        plt.subplot(1,sbplt, pred+1)
        plt.plot(cpd_HP_list[:,pred]*100, color = isl[0])
        plt.plot(cpd_PFC_list[:,pred]*100, color = isl[3])
        plt.xticks(inds,['C', 'R', '+0.5','+1'])
        plt.plot(sign_95[pred], np.ones(len(sign_95[pred]))*(np.max([cpd_HP_list[:,pred],cpd_PFC_list[:,pred]]))*100+ offset[pred], '.', markersize = 2, color = 'grey')
        plt.plot(sign_99[pred], np.ones(len(sign_99[pred]))*(np.max([cpd_HP_list[:,pred],cpd_PFC_list[:,pred]]))*100+ offset[pred], '.', markersize = 4, color = 'grey',label = '< .01 diff PFC and HP ')
        plt.ylim(-0.1,(np.max([cpd_HP_list[:,pred], cpd_PFC_list[:,pred]]))*100+offset[pred]+1)
        plt.ylabel('CPD (%)')
        plt.title(pred_str[pred])
        plt.xlabel('Time in Trial (s)')
    plt.tight_layout(); sns.despine()

    cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap
    c_ch_HP = corr_HP[inds[0]]; c_ch_PFC = corr_PFC[inds[0]]
    c_rew_HP = corr_HP[inds[1]]; c_rew_PFC = corr_PFC[inds[1]]

    all_corr_ = [np.corrcoef(c_ch_HP)-np.mean(np.corrcoef(c_ch_HP)), np.corrcoef(c_ch_PFC)-np.mean(np.corrcoef(c_ch_PFC)),\
                np.corrcoef(c_rew_HP)-np.mean(np.corrcoef(c_rew_HP)), np.corrcoef(c_rew_PFC)-np.mean(np.corrcoef(c_rew_PFC))]
    title_str = ['CA1 Choice Time', 'PFC Choice Time', 'CA1 Reward Time', 'PFC Reward Time']
    plt.figure(figsize = (20,10))
    for i, ii in enumerate(all_corr_):
        plt.subplot(2,2,i+1)
        plt.imshow(ii, aspect = 1, cmap = cmap)
        plt.xticks(range(15), ('A T1 R', 'A T1 NR','A T2 R', 'A T2 NR',\
                           'A T3 R','A T3 NR', ' I T1',\
                           'I T3', 'I T2', 'B T3 R',\
                           ' B T3 NR','B T2 R', 'B T2 NR', 'B T1 R', 'B T1 NR'), rotation = 'vertical')

        plt.yticks(range(15), ('A T1 R', 'A T1 NR','A T2 R', 'A T2 NR',\
                           'A T3 R','A T3 NR', ' I T1',\
                           'I T3', ' T2', 'B T3 R',\
                           ' B T3 NR','B T2 R', 'B T2 NR', 'B T1 R', 'B T1 NR'))

        plt.title(title_str[i])
        plt.colorbar(label ='Correlation Coefficient' )

    plt.tight_layout()

    physical_rsa = np.asarray(rs.RSA_physical_rdm(),dtype=np.float32)
    choice_ab_rsa = np.asarray(rs.RSA_a_b_rdm(),dtype=np.float32)
    reward_no_reward = np.asarray(rs.reward_rdm(),dtype=np.float32)
    reward_at_choices = rs.reward_choice_space()
    choice_initiation_rsa =  np.asarray(rs.choice_vs_initiation(),dtype=np.float64)
    a_bs_task_specific_rsa = np.asarray(rs.a_bs_task_specific(),dtype=np.float64)
    a_specific =  np.asarray(a_bs_task_specific_rsa,dtype=np.float32); a_specific[6:,6:]= 0
    if dlc == False:
        all_rsas =[physical_rsa,choice_ab_rsa,reward_no_reward,reward_at_choices,choice_initiation_rsa,a_specific]
    else:
        all_rsas =[choice_ab_rsa,reward_no_reward,reward_at_choices,choice_initiation_rsa,a_specific]

    plt.figure(figsize = (20,10))
    for i,ii in enumerate(all_rsas):

        plt.subplot(1,sbplt,i+1)
        plt.imshow(ii,cmap = cmap)
        plt.xticks(range(15), ('A T1 R', 'A T1 NR','A T2 R', 'A T2 NR',\
                           'A T3 R','A T3 NR', ' I T1',\
                           'I T3', 'I T2', 'B T3 R',\
                           ' B T3 NR','B T2 R', 'B T2 NR', 'B T1 R', 'B T1 NR'), rotation = 'vertical')

        plt.yticks(range(15), ('A T1 R', 'A T1 NR','A T2 R', 'A T2 NR',\
                           'A T3 R','A T3 NR', ' I T1',\
                           'I T3', 'I T2', 'B T3 R',\
                           ' B T3 NR','B T2 R', 'B T2 NR', 'B T1 R', 'B T1 NR'))
        plt.title(pred_str[i])

    plt.tight_layout()

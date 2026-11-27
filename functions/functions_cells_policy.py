"""Neural correlates of behavioural policy (Figure 6).

Analyses how neural firing rates relate to the animal's behavioural policy
(estimated from a logistic regression on the history of choices and outcomes).
Correlates the policy weights of neurons across problems at each pair of time
points, with session-shuffle and animal-shuffle permutation tests.

Key functions:
- plot_policy_during_recordings: logistic regression of choice history (Fig 6A)
- plot_policy_simple_GLM: GLM of policy vs neural activity (Fig 6B)
- plot_correlations_slice: slices through the cross-problem correlation matrices (Fig 6E)
- plot_diagonal_sums: correlation matrices and their diagonal sums (Fig 6C-D)
"""

import scipy
import palettable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from palettable import wesanderson as wes
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from itertools import combinations
from tqdm.auto import trange
from functions.helper_functions import (task_ind, _CPD, precompute_CPD, CPD_from_precomputed, circular_shift,
    animal_exp_permute, exact_perm_threshold, HP_SESSIONS_PER_ANIMAL, PFC_SESSIONS_PER_ANIMAL,
    XTICK_POSITIONS, XTICK_POSITIONS_DLC, XTICK_LABELS, TIME_INDICES, TIME_INDICES_DLC)


def _batch_corrcoef(X, y):
    """Vectorized Pearson correlation between each column of X and vector y.

    Parameters
    ----------
    X : ndarray, shape (n_neurons, n_timepoints)
    y : ndarray, shape (n_neurons,)

    Returns
    -------
    r : ndarray, shape (n_timepoints,)
    """
    X_c = X - X.mean(axis=0)
    y_c = y - y.mean()
    numer = (X_c * y_c[:, None]).sum(axis=0)
    denom = np.sqrt((X_c ** 2).sum(axis=0)) * np.sqrt((y_c ** 2).sum())
    return numer / denom


def _policy_design(choices, reward, n):
    """Design matrix of the policy model: n-back rewards, choices (A = 1, B = 0), choice x reward interactions, constant.

    Used both to fit the policy weights and to apply them, so the two codings cannot differ.
    Row t predicts trial t + n. Shape (n_trials - n, 3n + 1).
    """
    lags = lambda x: scipy.linalg.toeplitz(x, np.zeros(n))[n-1:-1]
    return np.hstack([lags(reward), lags(choices), lags((choices-0.5)*(reward-0.5)*2), np.ones((len(choices)-n, 1))])


def plot_policy_during_recordings(PFC, HP, n = 12): 
    '''Coefficients from a logistic regression predicting current choices using the history of previous choices (I),
    outcomes (not shown) and choice-outcome interactions, Figure 6A.'''
    
    # Split sessions by animal using shared constants
    all_subjects_DM, _ = animal_exp_permute(HP, PFC)
    d = all_subjects_DM
     
    results_array_subj = []; 
    for dm in d:
        results_array = []
        for  s, sess in enumerate(dm):        
            DM = dm[s]; choices = DM[:,1]; reward = DM[:,2]
            X = _policy_design(choices, reward, n)
            choices_current = choices[n:] # current choices need to start at nth trial
            model = LogisticRegression(fit_intercept = False) 
            results = model.fit(X,choices_current) # fit logistic regression predicting current choice based on history of behaviour
            results_array.append(results.coef_[0]) # store coefficients
        results_array_subj.append(np.mean(results_array,0))
    # calculate t-tests (across animals) where coefficients are different from zero
    test = scipy.stats.ttest_1samp(results_array_subj,0)
    pstat = np.round(test.pvalue[n*2:-1],3); tstat = np.round(test.statistic[n*2:-1],3)
    
    for t,tt in enumerate(tstat):
        print('Trial ' + str(t+1) + ' t-stat ' + str(tt), 'p-stat ' + str(pstat[t]))

    average_coefs = np.mean(results_array_subj,0)
    std = np.std(results_array_subj,0)/np.sqrt(len(results_array_subj))
    
    plt.figure(figsize = (8,4)); c = wes.Royal2_5.mpl_colors
    plt.subplot(1,2,1)
    plt.plot(np.arange(len(average_coefs))[n:n*2], average_coefs[n:n*2], color = c[2])
    plt.fill_between(np.arange(len(average_coefs))[n:n*2], average_coefs[n:n*2]+std[n:n*2], average_coefs[n:n*2]- std[n:n*2],alpha = 0.2, color =c[2])
    plt.hlines(0, xmin = np.arange(len(average_coefs))[n:n*2][0],xmax = np.arange(len(average_coefs))[n:n*2][-1])
    length = len(np.arange(len(average_coefs))[n:n*2])
    plt.xticks(np.arange(len(average_coefs))[n:n*2],np.arange(length)+1)
    plt.ylabel('Coefficient');  plt.xlabel('n-back');  sns.despine()
   

    plt.subplot(1,2,2)
    plt.plot(np.arange(len(average_coefs))[n*2:-1], average_coefs[n*2:-1], color = c[3])
    plt.fill_between(np.arange(len(average_coefs))[n*2:-1], average_coefs[n*2:-1]+std[n*2:-1], average_coefs[n*2:-1]- std[n*2:-1],alpha = 0.2, color =c[3])
    plt.hlines(0, xmin = np.arange(len(average_coefs))[n*2:-1][0],xmax = np.arange(len(average_coefs))[n*2:-1][-1])
    length = len(np.arange(len(average_coefs))[n*2:-1])
    plt.xticks(np.arange(len(average_coefs))[n*2:-1],np.arange(length)+1)
    plt.ylabel('Regression Coefficient'); plt.xlabel('n-back trials')
    sns.despine(); plt.tight_layout()
    
    
    
def GLM_policy_roll_tome(data, perm = 1000, n = 11):
    '''Simple linear regression predicting activity of each neuron at each time point across the trial, 
    as a function of the choice, outcome and outcome x choice interaction, policy, policy x choice.
    This function also randomly rolls firing rates with respect to trials.'''
    
    dm = data['DM'][0];  firing = data['Data'][0]
    cpd_perm  = [[] for i in range(perm)] # to store cpds based on permuted data
    cpd  = [] # to store cpds 
    policy_beh = policy_for_GLM(data, n = n, perm = False)
    for  s in trange(len(dm), desc='Policy GLM roll', leave=False):
        DM = dm[s]; firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        choices = DM[:,1]; reward = DM[:,2];  choices_current = choices-0.5

        X = _policy_design(choices, reward, n) # design matrix for calculating policy
        policy = np.matmul(X, policy_beh) # calculate policy on each trial
        
        choices_current = choices_current[n:];  reward = reward[n:];  firing_rates = firing_rates[n:] # current choices need to start at nth trial
        policy_choice = choices_current*policy ; rew_ch = choices_current*reward
                
        _1_back = choices_current[:-1]; choices_current = choices_current[1:]; reward = reward[1:];   policy = policy[1:]
        policy_choice = policy_choice[1:]; rew_ch = rew_ch[1:]; ones = np.ones(len(rew_ch)); firing_rates = firing_rates[1:]
    
        X = np.vstack([choices_current,_1_back,reward,policy, policy_choice,rew_ch, ones]).T # design matrix for cpds           
        n_predictors = X.shape[1]
        y = firing_rates.reshape([len(firing_rates),-1]) # activity matrix [n_trials, n_neurons*n_timepoints]
        cpd.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors)) # calculate coefficients of partial determination
        
        M_full, Ms = precompute_CPD(X)
        for i in range(perm):  # randomly rolls firing rates with respect to trials
            y_perm = np.roll(y, circular_shift(len(y), n), axis = 0) # shift within the rows actually regressed (bug 29)
            cpd_perm[i].append(CPD_from_precomputed(M_full, Ms, y_perm).reshape(n_neurons, n_timepoints, n_predictors))
    
    cpd_perm   = np.stack([np.concatenate(cpd_i,0) for cpd_i in cpd_perm],0)
    cpd = np.concatenate(cpd,0)
    return cpd, cpd_perm


def plot_policy_simple_GLM(HP,PFC, n_perm = 1000, n = 11, dlc = False): 
    ''' Plotting CPDs and significance levels (Figure 6B)''' 
    cpd_HP, cpd_perm_HP =  GLM_policy_roll_tome(HP, perm = n_perm, n = n)
    cpd_PFC, cpd_perm_PFC =  GLM_policy_roll_tome(PFC, perm =  n_perm, n = n)
    cpd_HP_m = np.nanmean(cpd_HP,0); cpd_PFC_m = np.nanmean(cpd_PFC,0)
    _xticks = XTICK_POSITIONS if not dlc else [0, 12.5, 25, 38, 45, 51, 63]
    # lists of cpds and permuted cpds for plotting
    cpds = [cpd_PFC_m[:,:-1],cpd_HP_m[:,:-1]]; cpds_perms = [cpd_perm_PFC[:,:,:,:-1], cpd_perm_HP[:,:,:,:-1]]
    # plotting params
    t = np.arange(0,cpd_HP_m.shape[0]); clrs = wes.Moonrise1_5.mpl_colors + wes.Royal2_5.mpl_colors[1:2:]
    plt.figure(figsize = (15,6))
    titles = ['PFC', 'CA1']; p = ['Choice','Prev Choice','Reward', 'Policy', 'Policy x Choice', 'Reward x Choice']
    # plot cpds and significance levels
    for c,cpd in enumerate(cpds): 
        # Max-statistic correction: max over time per permutation, then percentile
        perm_means = np.mean(cpds_perms[c], 1)  # average across neurons: (perm, n_timepoints, n_predictors)
        max_over_time = np.max(perm_means, axis=1)  # max over time: (perm, n_predictors)
        values_95 = np.percentile(max_over_time, 95, axis=0)  # (n_predictors,)
        values_99 = np.percentile(max_over_time, 99, axis=0)
        cpd_max = np.max(cpd); plt.subplot(1,2,c+1); plt.title(titles[c])
        array_pvals = np.ones((cpd.shape[0], cpd.shape[1])) # store where significant at .05 or .01
        for pred in range(cpd.shape[1]):  # loop through pred
            array_pvals[(np.where(cpd[:,pred] > values_95[pred])[0]),pred] = 0.05
            array_pvals[(np.where(cpd[:,pred] > values_99[pred])[0]),pred] = 0.01
       
        for pred in np.arange(cpd.shape[1]):
            plt.plot(cpd[:,pred]*100, color = clrs[pred], label = p[pred])
            ymax = cpd_max*(1.2+0.02*pred)*100
            p_vals = array_pvals[:,pred]; t05 = t[p_vals == 0.05]; t001 = t[p_vals == 0.01]
            plt.plot(t05, np.ones(t05.shape)*ymax, '.', markersize = 2, color = clrs[pred])
            plt.plot(t001, np.ones(t001.shape)*ymax, '.', markersize = 4, color = clrs[pred])
            plt.xticks(_xticks, XTICK_LABELS)
        plt.xlabel('Time in Trial (s)')
        plt.ylabel('CPD (%)')
    plt.legend(); sns.despine()
    

def policy_for_GLM(data, n, perm = False):
    '''Coefficients from a logistic regression predicting current choices using the history of previous choices (I),
    outcomes (not shown) and choice-outcome interactions to use for policy calculations for PFC and CA1 animals in the GLM of Figure 6B.'''
    if perm:
        dm = data[0]
    else:
        dm = data['DM'][0]
    results_array = []
    for  s, sess in enumerate(dm):     
        DM = dm[s]; choices = DM[:,1]; reward = DM[:,2];
        X = _policy_design(choices, reward, n)
        choices_current = choices[n:] # current choices need to start at nth trial
        model = LogisticRegression(fit_intercept = False) 
        results = model.fit(X,choices_current)# fit logistic regression predicting current choice based on history of behaviour
        results_array.append(results.coef_[0]) # store coefficients
         
    average = np.mean(results_array,0)
    return average

def policy_A_B(data, n = 10, perm = True):
    '''GLM predicting activity of each neuron at each time point across the trial,
    as a function of policy and reward.  Computes A and B choice regressions in a
    single pass through the data (halves runtime versus calling separately).
    perm = True if it is a permutation test.
    Returns C_1_a, C_2_a, C_3_a, C_1_b, C_2_b, C_3_b (shape: n_predictors x n_neurons x n_timepoints).'''

    if perm:
        dm = data[0]; firing = data[1]
    else:
        dm = data['DM'][0]; firing = data['Data'][0]

    C_1_a = []; C_2_a = []; C_3_a = []
    C_1_b = []; C_2_b = []; C_3_b = []
    policy_beh = policy_for_GLM(data, n = n, perm = perm)
    ols = LinearRegression(fit_intercept = False)
    for s, sess in enumerate(dm):
        DM = dm[s]; firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        choices = DM[:,1]; reward = DM[:,2]; task = DM[:,5]; a_pokes = DM[:,6]; b_pokes = DM[:,7]
        taskid = task_ind(task, a_pokes, b_pokes)
        reward_current = reward; choices_current = choices - 0.5
        for i in [1, 2, 3]:
            task_index = np.where(taskid == i)[0]
            rewards_task = reward_current[task_index]; choices_task = choices_current[task_index]
            X = _policy_design(choices[task_index], rewards_task, n)
            policy = np.matmul(X, policy_beh)
            rewards_task = rewards_task[n:]; choices_task = choices_task[n:]
            ones_task = np.ones(len(choices_task))
            firing_rates_task = firing_rates[task_index][n:]
            a = np.where(choices_task == 0.5)[0]; b = np.where(choices_task == -0.5)[0]

            # A choices
            X_a = np.vstack([rewards_task[a], policy[a], ones_task[a]]).T
            n_predictors = X_a.shape[1]
            ols.fit(X_a, firing_rates_task[a].reshape(len(a), -1))
            coef_a = ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)

            # B choices
            X_b = np.vstack([rewards_task[b], policy[b], ones_task[b]]).T
            ols.fit(X_b, firing_rates_task[b].reshape(len(b), -1))
            coef_b = ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)

            if i == 1:
                C_1_a.append(coef_a); C_1_b.append(coef_b)
            elif i == 2:
                C_2_a.append(coef_a); C_2_b.append(coef_b)
            elif i == 3:
                C_3_a.append(coef_a); C_3_b.append(coef_b)

    C_1_a = np.concatenate(C_1_a, 0); C_2_a = np.concatenate(C_2_a, 0); C_3_a = np.concatenate(C_3_a, 0)
    C_1_b = np.concatenate(C_1_b, 0); C_2_b = np.concatenate(C_2_b, 0); C_3_b = np.concatenate(C_3_b, 0)

    # keep neurons whose coefficients are finite in every layout and choice (coefficients are neurons x time x predictors)
    keep = np.all([np.isfinite(C).all(axis=(1, 2)) for C in (C_1_a, C_2_a, C_3_a, C_1_b, C_2_b, C_3_b)], axis=0)
    C_1_a, C_2_a, C_3_a, C_1_b, C_2_b, C_3_b = [np.transpose(C[keep], [2, 0, 1]) for C in (C_1_a, C_2_a, C_3_a, C_1_b, C_2_b, C_3_b)]
    return C_1_a, C_2_a, C_3_a, C_1_b, C_2_b, C_3_b

def _compute_perm_result(C_1_HP_a, C_2_HP_a, C_3_HP_a, C_1_HP_b, C_2_HP_b, C_3_HP_b,
                          C_1_PFC_a, C_2_PFC_a, C_3_PFC_a, C_1_PFC_b, C_2_PFC_b, C_3_PFC_b,
                          c_1, time_ind):
    '''Shared inner computation for permute_policy_A_B: given policy coefficients for HP and PFC
    on A and B choices across three tasks, compute cross-task correlation slices and return the
    PFC - CA1 difference for A (index 0) and B (index 1) choices.'''
    init_t = time_ind[0]; ch_t = time_ind[1]; r_t = time_ind[2]
    C_1_HP = [C_1_HP_a, C_1_HP_b]; C_2_HP = [C_2_HP_a, C_2_HP_b]; C_3_HP = [C_3_HP_a, C_3_HP_b]
    C_1_PFC = [C_1_PFC_a, C_1_PFC_b]; C_2_PFC = [C_2_PFC_a, C_2_PFC_b]; C_3_PFC = [C_3_PFC_a, C_3_PFC_b]
    task_mean_A_B = []
    for idx in range(2):  # 0 = A choices, 1 = B choices
        HP_1 = C_1_HP[idx][c_1,:]; PFC_1 = C_1_PFC[idx][c_1,:]
        HP_2 = C_2_HP[idx][c_1,:]; PFC_2 = C_2_PFC[idx][c_1,:]
        HP_3 = C_3_HP[idx][c_1,:]; PFC_3 = C_3_PFC[idx][c_1,:]
        HP_1_init = C_1_HP[idx][c_1,:,init_t]; PFC_1_init = C_1_PFC[idx][c_1,:,init_t]
        HP_2_init = C_2_HP[idx][c_1,:,init_t]; PFC_2_init = C_2_PFC[idx][c_1,:,init_t]
        HP_3_init = C_3_HP[idx][c_1,:,init_t]; PFC_3_init = C_3_PFC[idx][c_1,:,init_t]
        HP_1_ch = C_1_HP[idx][c_1,:,ch_t]; PFC_1_ch = C_1_PFC[idx][c_1,:,ch_t]
        HP_2_ch = C_2_HP[idx][c_1,:,ch_t]; PFC_2_ch = C_2_PFC[idx][c_1,:,ch_t]
        HP_3_ch = C_3_HP[idx][c_1,:,ch_t]; PFC_3_ch = C_3_PFC[idx][c_1,:,ch_t]
        HP_1_rew = C_1_HP[idx][c_1,:,r_t]; PFC_1_rew = C_1_PFC[idx][c_1,:,r_t]
        HP_2_rew = C_2_HP[idx][c_1,:,r_t]; PFC_2_rew = C_2_PFC[idx][c_1,:,r_t]
        HP_3_rew = C_3_HP[idx][c_1,:,r_t]; PFC_3_rew = C_3_PFC[idx][c_1,:,r_t]

        policy_init_ch_rew = [HP_1_init, HP_2_init, HP_3_init,
                              HP_1_ch, HP_2_ch, HP_3_ch,
                              HP_1_rew, HP_2_rew, HP_3_rew,
                              PFC_1_init, PFC_2_init, PFC_3_init,
                              PFC_1_ch, PFC_2_ch, PFC_3_ch,
                              PFC_1_rew, PFC_2_rew, PFC_3_rew]
        policy_all_time = [HP_2, HP_3, HP_1,
                           HP_2, HP_3, HP_1,
                           HP_2, HP_3, HP_1,
                           PFC_2, PFC_3, PFC_1,
                           PFC_2, PFC_3, PFC_1,
                           PFC_2, PFC_3, PFC_1]

        # vectorized correlation: replaces the per-timepoint np.corrcoef loop
        coefs_tasks = [_batch_corrcoef(policy_all_time[t], policy_init_ch_rew[t])
                       for t in range(len(policy_all_time))]
        task_mean_A_B.append([np.mean(coefs_tasks[:3], 0), np.mean(coefs_tasks[3:6], 0),
                               np.mean(coefs_tasks[6:9], 0), np.mean(coefs_tasks[9:12], 0),
                               np.mean(coefs_tasks[12:15], 0), np.mean(coefs_tasks[15:18], 0)])

    return np.asarray(task_mean_A_B)[:,3:] - np.asarray(task_mean_A_B)[:,:3]


def _one_perm_policy_A_B(all_subjects, all_subjects_firing, n_pfc_sessions, n, c_1, time_ind):
    '''Single permutation iteration for permute_policy_A_B (session-shuffle branch).'''
    sessions_n = np.arange(len(all_subjects))
    np.random.shuffle(sessions_n)
    indices_PFC = sessions_n[:n_pfc_sessions]; indices_HP = sessions_n[n_pfc_sessions:]
    HP_shuffle = [all_subjects[indices_HP], all_subjects_firing[indices_HP]]
    PFC_shuffle = [all_subjects[indices_PFC], all_subjects_firing[indices_PFC]]
    C_1_HP_a, C_2_HP_a, C_3_HP_a, C_1_HP_b, C_2_HP_b, C_3_HP_b = policy_A_B(HP_shuffle, n=n, perm=True)
    C_1_PFC_a, C_2_PFC_a, C_3_PFC_a, C_1_PFC_b, C_2_PFC_b, C_3_PFC_b = policy_A_B(PFC_shuffle, n=n, perm=True)
    return _compute_perm_result(C_1_HP_a, C_2_HP_a, C_3_HP_a, C_1_HP_b, C_2_HP_b, C_3_HP_b,
                                 C_1_PFC_a, C_2_PFC_a, C_3_PFC_a, C_1_PFC_b, C_2_PFC_b, C_3_PFC_b,
                                 c_1, time_ind)


def permute_policy_A_B(HP, PFC, c_1 = 1, n = 11, perm_n = 500, dlc = False, animal_perm = False):
    ''' This function permutes sessions between HP and PFC groups and finds correlation slices through time
    of coefficients at initiation, choice and outcome times.
    c_1 is an argument to select policy coefficients (1)
    n is the number of trials in the past that the history of choices is used for
    policy calculations; default is 11 based on significance testing after training
    perm_n is the number of permutation to run.'''
    # Use same event anchors as the real data computation
    time_ind = TIME_INDICES if not dlc else TIME_INDICES_DLC

    all_subjects = np.hstack([PFC['DM'][0], HP['DM'][0]])
    all_subjects_firing = np.hstack([PFC['Data'][0], HP['Data'][0]])
    n_pfc_sessions = PFC['DM'][0].shape[0]

    if animal_perm == False:
        PFC_HP_perm = [_one_perm_policy_A_B(all_subjects, all_subjects_firing, n_pfc_sessions, n, c_1, time_ind)
                       for _ in trange(perm_n, desc='Policy corr perms', leave=False)]
    else:
        animals_PFC = [0,1,2,3]; animals_HP = [4,5,6]
        m, n_animals = len(animals_PFC), len(animals_HP)
        all_subjects_DM, all_subjects_fr = animal_exp_permute(HP, PFC)
        PFC_HP_perm = []
        for indices_PFC in combinations(range(m + n_animals), m):
            indices_HP = [i for i in range(m + n_animals) if i not in indices_PFC]
            DM_PFC_perm = np.concatenate(np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_PFC)], 0)
            firing_PFC_perm = np.concatenate(np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_PFC)], 0)
            DM_HP_perm = np.concatenate(np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_HP)], 0)
            firing_HP_perm = np.concatenate(np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_HP)], 0)
            HP_shuffle = [DM_HP_perm, firing_HP_perm]; PFC_shuffle = [DM_PFC_perm, firing_PFC_perm]
            C_1_HP_a, C_2_HP_a, C_3_HP_a, C_1_HP_b, C_2_HP_b, C_3_HP_b = policy_A_B(HP_shuffle, n=n, perm=True)
            C_1_PFC_a, C_2_PFC_a, C_3_PFC_a, C_1_PFC_b, C_2_PFC_b, C_3_PFC_b = policy_A_B(PFC_shuffle, n=n, perm=True)
            PFC_HP_perm.append(_compute_perm_result(C_1_HP_a, C_2_HP_a, C_3_HP_a, C_1_HP_b, C_2_HP_b, C_3_HP_b,
                                                    C_1_PFC_a, C_2_PFC_a, C_3_PFC_a, C_1_PFC_b, C_2_PFC_b, C_3_PFC_b,
                                                    c_1, time_ind))
    return PFC_HP_perm

def plot_correlations_slice(PFC, HP, n = 11, c_1 = 1, perm_n = 2, dlc = False, animal_perm = False):
    ''' This function calculates and plots correlation slices through time of coefficients at initiation, 
    choice and outcome times in PFC and CA1 data  and compares it to permuted differences.
    c_1 is an argument to select policy coefficients (1)
    n is the number of trials in the past that the history of choices is used for
    policy calculations; default is 11 based on significance testing after training
    perm_n is the number of permutation to run 
    '''
    if not dlc:
        time_ind = TIME_INDICES; _xticks = XTICK_POSITIONS
    else:
        time_ind = TIME_INDICES_DLC; _xticks = XTICK_POSITIONS_DLC
        
    init_t = time_ind[0]; ch_t = time_ind[1]; r_t = time_ind[2] # initiation, choice, outcome
    # extract policy coefficients on A and B choices in PFC and CA1
    C_1_HP_a, C_2_HP_a, C_3_HP_a, C_1_HP_b, C_2_HP_b, C_3_HP_b = policy_A_B(HP, n = n, perm = False)
    C_1_PFC_a, C_2_PFC_a, C_3_PFC_a, C_1_PFC_b, C_2_PFC_b, C_3_PFC_b = policy_A_B(PFC, n = n, perm = False)
    
    C_1_HP = [C_1_HP_a, C_1_HP_b]; C_2_HP = [C_2_HP_a, C_2_HP_b]; C_3_HP = [C_3_HP_a, C_3_HP_b]
    C_1_PFC = [C_1_PFC_a, C_1_PFC_b]; C_2_PFC = [C_2_PFC_a, C_2_PFC_b]; C_3_PFC = [C_3_PFC_a, C_3_PFC_b]
    task_mean_A_B  = []
    for i,ii in enumerate(C_1_HP): # loop through either A or B choices
        # select policy coefficients on A/B in each task across all time points
        HP_1 = C_1_HP[i][c_1,:]; PFC_1  = C_1_PFC[i][c_1,:] 
        HP_2 = C_2_HP[i][c_1,:]; PFC_2 = C_2_PFC[i][c_1,:]   
        HP_3 = C_3_HP[i][c_1,:]; PFC_3 = C_3_PFC[i][c_1,:]
        # select policy coefficients on A/B in each task at initiation time
        HP_1_init = C_1_HP[i][c_1,:,init_t]; PFC_1_init  = C_1_PFC[i][c_1,:,init_t]
        HP_2_init = C_2_HP[i][c_1,:,init_t]; PFC_2_init= C_2_PFC[i][c_1,:,init_t]   
        HP_3_init = C_3_HP[i][c_1,:,init_t]; PFC_3_init = C_3_PFC[i][c_1,:,init_t]
        # select policy coefficients on A/B in each task at choice time
        HP_1_ch = C_1_HP[i][c_1,:,ch_t]; PFC_1_ch  = C_1_PFC[i][c_1,:,ch_t]     
        HP_2_ch = C_2_HP[i][c_1,:,ch_t]; PFC_2_ch = C_2_PFC[i][c_1,:,ch_t]  
        HP_3_ch = C_3_HP[i][c_1,:,ch_t]; PFC_3_ch = C_3_PFC[i][c_1,:,ch_t]
        # select policy coefficients on A/B in each task at outcome time
        HP_1_rew = C_1_HP[i][c_1,:,r_t]; PFC_1_rew  = C_1_PFC[i][c_1,:,r_t]
        HP_2_rew = C_2_HP[i][c_1,:,r_t]; PFC_2_rew = C_2_PFC[i][c_1,:,r_t]
        HP_3_rew = C_3_HP[i][c_1,:,r_t]; PFC_3_rew = C_3_PFC[i][c_1,:,r_t]

        # store policy at init, choice, outcome for each task both PFC and CA1 for later cross correlations
        policy_init_ch_rew = [HP_1_init, HP_2_init, HP_3_init,\
                              HP_1_ch, HP_2_ch, HP_3_ch,\
                              HP_1_rew, HP_2_rew, HP_3_rew,\
                              PFC_1_init, PFC_2_init, PFC_3_init,\
                              PFC_1_ch, PFC_2_ch, PFC_3_ch,\
                              PFC_1_rew, PFC_2_rew, PFC_3_rew] 
                        
        # store policy at all time points for each task both PFC and CA1 for later cross correlations
        policy_all_time = [HP_2, HP_3, HP_1,\
                           HP_2, HP_3, HP_1,\
                           HP_2, HP_3, HP_1,\
                           PFC_2, PFC_3, PFC_1,\
                           PFC_2, PFC_3, PFC_1,\
                           PFC_2,PFC_3, PFC_1]
            
        coefs_tasks = [] # to store correlations in task 1 vs 2; task 2 vs; task 3 vs 1 HP and PFC
        for t in range(len(policy_all_time)): # compute correlations in different task combinations
            coefs_tasks.append(_batch_corrcoef(policy_all_time[t], policy_init_ch_rew[t]))
        # store mean corr coef across 3 tasks for init, choice and outcome HP and PFC (# CA1 init, CA1 ch, CA1 rew, PFC init, PFC ch, PFC rew)
        task_mean_A_B.append([np.mean(coefs_tasks[:3],0),np.mean(coefs_tasks[3:6],0), np.mean(coefs_tasks[6:9],0),np.mean(coefs_tasks[9:12],0),np.mean(coefs_tasks[12:15],0), np.mean(coefs_tasks[15:18],0)])
    
    # find permuted differences
    PFC_HP_perm = np.asarray(permute_policy_A_B(HP, PFC, c_1 = c_1, n = n, perm_n = perm_n, dlc = dlc, animal_perm = animal_perm))
    task_mean_A_B = np.asarray(task_mean_A_B)

    # animal permutations: only 35 splits, so use exact rank-based thresholds (P < .01 cannot be reached)
    if animal_perm:
        _thr = exact_perm_threshold
    else:
        _thr = lambda null, alpha: np.percentile(null, 100*(1 - alpha))

    # Max-statistic correction for A and B separately: max across anchors and time per permutation, then percentile
    # PFC_HP_perm shape: (perm_n, 2, 3, n_timepoints) — 2=A/B, 3=init/ch/rew
    n_p = len(PFC_HP_perm)
    max_a_per_perm = np.max(PFC_HP_perm[:, 0].reshape(n_p, -1), axis=1)  # (perm_n,)
    max_b_per_perm = np.max(PFC_HP_perm[:, 1].reshape(n_p, -1), axis=1)
    _95th_a = _thr(max_a_per_perm, 0.05); _99th_a = _thr(max_a_per_perm, 0.01)
    _95th_b = _thr(max_b_per_perm, 0.05); _99th_b = _thr(max_b_per_perm, 0.01)

    # real A/B difference between CA1 and PFC on A and B trials
    a_b_real = task_mean_A_B[:,3:,:] - task_mean_A_B[:,:3,:] # PFC - CA1
    significant_diff_a_05 = np.where(a_b_real[0] > _95th_a)
    significant_diff_b_05 = np.where(a_b_real[1] > _95th_b)
    significant_diff_a_01 = np.where(a_b_real[0] > _99th_a)
    significant_diff_b_01 = np.where(a_b_real[1] > _99th_b)

    fig, axs = plt.subplots(1,3, figsize = (15, 3)); isl = [wes.Royal2_5.mpl_colors[0],wes.Royal2_5.mpl_colors[3]];
    titles_str = ['Initiation','Choice', 'Outcome'] # list of titles
    HP_corr_plot = [task_mean_A_B[0][:3], task_mean_A_B[1][:3]]; PFC_corr_plot = [task_mean_A_B[0][3:], task_mean_A_B[1][3:]]
    regions_label = [['CA1 A', 'CA1 B'],['PFC A', 'PFC B']];
    ymax = np.max([np.max(np.max(HP_corr_plot,0),0),np.max(np.max(PFC_corr_plot,0),0)])+0.05
    linestyles = ['solid', 'dashed']
    for region, correlation in enumerate([HP_corr_plot, PFC_corr_plot]): # loop through A and B
        for ab, ab_corr in enumerate(correlation):
            for per,period in enumerate(ab_corr):
                axs[per].plot(ab_corr[per], color = isl[region], label = regions_label[region][ab], linestyle = linestyles[ab] ) # plot correlations
                axs[per].set_title(titles_str[per])
                if region == 1 and ab == 1:
                    # A: P < .05 (small dots) and P < .01 (large dots)
                    p_a_05 = significant_diff_a_05[1][np.where(significant_diff_a_05[0] == per)[0]]
                    p_a_01 = significant_diff_a_01[1][np.where(significant_diff_a_01[0] == per)[0]]
                    axs[per].plot(p_a_05, np.ones(p_a_05.shape)*ymax + 0.02, '.', markersize = 2, color= 'green', label = 'A PFC vs CA1 < .05' if animal_perm else None)
                    axs[per].plot(p_a_01, np.ones(p_a_01.shape)*ymax + 0.02, '.', markersize = 4, color= 'green', label = None if animal_perm else 'A PFC vs CA1 < .01')
                    # B: P < .05 (small dots) and P < .01 (large dots)
                    p_b_05 = significant_diff_b_05[1][np.where(significant_diff_b_05[0] == per)[0]]
                    p_b_01 = significant_diff_b_01[1][np.where(significant_diff_b_01[0] == per)[0]]
                    axs[per].plot(p_b_05, np.ones(p_b_05.shape)*ymax + 0.05, '.', markersize = 2, color = 'pink', label = 'B PFC vs CA1 < .05' if animal_perm else None)
                    axs[per].plot(p_b_01, np.ones(p_b_01.shape)*ymax + 0.05, '.', markersize = 4, color = 'pink', label = None if animal_perm else 'B PFC vs CA1 < .01')
                    axs[per].set_xticks(_xticks)
                    axs[per].set_ylim(-0.15, ymax+0.1)
                    axs[per].set_xticklabels(XTICK_LABELS)
                    axs[per].set_xlabel('Time in Trial (s)')
                    axs[per].set_ylabel('Correlation Coefficient')

    axs[per].legend()
    sns.despine()


def correlations_ab(d, n = 11, perm = False, c_1 = 1, dlc = False):
    ''' This function finds the diagonal from the correlation matrices of policy regression coefficients
    between different tasks on A choices and on B choices 
    c_1 is an argument to select policy coefficients (1)
    n is the number of trials in the past that the history of choices is used for
    policy calculations; default is 11 based on significance testing after training
    perm is used to access the data depending on whether it's a permutation test '''
    if dlc == False:
        index = 63
    else:
        index = 64
    C_1_a, C_2_a, C_3_a, C_1_b, C_2_b, C_3_b = policy_A_B(d, n = n, perm = perm)

    mean_value_a = np.mean([np.corrcoef(C_1_a[c_1].T,C_2_a[c_1].T), np.corrcoef(C_1_a[c_1].T,C_3_a[c_1].T), np.corrcoef(C_2_a[c_1].T,C_3_a[c_1].T)],0)
    mean_value_b = np.mean([np.corrcoef(C_1_b[c_1].T,C_2_b[c_1].T), np.corrcoef(C_1_b[c_1].T,C_3_b[c_1].T), np.corrcoef(C_2_b[c_1].T,C_3_b[c_1].T)],0)
    
    diag_a = np.sum(np.diagonal(mean_value_a[index:,:index])); diag_b = np.sum(np.diagonal(mean_value_b[index:,:index]))   
    return mean_value_a, mean_value_b, diag_a, diag_b

def _one_perm_diagonal(all_subjects, all_subjects_firing, n_pfc_sessions, n, c_1, dlc):
    '''Single permutation iteration for permute_sessions_diagonal (session-shuffle branch).'''
    sessions_n = np.arange(len(all_subjects))
    np.random.shuffle(sessions_n)
    indices_PFC = sessions_n[:n_pfc_sessions]; indices_HP = sessions_n[n_pfc_sessions:]
    HP_shuffle = [all_subjects[indices_HP], all_subjects_firing[indices_HP]]
    PFC_shuffle = [all_subjects[indices_PFC], all_subjects_firing[indices_PFC]]
    _all_diff_a = []; _all_diff_b = []
    for d in [HP_shuffle, PFC_shuffle]:
        _, _, diag_a, diag_b = correlations_ab(d, n=n, perm=True, c_1=c_1, dlc=dlc)
        _all_diff_a.append(diag_a); _all_diff_b.append(diag_b)
    return _all_diff_a, _all_diff_b


def permute_sessions_diagonal(HP, PFC, c_1 = 1, n = 11, perm_n = 2, dlc = False, animal_perm = False):
    ''' This function permutes sessions between HP and PFC groups and finds
     the diagonal from the correlation matrices of policy regression coefficients
     between different tasks on A choices and on B choices
     c_1 is an argument to select policy coefficients (1)
     perm_n is the number of permutation to run '''
    if animal_perm == False:
        all_subjects = np.hstack([PFC['DM'][0], HP['DM'][0]])
        all_subjects_firing = np.hstack([PFC['Data'][0], HP['Data'][0]])
        n_pfc_sessions = PFC['DM'][0].shape[0]
        results = [_one_perm_diagonal(all_subjects, all_subjects_firing, n_pfc_sessions, n, c_1, dlc)
                   for _ in trange(perm_n, desc='Diagonal perms', leave=False)]
        all_shuffle_a = [r[0] for r in results]
        all_shuffle_b = [r[1] for r in results]
    else:
        animals_PFC = [0,1,2,3]; animals_HP = [4,5,6]
        m, n_animals = len(animals_PFC), len(animals_HP)
        all_subjects_DM, all_subjects_fr = animal_exp_permute(HP, PFC)
        all_shuffle_a = []; all_shuffle_b = []
        for indices_PFC in combinations(range(m + n_animals), m):
            indices_HP = [i for i in range(m + n_animals) if i not in indices_PFC]
            DM_PFC_perm = np.concatenate(np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_PFC)], 0)
            firing_PFC_perm = np.concatenate(np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_PFC)], 0)
            DM_HP_perm = np.concatenate(np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_HP)], 0)
            firing_HP_perm = np.concatenate(np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_HP)], 0)
            HP_shuffle = [DM_HP_perm, firing_HP_perm]; PFC_shuffle = [DM_PFC_perm, firing_PFC_perm]
            _all_diff_a = []; _all_diff_b = []
            for d in [HP_shuffle, PFC_shuffle]:
                _, _, diag_a, diag_b = correlations_ab(d, n=n, perm=True, c_1=c_1, dlc=dlc)
                _all_diff_a.append(diag_a); _all_diff_b.append(diag_b)
            all_shuffle_a.append(_all_diff_a); all_shuffle_b.append(_all_diff_b)

    diff_all_a = [all_shuffle_a[i][1] - all_shuffle_a[i][0] for i in range(len(all_shuffle_a))]
    diff_all_b = [all_shuffle_b[i][1] - all_shuffle_b[i][0] for i in range(len(all_shuffle_b))]
    if animal_perm: # only 35 animal splits: exact rank-based threshold
        _all_95_a = exact_perm_threshold(diff_all_a, 0.05)
        _all_95_b = exact_perm_threshold(diff_all_b, 0.05)
    else:
        _all_95_a = np.percentile(diff_all_a, 95)
        _all_95_b = np.percentile(diff_all_b, 95)
    return diff_all_a, diff_all_b, _all_95_a, _all_95_b
    
 
def plot_diagonal_sums(HP, PFC,c_1 = 1, perm_n = 2, dlc = False, animal_perm = False):
    if dlc == False:
        index = 63; _xticks = [0,12, 24, 35, 42, 49, 62]
    else:
        index = 64; _xticks = [0,12.5, 25, 38, 45, 51, 63]
    mean_value_a_HP, mean_value_b_HP, diag_a_HP ,diag_b_HP =  correlations_ab(HP, n = 11, perm = False, c_1 = 1, dlc = dlc)
    mean_value_a_PFC, mean_value_b_PFC, diag_a_PFC,diag_b_PFC = correlations_ab(PFC, n = 11, perm = False, c_1 = 1, dlc = dlc)
    hp = np.hstack((mean_value_a_HP[index:,:index], mean_value_b_HP[index:,:index]))
    pfc = np.hstack((mean_value_a_PFC[index:,:index], mean_value_b_PFC[index:,:index]))
    hp_pfc = np.vstack((hp,pfc))
    cmin = np.min(hp_pfc); cmax = np.max(hp_pfc)

    plt.figure(figsize = (20,5))
    cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap; 
    str_title = ['CA1 policy on A choice', 'CA1 policy on B choice', 'PFC policy on A choice', 'PFC policy on B choice']
    for n, corr in enumerate([mean_value_a_HP, mean_value_b_HP,mean_value_a_PFC,mean_value_b_PFC]):
        plt.subplot(1,4,n+1)
        plt.imshow(corr[index:,:index], cmap = cmap, vmin = cmin, vmax = cmax)
        plt.xticks(_xticks, XTICK_LABELS)
        plt.yticks(_xticks, XTICK_LABELS)
        plt.title(str_title[n])
        plt.colorbar(label = 'Correlation Coefficient')
    plt.tight_layout()


    all_diff_real_a = diag_a_PFC - diag_a_HP; all_diff_real_b = diag_b_PFC - diag_b_HP
    diff_all_a,diff_all_b, _all_95_a, _all_95_b = permute_sessions_diagonal(HP, PFC, n = 11, c_1 = 1, perm_n = perm_n, dlc = dlc, animal_perm = animal_perm)
    
    plt.figure(figsize = (10,3))
    diffs = [diff_all_a,diff_all_b]; real = [all_diff_real_a,all_diff_real_b]; perm_diff = [_all_95_a,_all_95_b]
    str_title = ['Policy on A difference between CA1 and PFC', 'Policy on B difference between CA1 and PFC']
    for d, diff in enumerate(diffs):
        plt.subplot(1,2,d+1)
        plt.hist(diff, color = 'grey', alpha = 0.5)
        plt.vlines(real[d],ymin = 0, ymax = max(np.histogram(diff)[0]),  color = 'pink', label = 'Difference in the data')
        plt.vlines(perm_diff[d], ymin = 0, ymax = max(np.histogram(diff)[0]), color = 'black', label = 'permutation threshold (P < .05)')
        plt.xlabel('Permuted differences'); plt.ylabel('Count')
        plt.legend(fontsize = 7)
        plt.title(str_title[d])
        sns.despine()
    plt.tight_layout()


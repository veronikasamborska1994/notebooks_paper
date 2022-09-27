import scipy
import palettable
import numpy as np
import pylab as plt
import seaborn as sns
from palettable import wesanderson as wes
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from itertools import combinations 
from functions.helper_functions import  task_ind
from functions.helper_functions import _CPD


def animal_exp_permute(all_sessions_HP,all_sessions_PFC):
   
    HP_1 = 15; HP_2 = 7 # CA1 sessions
    PFC_1 = 9; PFC_2 = 16; PFC_3 = 13 # PFC sessions
    
    # trial info data for each CA1 and PFC animal
    HP_all_sessions_1 = all_sessions_HP['DM'][0][:HP_1]; HP_all_sessions_2 = all_sessions_HP['DM'][0][HP_1:HP_1+HP_2]
    HP_all_sessions_3 = all_sessions_HP['DM'][0][HP_1+HP_2:]

    PFC_all_sessions_1 = all_sessions_PFC['DM'][0][:PFC_1]; PFC_all_sessions_2 = all_sessions_PFC['DM'][0][PFC_1:PFC_1+PFC_2]
    PFC_all_sessions_3 = all_sessions_PFC['DM'][0][PFC_1+PFC_2:PFC_1+PFC_2+PFC_3]; PFC_all_sessions_4 = all_sessions_PFC['DM'][0][PFC_1+PFC_2+PFC_3:]
    
    # firing rate data for each CA1 and PFC animal
    HP_all_sessions_1_fr = all_sessions_HP['Data'][0][:HP_1]; HP_all_sessions_2_fr = all_sessions_HP['Data'][0][HP_1:HP_1+HP_2]
    HP_all_sessions_3_fr = all_sessions_HP['Data'][0][HP_1+HP_2:]

    PFC_all_sessions_1_fr = all_sessions_PFC['Data'][0][:PFC_1]; PFC_all_sessions_2_fr = all_sessions_PFC['Data'][0][PFC_1:PFC_1+PFC_2]
    PFC_all_sessions_3_fr = all_sessions_PFC['Data'][0][PFC_1+PFC_2:PFC_1+PFC_2+PFC_3];   PFC_all_sessions_4_fr = all_sessions_PFC['Data'][0][PFC_1+PFC_2+PFC_3:]
   
    all_subjects_DM = [HP_all_sessions_1,HP_all_sessions_2, HP_all_sessions_3,PFC_all_sessions_1,PFC_all_sessions_2, PFC_all_sessions_3,PFC_all_sessions_4]
    all_subjects_fr = [HP_all_sessions_1_fr,HP_all_sessions_2_fr, HP_all_sessions_3_fr,PFC_all_sessions_1_fr,PFC_all_sessions_2_fr, PFC_all_sessions_3_fr,PFC_all_sessions_4_fr]

    return  all_subjects_DM ,all_subjects_fr


def plot_policy_during_recordings(PFC, HP, n = 12): 
    ''''Coefficients from a logistic regression predicting current choices using the history of previous choices (I),
    outcomes (not shown) and choice-outcome interactions, Figure 5A '''
    
    # sessions to split by animals
    HP_1 = 15; HP_2 = 7  
    PFC_1 = 9; PFC_2 = 16; PFC_3 = 13
    HP_1_sess = HP['DM'][0][:HP_1]; HP_2_sess = HP['DM'][0][HP_1:HP_1+HP_2]
    HP_3_sess = HP['DM'][0][HP_1+HP_2:]
    PFC_1_sess = PFC['DM'][0][:PFC_1]; PFC_2_sess = PFC['DM'][0][PFC_1:PFC_1+PFC_2]
    PFC_3_sess = PFC['DM'][0][PFC_1+PFC_2:PFC_1+PFC_2+PFC_3];  PFC_4_sess = PFC['DM'][0][PFC_1+PFC_2+PFC_3:]
    d = [HP_1_sess, HP_2_sess, HP_3_sess, PFC_1_sess, PFC_2_sess, PFC_3_sess, PFC_4_sess]
     
    results_array_subj = []; 
    for dm in d:
        results_array = []
        for  s, sess in enumerate(dm):        
            DM = dm[s]; choices = DM[:,1]; reward = DM[:,2]
            previous_rewards = scipy.linalg.toeplitz(reward, np.zeros((1,n)))[n-1:-1] # find rewards on n-back trials
            previous_choices = scipy.linalg.toeplitz(choices, np.zeros((1,n)))[n-1:-1] # find choices on n-back trials
            interactions = scipy.linalg.toeplitz((((choices-0.5)*(reward-0.5))*2),np.zeros((1,n)))[n-1:-1] #interactions rewards x choice
            choices_current = (choices[n:]) # current choices need to start at nth trial
            ones = np.ones(len(interactions)).reshape(len(interactions),1) # add constant
            X = (np.hstack([previous_rewards,previous_choices,interactions,ones])) # create design matrix
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
    plt.fill_between(np.arange(len(average_coefs))[n:n*2], average_coefs[n:n*2]+std[n*2:-1], average_coefs[n:n*2]- std[n:n*2],alpha = 0.2, color =c[2])
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
    for  s, sess in enumerate(dm):
        DM = dm[s]; firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        choices = DM[:,1]; reward = DM[:,2];  choices_current = choices-0.5 
    
        previous_rewards = scipy.linalg.toeplitz(reward, np.zeros((1,n)))[n-1:-1] # find rewards on n-back trials 
        previous_choices = scipy.linalg.toeplitz(0.5-choices, np.zeros((1,n)))[n-1:-1] # find choices on n-back trials       
        interactions = scipy.linalg.toeplitz((((0.5-choices)*(reward-0.5))*2),np.zeros((1,n)))[n-1:-1] # interactions rewards x choice
        ones = np.ones(len(interactions)).reshape(len(interactions),1) # constant
        X = np.hstack([previous_rewards,previous_choices,interactions,ones]) # design matrix for calculating policy
        policy = np.matmul(X, policy_beh) # calculate policy on each trial
        
        choices_current = choices_current[n:];  reward = reward[n:];  firing_rates = firing_rates[n:] # current choices need to start at nth trial
        policy_choice = choices_current*policy ; rew_ch = choices_current*reward
                
        _1_back = choices_current[:-1]; choices_current = choices_current[1:]; reward = reward[1:];   policy = policy[1:]
        policy_choice = policy_choice[1:]; rew_ch = rew_ch[1:]; ones = np.ones(len(rew_ch)); firing_rates = firing_rates[1:]
    
        X = np.vstack([choices_current,_1_back,reward,policy, policy_choice,rew_ch, ones]).T # design matrix for cpds           
        n_predictors = X.shape[1]
        y = firing_rates.reshape([len(firing_rates),-1]) # activity matrix [n_trials, n_neurons*n_timepoints]
        cpd.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors)) # calculate coefficients of partial determination
        
        for i in range(perm):  # randomly rolls firing rates with respect to trials
            y_perm = np.roll(y,np.random.randint(n, n_trials), axis = 0)
            cpd_perm[i].append(_CPD(X,y_perm).reshape(n_neurons, n_timepoints, n_predictors))
    
    cpd_perm   = np.stack([np.concatenate(cpd_i,0) for cpd_i in cpd_perm],0)
    cpd = np.concatenate(cpd,0)
    return cpd, cpd_perm


def plot_policy_simple_GLM(HP,PFC, n_perm = 1000, n = 11, dlc = False): 
    ''' Plotting CPDs and significance levels, Figure 5B)''' 
    cpd_HP, cpd_perm_HP =  GLM_policy_roll_tome(HP, perm = n_perm, n = n)
    cpd_PFC, cpd_perm_PFC =  GLM_policy_roll_tome(PFC, perm =  n_perm, n = n)
    cpd_HP_m = np.nanmean(cpd_HP,0); cpd_PFC_m = np.nanmean(cpd_PFC,0)
    if  dlc == False:
        _xticks = [0,12, 24, 35, 42, 49, 63]
    else:
        _xticks = [0,12.5, 25, 38, 45, 51, 63]
    # lists of cpds and permuted cpds for plotting
    cpds = [cpd_PFC_m[:,:-1],cpd_HP_m[:,:-1]]; cpds_perms = [cpd_perm_PFC[:,:,:,:-1], cpd_perm_HP[:,:,:,:-1]]
    # plotting params
    t = np.arange(0,cpd_HP_m.shape[0]); clrs = wes.Moonrise1_5.mpl_colors + wes.Royal2_5.mpl_colors[1:2:]
    plt.figure(figsize = (15,6))
    titles = ['PFC', 'CA1']; p = ['Choice','Prev Choice','Reward', 'Policy', 'Policy x Choice', 'Reward x Choice']
    # plot cpds and significance levels
    for c,cpd in enumerate(cpds): 
        values_95 = np.max(np.percentile(np.mean(cpds_perms[c],1),95,0),0) # indicate where significant at 0.05 for each region (loop) 
        values_99 = np.max(np.percentile(np.mean(cpds_perms[c],1),99,0),0) # indicate where significant at 0.001 for each region (loop)
        cpd_max = np.max(cpd); plt.subplot(1,2,c+1); plt.title(titles[c])
        array_pvals = np.ones((cpd.shape[0], cpd.shape[1])) # store where significant at .05 or .001
        for pred in range(cpd.shape[1]):  # loop through pred
            array_pvals[(np.where(cpd[:,pred] > values_95[pred])[0]),pred] = 0.05
            array_pvals[(np.where(cpd[:,pred] > values_99[pred])[0]),pred] = 0.001  
       
        for pred in np.arange(cpd.shape[1]):
            plt.plot(cpd[:,pred]*100, color = clrs[pred], label = p[pred])
            ymax = cpd_max*(1.2+0.02*pred)*100
            p_vals = array_pvals[:,pred]; t05 = t[p_vals == 0.05]; t001 = t[p_vals == 0.001]
            plt.plot(t05, np.ones(t05.shape)*ymax, '.', markersize = 2, color = clrs[pred])
            plt.plot(t001, np.ones(t001.shape)*ymax, '.', markersize = 4, color = clrs[pred])
            plt.xticks(_xticks, ['-1','-0.5','Init', 'Ch','R', '+0.5', '+1']) 
        plt.xlabel('Time in Trial (s)')
        plt.ylabel('CPD (%)')
    plt.legend(); sns.despine()
    

def policy_for_GLM(data, n, perm = False):
    ''''Coefficients from a logistic regression predicting current choices using the history of previous choices (I),
    outcomes (not shown) and choice-outcome interactions to use for policy calculations for PFC and CA1 animals in GLM on Figure 5A '''
    if perm:
        dm = data[0]
    else:
        dm = data['DM'][0]
    results_array = []
    for  s, sess in enumerate(dm):     
        DM = dm[s]; choices = DM[:,1]; reward = DM[:,2];
        previous_rewards = scipy.linalg.toeplitz(reward, np.zeros((1,n)))[n-1:-1] # find rewards on n-back trials
        previous_choices = scipy.linalg.toeplitz(choices, np.zeros((1,n)))[n-1:-1] # find choices on n-back trials
        interactions = scipy.linalg.toeplitz((((choices-0.5)*(reward-0.5))*2),np.zeros((1,n)))[n-1:-1] # interactions rewards x choice
        choices_current = (choices[n:]) # current choices need to start at nth trial
        ones = np.ones(len(interactions)).reshape(len(interactions),1) # add constant
        X = (np.hstack([previous_rewards,previous_choices,interactions,ones])) # create design matrix
        model = LogisticRegression(fit_intercept = False) 
        results = model.fit(X,choices_current)# fit logistic regression predicting current choice based on history of behaviour
        results_array.append(results.coef_[0]) # store coefficients
         
    average = np.mean(results_array,0)
    return average

def policy_A_B(data, n = 10, plot_a = False, perm = True):
    '''GLM predicting activity of each neuron at each time point across the trial, 
    as a function of policy and reward on A and B trials independently.
    plot_a = True for A choices; plot_a = False for B choices, perm = True if it is a permutation test'''
    
    if perm: # extact data if it's a permutation test (different format)
        dm = data[0]; firing = data[1]
    else:
        dm = data['DM'][0]; firing = data['Data'][0] # extract data

    C_1 = []; C_2 = []; C_3 = [] # to store coefficients from a linear regression for each neuron 
    policy_beh = policy_for_GLM(data, n = n, perm = perm)
    for  s, sess in enumerate(dm):
        DM = dm[s]; firing_rates = firing[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape
        # trial information
        choices = DM[:,1]; reward = DM[:,2];  task =  DM[:,5];  a_pokes = DM[:,6]; b_pokes = DM[:,7]
        taskid = task_ind(task, a_pokes, b_pokes)
        reward_current = reward;  choices_current = choices-0.5
        # make a design matrix for each task and find regression coefficients within each task 
        tasks = [1,2,3]
        for i in tasks: 
            task_index = np.where(taskid == i)[0]
            rewards_task = reward_current[task_index]; choices_task = choices_current[task_index] 
            previous_rewards_task = scipy.linalg.toeplitz(rewards_task, np.zeros((1,n)))[n-1:-1] # find rewards on n-back trials 
            previous_choices_task = scipy.linalg.toeplitz(0.5-choices_task, np.zeros((1,n)))[n-1:-1] # find choices on n-back trials       
            interactions_task = scipy.linalg.toeplitz((((0.5-choices_task)*(rewards_task-0.5))*2),np.zeros((1,n)))[n-1:-1] # interactions rewards x choice
            ones = np.ones(len(interactions_task)).reshape(len(interactions_task),1) # constant
            X = np.hstack([previous_rewards_task, previous_choices_task, interactions_task, ones]) # design matrix for calculating policy
            policy = np.matmul(X, policy_beh) # calculate policy on each trial
            rewards_task = rewards_task[n:];  choices_task = choices_task[n:] # current choices need to start at nth trial
            ones_task = np.ones(len(choices_task)) # constant 
            firing_rates_task = firing_rates[task_index][n:] # firing rates in task
            a = np.where(choices_task == 0.5)[0];   b = np.where(choices_task == -0.5)[0] # indicies for A/B choices
            if plot_a == True: # calculate policy loadings on A choices
                rewards_task = rewards_task[a]; policy = policy[a]; ones_task  = ones_task[a]; firing_rates_task = firing_rates_task[a]
            
            else: # calculate policy loadings on B choices
                rewards_task = rewards_task[b];  policy = policy[b]; ones_task  = ones_task[b]; firing_rates_task = firing_rates_task[b]
            
            X = np.vstack([rewards_task, policy, ones_task]).T
            n_predictors = X.shape[1]
            y = firing_rates_task.reshape([len(firing_rates_task),-1]) # activity matrix [n_trials, n_neurons*n_timepoints]
            ols = LinearRegression(fit_intercept = False)
            ols.fit(X,y)
            
            if i == 1:
                C_1.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # store predictor loadings
            elif i == 2:
                C_2.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # store predictor loadings
            elif i == 3:
                C_3.append(ols.coef_.reshape(n_neurons, n_timepoints, n_predictors)) # store predictor loadings

    C_1 = np.concatenate(C_1,0); C_2 = np.concatenate(C_2,0); C_3 = np.concatenate(C_3,0)
    
    C_2_inf = [~np.isinf(C_2[0]).any(axis=1)]; C_2_nan = [~np.isnan(C_2[0]).any(axis=1)]
    C_3_inf = [~np.isinf(C_3[0]).any(axis=1)];  C_3_nan = [~np.isnan(C_3[0]).any(axis=1)]
    C_1_inf = [~np.isinf(C_1[0]).any(axis=1)];  C_1_nan = [~np.isnan(C_1[0]).any(axis=1)]
    nans = np.asarray(C_1_inf) & np.asarray(C_1_nan) & np.asarray(C_3_inf) & np.asarray(C_3_nan) & np.asarray(C_2_inf)& np.asarray(C_2_nan)
    C_1 = np.transpose(C_1[:,nans[0],:],[2,0,1]); C_2 = np.transpose(C_2[:,nans[0],:],[2,0,1]);  C_3 = np.transpose(C_3[:,nans[0],:],[2,0,1])
    return C_1, C_2, C_3

def perumute_policy_A_B(HP, PFC, c_1 = 1, n = 11, perm_n = 500, dlc = False, animal_perm = False):
    ''' This function permutes sessions between HP and PFC groups and finds correlation slices through time
    of coefficients at initiation, choice and outcome times.
    c_1 is an argument to select policy coefficients (1)
    n is the number of trials in the past that the history of choices is used for
    policy calculations; default is 11 based on significance testing after training
    perm_n is the number of permutation to run.'''
    if dlc == False:
        time_ind = [25, 36, 43];
    else:
        time_ind = [26, 39, 46]; 
    
    init_t = time_ind[0]; ch_t = time_ind[1]; r_t = time_ind[2] # initiation, choice, outcome
    # permute sessions
    PFC_HP_perm = []; diff_PFC_HP_perm = []; 
    all_subjects = np.hstack([PFC['DM'][0], HP['DM'][0]]); all_subjects_firing = np.hstack([PFC['Data'][0], HP['Data'][0]])
    sessions_n = np.arange(len(all_subjects))
    if animal_perm == False:
        for i in range(perm_n):
            np.random.shuffle(sessions_n) # shuffle PFC/HP sessions
            indices_HP = sessions_n[PFC['DM'][0].shape[0]:];  indices_PFC = sessions_n[:PFC['DM'][0].shape[0]]
            PFC_shuffle_dm = all_subjects[np.asarray(indices_PFC)];  HP_shuffle_dm = all_subjects[np.asarray(indices_HP)] 
            PFC_shuffle_f = all_subjects_firing[np.asarray(indices_PFC)];  HP_shuffle_f = all_subjects_firing[np.asarray(indices_HP)]
            HP_shuffle= [HP_shuffle_dm,HP_shuffle_f]; PFC_shuffle= [PFC_shuffle_dm,PFC_shuffle_f]
            # find coefficients for permuted PFC and CA1 data on A trials and B trials 
            C_1_HP_b, C_2_HP_b, C_3_HP_b = policy_A_B(HP_shuffle, n = n, plot_a = False, perm = True)
            C_1_PFC_b ,C_2_PFC_b, C_3_PFC_b = policy_A_B(PFC_shuffle, n = n, plot_a = False, perm = True)
            C_1_HP_a, C_2_HP_a, C_3_HP_a = policy_A_B(HP_shuffle, n = n, plot_a = True, perm = True)
            C_1_PFC_a ,C_2_PFC_a, C_3_PFC_a = policy_A_B(PFC_shuffle, n = n, plot_a = True, perm = True)

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

                coefs_tasks = [] # to store correlations in task 1 vs 2; task 2 vs task 3; task 3 vs 1 in HP and PFC
                for t,pol_t in enumerate(policy_all_time): # compute correlations in different task combinations
                    policy_all_time_task = policy_all_time[t]; policy_init_ch_rew_task = policy_init_ch_rew[t]
                    coef_time = []
                    for pol in policy_all_time_task.T: # find a correlation coefficient for each 
                        coef_time.append(np.corrcoef(pol,policy_init_ch_rew_task)[0][1])
                    coefs_tasks.append(coef_time) # store coefficients
                # store mean corr coef across 3 tasks for init, choice and outcome HP and PFC (# CA1 init, CA1 ch, CA1 rew, PFC init, PFC ch, PFC rew)
                task_mean_A_B.append([np.mean(coefs_tasks[:3],0),np.mean(coefs_tasks[3:6],0), np.mean(coefs_tasks[6:9],0),np.mean(coefs_tasks[9:12],0),np.mean(coefs_tasks[12:15],0), np.mean(coefs_tasks[15:18],0)])
        b_a = np.asarray(task_mean_A_B[0]) - np.asarray(task_mean_A_B[1]) # difference between A and B (CA1 init, CA1 ch, CA1 rew, PFC init, PFC ch, PFC rew)
        # store permuted differences of differences between B and A between CA1 and PFC at initiation, choice and outcome time       
        diff_PFC_HP_perm.append((np.asarray(b_a[3:])- np.asarray(b_a[:3]))) # difference in B - A PFC - CA1
        PFC_HP_perm.append((np.asarray(task_mean_A_B)[:,3:]- np.asarray(task_mean_A_B)[:,:3])) # store difference between PFC and CA1 on A and on B choices
      
    else:
        animals_PFC = [0,1,2,3]; animals_HP = [4,5,6]
        m, n = len(animals_PFC), len(animals_HP)
        all_subjects_DM, all_subjects_fr  = animal_exp_permute(HP,PFC)
        for indices_PFC in combinations(range(m + n), m):
            indices_HP = [i for i in range(m + n) if i not in indices_PFC]
            DM_PFC_perm = np.concatenate(np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_PFC)],0)
            firing_PFC_perm = np.concatenate(np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_PFC)],0)
                
            DM_HP_perm = np.concatenate(np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_HP)],0)
            firing_HP_perm = np.concatenate(np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_HP)],0)
            HP_shuffle = [DM_HP_perm,firing_HP_perm]; PFC_shuffle= [DM_PFC_perm,firing_PFC_perm]

            # find coefficients for permuted PFC and CA1 data on A trials and B trials 
            C_1_HP_b, C_2_HP_b, C_3_HP_b = policy_A_B(HP_shuffle, n = n, plot_a = False, perm = True)
            C_1_PFC_b ,C_2_PFC_b, C_3_PFC_b = policy_A_B(PFC_shuffle, n = n, plot_a = False, perm = True)
            C_1_HP_a, C_2_HP_a, C_3_HP_a = policy_A_B(HP_shuffle, n = n, plot_a = True, perm = True)
            C_1_PFC_a ,C_2_PFC_a, C_3_PFC_a = policy_A_B(PFC_shuffle, n = n, plot_a = True, perm = True)

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

                coefs_tasks = [] # to store correlations in task 1 vs 2; task 2 vs task 3; task 3 vs 1 in HP and PFC
                for t,pol_t in enumerate(policy_all_time): # compute correlations in different task combinations
                    policy_all_time_task = policy_all_time[t]; policy_init_ch_rew_task = policy_init_ch_rew[t]
                    coef_time = []
                    for pol in policy_all_time_task.T: # find a correlation coefficient for each 
                        coef_time.append(np.corrcoef(pol,policy_init_ch_rew_task)[0][1])
                    coefs_tasks.append(coef_time) # store coefficients
                # store mean corr coef across 3 tasks for init, choice and outcome HP and PFC (# CA1 init, CA1 ch, CA1 rew, PFC init, PFC ch, PFC rew)
                task_mean_A_B.append([np.mean(coefs_tasks[:3],0),np.mean(coefs_tasks[3:6],0), np.mean(coefs_tasks[6:9],0),np.mean(coefs_tasks[9:12],0),np.mean(coefs_tasks[12:15],0), np.mean(coefs_tasks[15:18],0)])

        b_a = np.asarray(task_mean_A_B[0]) - np.asarray(task_mean_A_B[1]) # difference between A and B (CA1 init, CA1 ch, CA1 rew, PFC init, PFC ch, PFC rew)
        # store permuted differences of differences between B and A between CA1 and PFC at initiation, choice and outcome time       
        diff_PFC_HP_perm.append((np.asarray(b_a[3:])- np.asarray(b_a[:3]))) # difference in B - A PFC - CA1
        PFC_HP_perm.append((np.asarray(task_mean_A_B)[:,3:]- np.asarray(task_mean_A_B)[:,:3])) # store difference between PFC and CA1 on A and on B choices
    return diff_PFC_HP_perm, PFC_HP_perm

def plot_correlations_slice(PFC, HP, n = 11, c_1 = 1, perm_n = 2, dlc = False, animal_perm = False):
    ''' This function calculates and plots correlation slices through time of coefficients at initiation, 
    choice and outcome times in PFC and CA1 data  and compares it to permuted differences.
    c_1 is an argument to select policy coefficients (1)
    n is the number of trials in the past that the history of choices is used for
    policy calculations; default is 11 based on significance testing after training
    perm_n is the number of permutation to run 
    '''
    if dlc == False:
        time_ind = [24, 35, 42]; _xticks = [0,12, 24, 35, 42, 49, 63]
    else:
        time_ind = [25, 38, 45]; _xticks = [0,12.5, 25, 38, 45, 51, 64]
        
    init_t = time_ind[0]; ch_t = time_ind[1]; r_t = time_ind[2] # initiation, choice, outcome
    # extract policy coefficients on A and B choices in PFC and CA1
    C_1_HP_b, C_2_HP_b, C_3_HP_b = policy_A_B(HP, n = n, plot_a = False,  perm = False)
    C_1_PFC_b ,C_2_PFC_b, C_3_PFC_b = policy_A_B(PFC, n = n, plot_a = False, perm = False)
    C_1_HP_a, C_2_HP_a, C_3_HP_a = policy_A_B(HP, n = n, plot_a = True, perm = False)
    C_1_PFC_a ,C_2_PFC_a, C_3_PFC_a = policy_A_B(PFC,n = n, plot_a = True,  perm = False)
    
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
        for t,pol_t in enumerate(policy_all_time): # compute correlations in different task combinations
            policy_all_time_task = policy_all_time[t]; policy_init_ch_rew_task = policy_init_ch_rew[t]
            coef_time = []
            for pol in policy_all_time_task.T: # find a correlation coefficient for each 
                coef_time.append(np.corrcoef(pol,policy_init_ch_rew_task)[0][1])
            coefs_tasks.append(coef_time) # store coefficients
        # store mean corr coef across 3 tasks for init, choice and outcome HP and PFC (# CA1 init, CA1 ch, CA1 rew, PFC init, PFC ch, PFC rew)
        task_mean_A_B.append([np.mean(coefs_tasks[:3],0),np.mean(coefs_tasks[3:6],0), np.mean(coefs_tasks[6:9],0),np.mean(coefs_tasks[9:12],0),np.mean(coefs_tasks[12:15],0), np.mean(coefs_tasks[15:18],0)])
    
    # find permuted differences
    diff_PFC_HP_perm, PFC_HP_perm = perumute_policy_A_B(HP, PFC, c_1 = c_1, n = n, perm_n = perm_n, dlc = dlc, animal_perm = animal_perm)
    diff_PFC_HP_perm = np.asarray(diff_PFC_HP_perm);  PFC_HP_perm  = np.asarray(PFC_HP_perm); task_mean_A_B = np.asarray(task_mean_A_B)
    _95th = np.percentile(diff_PFC_HP_perm, 95,0) # .05 difference in CA1 and PFC B - A choice policy
    
    b_a = np.asarray(task_mean_A_B[0])- np.asarray(task_mean_A_B[1]) # real B - A choice policy difference
    diff_PFC_HP = np.asarray(b_a[3:])- np.asarray(b_a[:3]) # difference in B - A PFC - CA1
    indx_b_a_diff = np.where(diff_PFC_HP > np.max(_95th)) # where difference between CA1 and PFC in B - A significant at .05 (corrected for multiple comparisons)
   
    # significant differences between CA1 and PFC on A and B choices
    _95th_a_b = np.percentile(PFC_HP_perm, 95, 0)
    
    # real A/B difference between CA1 and PFC on A and B trials
    a_b_real = task_mean_A_B[:,3:,:] - task_mean_A_B[:,:3,:] # PFC - CA1 
    significant_diff_a = np.where(a_b_real[0] > np.max(_95th_a_b[0])) # .05 significant on A
    significant_diff_b = np.where(a_b_real[1] > np.max(_95th_a_b[1]))  # .05 significant on B
    
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
                    p_a = significant_diff_a[1][np.where(significant_diff_a[0] == per)[0]]
                    p_b = significant_diff_b[1][np.where(significant_diff_b[0] == per)[0]]
                    p_a_b = indx_b_a_diff[1][np.where(indx_b_a_diff[0] == per)[0]] 
                    axs[per].plot(p_a, np.ones(p_a.shape)*ymax + 0.02, '.', markersize = 4, color= 'green', label = 'A PFC vs CA1')
                    axs[per].plot(p_b, np.ones(p_b.shape)*ymax + 0.05, '.', markersize = 4, color = 'pink', label = 'B PFC vs CA1')
                    axs[per].plot(p_a_b, np.ones(p_a_b.shape)*ymax + 0.07, '.', markersize = 4, color = 'grey',label = 'B - A PFC vs CA1')
                    axs[per].set_xticks(_xticks)
                    axs[per].set_ylim(-0.15, ymax+0.08)
                    axs[per].set_xticklabels(['-1','-0.5','Init', 'Ch','R', '+0.5', '+1'])
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
    C_1_b, C_2_b, C_3_b = policy_A_B(d, n = n, plot_a = False, perm = perm)
    C_1_a, C_2_a, C_3_a = policy_A_B(d, n = n, plot_a = True, perm = perm)

    mean_value_a = np.mean([np.corrcoef(C_1_a[c_1].T,C_2_a[c_1].T), np.corrcoef(C_1_a[c_1].T,C_3_a[c_1].T), np.corrcoef(C_2_a[c_1].T,C_3_a[c_1].T)],0)
    mean_value_b = np.mean([np.corrcoef(C_1_b[c_1].T,C_2_b[c_1].T), np.corrcoef(C_1_b[c_1].T,C_3_b[c_1].T), np.corrcoef(C_2_b[c_1].T,C_3_b[c_1].T)],0)
    
    diag_a = np.sum(np.diagonal(mean_value_a[index:,:index])); diag_b = np.sum(np.diagonal(mean_value_b[index:,:index]))   
    return mean_value_a, mean_value_b, diag_a, diag_b

def permute_sessions_diagonal(HP, PFC, c_1 = 1, n = 11, perm_n = 2, dlc = False, animal_perm = False):
    ''' This function permutes sessions between HP and PFC groups and finds
     the diagonal from the correlation matrices of policy regression coefficients
     between different tasks on A choices and on B choices 
     c_1 is an argument to select policy coefficients (1)
     perm_n is the number of permutation to run '''
    if animal_perm == False:
        
        all_shuffle_a = [];  all_shuffle_b = []
        all_subjects = np.hstack([PFC['DM'][0], HP['DM'][0]]); all_subjects_firing = np.hstack([PFC['Data'][0], HP['Data'][0]])
        sessions_n = np.arange(len(all_subjects))

        for i in range(perm_n):
            np.random.shuffle(sessions_n) # shuffle PFC/HP sessions
            indices_HP = sessions_n[PFC['DM'][0].shape[0]:]; indices_PFC = sessions_n[:PFC['DM'][0].shape[0]]

            PFC_shuffle_dm = all_subjects[np.asarray(indices_PFC)];  HP_shuffle_dm = all_subjects[np.asarray(indices_HP)]
            PFC_shuffle_f = all_subjects_firing[np.asarray(indices_PFC)]; HP_shuffle_f = all_subjects_firing[np.asarray(indices_HP)]

            HP_shuffle = [HP_shuffle_dm,HP_shuffle_f]; PFC_shuffle = [PFC_shuffle_dm,PFC_shuffle_f]
            _all_diff_a = []; _all_diff_b = []

            for d in [HP_shuffle,PFC_shuffle]:
                mean_value_a_HP, mean_value_b_HP, diag_a_HP ,diag_b_HP =  correlations_ab(d, n = n, perm = True, c_1 = c_1, dlc = dlc)
                _all_diff_a.append(diag_a_HP)
                _all_diff_b.append(diag_b_HP)
            all_shuffle_a.append(_all_diff_a);  all_shuffle_b.append(_all_diff_b)
    else:
        animals_PFC = [0,1,2,3]; animals_HP = [4,5,6]
        m, n = len(animals_PFC), len(animals_HP)
        all_subjects_DM, all_subjects_fr  = animal_exp_permute(HP,PFC)
        all_shuffle_a = [];  all_shuffle_b = []

        for indices_PFC in combinations(range(m + n), m):
            indices_HP = [i for i in range(m + n) if i not in indices_PFC]
            DM_PFC_perm = np.concatenate(np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_PFC)],0)
            firing_PFC_perm = np.concatenate(np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_PFC)],0)
                
            DM_HP_perm = np.concatenate(np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_HP)],0)
            firing_HP_perm = np.concatenate(np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_HP)],0)
            HP_shuffle= [DM_HP_perm,firing_HP_perm]; PFC_shuffle= [DM_PFC_perm,firing_PFC_perm]
            _all_diff_a = []; _all_diff_b = []
            for d in [HP_shuffle,PFC_shuffle]:
                mean_value_a_HP, mean_value_b_HP, diag_a_HP ,diag_b_HP =  correlations_ab(d, n = n, perm = True, c_1 = c_1, dlc = dlc)
                _all_diff_a.append(diag_a_HP)
                _all_diff_b.append(diag_b_HP)
            all_shuffle_a.append(_all_diff_a);  all_shuffle_b.append(_all_diff_b)

    diff_all_a  = []; diff_all_b  = []
    for i,ii in enumerate(all_shuffle_a):
        diff_all_a.append(all_shuffle_a[i][1] - all_shuffle_a[i][0])
        diff_all_b.append(all_shuffle_b[i][1] - all_shuffle_b[i][0])
        
    _all_95_a = np.percentile(diff_all_a,95)
    _all_95_b = np.percentile(diff_all_b,95)
    
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
        plt.xticks(_xticks, ['-1','-0.5','Init', 'Ch','R', '+0.5', '+1'])    
        plt.yticks(_xticks, ['-1','-0.5','Init', 'Ch','R', '+0.5', '+1']) 
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
        plt.vlines(real[d],ymin = 0, ymax = max(np.histogram(diff)[0]),  color = 'pink')
        plt.vlines(perm_diff[d], ymin = 0, ymax = max(np.histogram(diff)[0]))
        plt.xlabel('Permuted Differences'); plt.ylabel('Count')
        plt.title(str_title[d])
        sns.despine()
    plt.tight_layout()


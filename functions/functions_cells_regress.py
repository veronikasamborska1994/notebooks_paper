import numpy as np
import pylab as plt
import seaborn as sns
from palettable import wesanderson as wes
from itertools import combinations 
from math import factorial
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


def GLM_perm_across_regions(HP,PFC, perm = 1000, animal_perm = False):
    '''Simple linear regression predicting activity of each neuron at each time point across the trial, 
    as a function of the choice, outcome and outcome x choice interaction. 
    This function permutes sessions across animal groups.'''
    
    dm_HP = HP['DM'][0];  firing_HP = HP['Data'][0] # HP data
    dm_PFC = PFC['DM'][0];  firing_PFC = PFC['Data'][0] # PFC data
   
    cpd_1_HP  = []; cpd_2_HP  = []; cpd_3_HP  = [] # cpds to store within each task HP
    cpd_1_PFC  = []; cpd_2_PFC = []; cpd_3_PFC = [] # cpds to store within each task PFC
    dms = [dm_HP,dm_PFC]; firings = [firing_HP,firing_PFC] 
    
    for d,dm in enumerate(dms):
        firing = firings[d]
        for  s, sess in enumerate(dm):
            DM = dm[s]; firing_rates_all = firing[s]     
            # trial information
            task =  DM[:,5]; choices_all = DM[:,1]; reward_all = DM[:,2]  
            choices_all = choices_all-0.5
            rew_ch_all = choices_all*reward_all
            
            # make a design matrix for each task and calculate CPDs within each task
            tasks = [1,2,3]
            for i in tasks: 
                choice  = choices_all[np.where(task == i)[0]]
                reward = reward_all[np.where(task == i)[0]]
                reward_ch = rew_ch_all[np.where(task == i)[0]]
                ones = np.ones(len(reward_ch))
                firing_rates = firing_rates_all[np.where(task == i)[0]]
                n_trials, n_neurons, n_timepoints = firing_rates.shape
                X = np.vstack([choice,reward,reward_ch,ones]).T
                n_predictors = X.shape[1]
                y = firing_rates.reshape([len(firing_rates),-1]) # activity matrix [n_trials, n_neurons*n_timepoints]
                
                # calculate coefficients of partial determination
                if d == 0:
                    if i == 1:
                        cpd_1_HP.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
                    elif i == 2:
                        cpd_2_HP.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
                    elif i == 3:
                        cpd_3_HP.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))

                else:
                    if i == 1:
                        cpd_1_PFC.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
                    elif i == 2:
                        cpd_2_PFC.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
                    elif i == 3:
                        cpd_3_PFC.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))


    cpd_1_PFC = np.concatenate(cpd_1_PFC,0); cpd_2_PFC = np.concatenate(cpd_2_PFC,0); cpd_3_PFC = np.concatenate(cpd_3_PFC,0)
    cpd_1_HP = np.concatenate(cpd_1_HP,0); cpd_2_HP = np.concatenate(cpd_2_HP,0); cpd_3_HP = np.concatenate(cpd_3_HP,0)
    
    cpd_HP = np.nanmean([cpd_1_HP,cpd_2_HP,cpd_3_HP],0)   
    cpd_PFC = np.nanmean([cpd_1_PFC,cpd_2_PFC,cpd_3_PFC],0)   

    if perm: # permutation test
        all_subjects_DM = np.concatenate((HP['DM'][0],PFC['DM'][0]),0)     
        all_subjects_fr = np.concatenate((HP['Data'][0],PFC['Data'][0]),0)     
        diff_perm = np.zeros((int(perm),(all_subjects_fr[0].shape)[2],n_predictors))
        n_sessions = np.arange(len(HP['DM'][0])+len(PFC['DM'][0]))
        if animal_perm == True:
            animals_PFC = [0,1,2,3]; animals_HP = [4,5,6]
            m, n = len(animals_PFC), len(animals_HP)
            num_rounds = factorial(m + n) / (factorial(m)*factorial(n))
            diff_perm = np.zeros((int(num_rounds),(all_subjects_fr[0].shape)[2],n_predictors))
            all_subjects_DM,all_subjects_fr  = animal_exp_permute(HP,PFC)
            p = -1
            for indices_PFC in combinations(range(m + n), m):
                cpd_HP_1_perm = []; cpd_HP_2_perm = []; cpd_HP_3_perm  = []
                cpd_PFC_1_perm = []; cpd_PFC_2_perm = []; cpd_PFC_3_perm  = []
                p += 1
                indices_HP = [i for i in range(m + n) if i not in indices_PFC]
                DM_PFC_perm = np.concatenate(np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_PFC)],0)
                firing_PFC_perm = np.concatenate(np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_PFC)],0)
                
                DM_HP_perm = np.concatenate(np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_HP)],0)
                firing_HP_perm = np.concatenate(np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_HP)],0)
                dms_perm = [DM_PFC_perm, DM_HP_perm]; firings_perms = [firing_PFC_perm,firing_HP_perm]
           
            
                for d, dm in enumerate(dms_perm): # same as above but calculate for permuted sessions
                    firing = firings_perms[d]
                    for  s, sess in enumerate(dm):
                        DM = dm[s]; firing_rates_all = firing[s]
                       # trial information
                        task =  DM[:,5]; choices_all = DM[:,1]; reward_all = DM[:,2]  
                        choices_all = choices_all-0.5
                        rew_ch_all = choices_all*reward_all

                        # make a design matrix for each task and calculate CPDs within each task
                        tasks = [1,2,3]
                        for i in tasks: 
                            choice  = choices_all[np.where(task == i)[0]]
                            reward = reward_all[np.where(task == i)[0]]
                            reward_ch = rew_ch_all[np.where(task == i)[0]]
                            ones = np.ones(len(reward_ch))
                            firing_rates = firing_rates_all[np.where(task == i)[0]]
                            n_trials, n_neurons, n_timepoints = firing_rates.shape
                            X = np.vstack([choice,reward,reward_ch,ones]).T
                            n_predictors = X.shape[1]
                            y = firing_rates.reshape([len(firing_rates),-1]) # activity matrix [n_trials, n_neurons*n_timepoints]

                            if d == 0:
                                if i == 1:
                                    cpd_HP_1_perm.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
                                elif i == 2:
                                    cpd_HP_2_perm.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
                                elif i == 3:
                                    cpd_HP_3_perm.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))

                            else:
                                if i == 1:
                                    cpd_PFC_1_perm.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
                                elif i == 2:
                                    cpd_PFC_2_perm.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
                                elif i == 3:
                                    cpd_PFC_3_perm.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
                cpd_HP_1_perm = np.concatenate(cpd_HP_1_perm,0); cpd_HP_2_perm = np.concatenate(cpd_HP_2_perm,0);cpd_HP_3_perm = np.concatenate(cpd_HP_3_perm,0)
                cpd_PFC_1_perm = np.concatenate(cpd_PFC_1_perm,0); cpd_PFC_2_perm = np.concatenate(cpd_PFC_2_perm,0);cpd_PFC_3_perm = np.concatenate(cpd_PFC_3_perm,0)
                cpd_PFC_perm = np.nanmean([cpd_PFC_1_perm,cpd_PFC_2_perm,cpd_PFC_3_perm],0);cpd_HP_perm = np.nanmean([cpd_HP_1_perm,cpd_HP_2_perm,cpd_HP_3_perm],0)
                diff_perm[p,:] = abs(np.mean(cpd_PFC_perm,0) - np.mean(cpd_HP_perm,0))
        else:
            for p in range(perm):
                np.random.shuffle(n_sessions) # shuffle PFC/HP sessions
                indices_HP = n_sessions[:len(HP['DM'][0])]
                indices_PFC = n_sessions[len(HP['DM'][0]):]
                DM_PFC_perm = all_subjects_DM[np.asarray(indices_PFC)]
                firing_PFC_perm = all_subjects_fr[np.asarray(indices_PFC)]

                DM_HP_perm = all_subjects_DM[np.asarray(indices_HP)]
                firing_HP_perm = all_subjects_fr[np.asarray(indices_HP)]

                cpd_HP_1_perm = []; cpd_HP_2_perm = []; cpd_HP_3_perm  = []
                cpd_PFC_1_perm = []; cpd_PFC_2_perm = []; cpd_PFC_3_perm  = []

                dms_perm = [DM_PFC_perm,DM_HP_perm]
                firings_perms = [firing_PFC_perm,firing_HP_perm]

                for d, dm in enumerate(dms_perm): # same as above but calculate for permuted sessions
                    firing = firings_perms[d]
                    for  s, sess in enumerate(dm):
                        DM = dm[s]; firing_rates_all = firing[s]
                       # trial information
                        task =  DM[:,5]; choices_all = DM[:,1]; reward_all = DM[:,2]  
                        choices_all = choices_all-0.5
                        rew_ch_all = choices_all*reward_all

                        # make a design matrix for each task and calculate CPDs within each task
                        tasks = [1,2,3]
                        for i in tasks: 
                            choice  = choices_all[np.where(task == i)[0]]
                            reward = reward_all[np.where(task == i)[0]]
                            reward_ch = rew_ch_all[np.where(task == i)[0]]
                            ones = np.ones(len(reward_ch))
                            firing_rates = firing_rates_all[np.where(task == i)[0]]
                            n_trials, n_neurons, n_timepoints = firing_rates.shape
                            X = np.vstack([choice,reward,reward_ch,ones]).T
                            n_predictors = X.shape[1]
                            y = firing_rates.reshape([len(firing_rates),-1]) # activity matrix [n_trials, n_neurons*n_timepoints]

                            if d == 0:
                                if i == 1:
                                    cpd_HP_1_perm.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
                                elif i == 2:
                                    cpd_HP_2_perm.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
                                elif i == 3:
                                    cpd_HP_3_perm.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))

                            else:
                                if i == 1:
                                    cpd_PFC_1_perm.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
                                elif i == 2:
                                    cpd_PFC_2_perm.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
                                elif i == 3:
                                    cpd_PFC_3_perm.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))

      
                cpd_HP_1_perm = np.concatenate(cpd_HP_1_perm,0); cpd_HP_2_perm = np.concatenate(cpd_HP_2_perm,0);cpd_HP_3_perm = np.concatenate(cpd_HP_3_perm,0)
                cpd_PFC_1_perm = np.concatenate(cpd_PFC_1_perm,0); cpd_PFC_2_perm = np.concatenate(cpd_PFC_2_perm,0);cpd_PFC_3_perm = np.concatenate(cpd_PFC_3_perm,0)
                cpd_PFC_perm = np.nanmean([cpd_PFC_1_perm,cpd_PFC_2_perm,cpd_PFC_3_perm],0);cpd_HP_perm = np.nanmean([cpd_HP_1_perm,cpd_HP_2_perm,cpd_HP_3_perm],0)
                diff_perm[p,:] = abs(np.mean(cpd_PFC_perm,0) - np.mean(cpd_HP_perm,0))
    p_95 = np.percentile(diff_perm,95, axis = 0) # find 95th percentile
    p_99 = np.percentile(diff_perm,99, axis = 0) # find 99th percentile
    real_diff = np.abs(np.mean(cpd_PFC,0) - np.mean(cpd_HP,0))
    return p_95, p_99, real_diff
    


def GLM_roll_time(data, perm = 1000):
    '''Simple linear regression predicting activity of each neuron at each time point across the trial, 
    as a function of the choice, outcome and outcome x choice interaction. 
    This function randomly rolls firing rates with respect to trials.'''
    
    dm = data['DM'][0];  firing = data['Data'][0]
    cpd_perm_1  = [[] for i in range(perm)]; cpd_perm_2  = [[] for i in range(perm)]; cpd_perm_3  = [[] for i in range(perm)] 
    cpd_1  = []; cpd_2  = [];  cpd_3 = []
    
    for  s, sess in enumerate(dm):
        DM = dm[s]; firing_rates_all = firing[s]     
        # trial information
        task =  DM[:,5]; choices_all = DM[:,1]; reward_all = DM[:,2]  
        choices_all = choices_all-0.5
        rew_ch_all = choices_all*reward_all
        block_all = DM[:,4]
        block_min = np.min(np.diff(np.where(np.diff(block_all)!=0)))
  
        # make a design matrix for each task and calculate CPDs within each task
        tasks = [1,2,3]
        firings_tasks = []; dm_tasks = []
        for i in tasks: 
            choice  = choices_all[np.where(task == i)[0]]
            reward = reward_all[np.where(task == i)[0]]
            reward_ch = rew_ch_all[np.where(task == i)[0]]
            ones = np.ones(len(reward_ch))
    
            firing_rates = firing_rates_all[np.where(task == i)[0]]
            X = np.vstack([choice,reward,reward_ch,ones]).T
            n_trials, n_neurons, n_timepoints = firing_rates.shape

            n_predictors = X.shape[1]
            y = firing_rates.reshape([len(firing_rates),-1]) # activity matrix [n_trials, n_neurons*n_timepoints]
            firings_tasks.append(y)
            dm_tasks.append(X)
            
            if i == 1:
                cpd_1.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
            elif i == 2:
                cpd_2.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))
            elif i == 3:
                cpd_3.append(_CPD(X,y).reshape(n_neurons, n_timepoints, n_predictors))


        for p in range(perm): # randomly rolls firing rates with respect to trials
            y_perm_1 = np.roll(firings_tasks[0],np.random.randint(block_min, n_trials), axis = 0)
            cpd_perm_1[p].append(_CPD(dm_tasks[0],y_perm_1).reshape(n_neurons, n_timepoints, n_predictors))
            
            y_perm_2 = np.roll(firings_tasks[1],np.random.randint(block_min, n_trials), axis = 0)
            cpd_perm_2[p].append(_CPD(dm_tasks[1],y_perm_2).reshape(n_neurons, n_timepoints, n_predictors))
            
            y_perm_3 = np.roll(firings_tasks[2],np.random.randint(block_min, n_trials), axis = 0)
            cpd_perm_3[p].append(_CPD(dm_tasks[2],y_perm_3).reshape(n_neurons, n_timepoints, n_predictors))
            
    cpd_perm_1   = np.stack([np.concatenate(cpd_i,0) for cpd_i in cpd_perm_1],0)
    cpd_perm_2   = np.stack([np.concatenate(cpd_i,0) for cpd_i in cpd_perm_2],0)
    cpd_perm_3  = np.stack([np.concatenate(cpd_i,0) for cpd_i in cpd_perm_3],0)

    cpd_1 = np.concatenate(cpd_1,0); cpd_2 = np.concatenate(cpd_2,0); cpd_3 = np.concatenate(cpd_3,0)
    cpd = np.nanmean([cpd_1,cpd_2,cpd_3],0);  cpd_perm = np.nanmean([cpd_perm_1,cpd_perm_2,cpd_perm_3],0)

    return cpd,cpd_perm

def plot_simple_GLM(HP,PFC, n_perm = 1000, animal_perm = False, dlc = False): 
    ''' Plotting CPDs and significance levels, Figure 3B)'''
    cpd_HP, cpd_perm_HP =  GLM_roll_time(HP, perm = n_perm)
    cpd_PFC, cpd_perm_PFC =  GLM_roll_time(PFC, perm = n_perm)
    p_95, p_99, real_diff = GLM_perm_across_regions(HP, PFC, perm = n_perm,  animal_perm = animal_perm)
    if dlc == False:
        _xticks = [0,12, 24, 35, 42, 49, 63]
    else:
        _xticks = [0,12.5, 25, 38, 45, 51, 64]
    # significance levels for session permutation across animal groups (between region comparison)
    time_controlled_95 = np.max(p_95,0); time_controlled_99 = np.max(p_99,0) # corrected for multiple tests
    indicies_95 = np.where(real_diff > time_controlled_95);indicies_99 = np.where(real_diff > time_controlled_99)
    
    cpd_HP_m = np.nanmean(cpd_HP,0); cpd_PFC_m = np.nanmean(cpd_PFC,0)
    t = np.arange(0, (cpd_HP.shape)[1]) # time index 
    cpds = [cpd_HP_m,cpd_PFC_m] # real cpds 
    cpds_perms = [cpd_perm_HP, cpd_perm_PFC] # permuted across time cpds (within each region)
    fig, axs = plt.subplots(1,3, figsize=(20, 5));  c = wes.Royal2_5.mpl_colors
    label_region = ['HP', 'PFC']; p = ['Choice','Reward', 'Reward x Choice'] # lists of labels

    for region,cpd in enumerate(cpds): # loop through HP and PFC
        cpd = cpds[region][:,:-1] # select all cpds except constant
        cpd_perm = cpds_perms[region][:,:,:,:-1]
        values_95 = np.max(np.percentile(np.mean(cpd_perm,1),95,0),0) # 95th percentile corrected for multiple tests
        values_99 = np.max(np.percentile(np.mean(cpd_perm,1),99,0),0)  # 99th percentile corrected for multiple tests
        array_pvals = np.ones((cpd.shape[0],cpd.shape[1]))
        for pred in range(cpd.shape[1]): # loop through pred
            array_pvals[(np.where(cpd[:,pred] > values_95[pred])[0]),pred] = 0.05 # indicate where significant at 0.05 for each region (loop)
            array_pvals[(np.where(cpd[:,pred] > values_99[pred])[0]),pred] = 0.001 # indicate where significant at 0.001 for each region (loop)

        for pred in np.arange(cpd.shape[1]):
            # random stylistic things
            ymax = np.max([np.max(cpd_HP_m[:,pred]),np.max(cpd_PFC_m[:,pred])])*100 
            if ymax > 10:
                offset = 0.5; offset2 = 0.8; offset3 = 1
            elif ymax > 5:
                offset = 0.5; offset2 = 0.7; offset3 = 0.9
            elif ymax < 5:
                offset = 0.5; offset2 = 0.55; offset3 = 0.6
            
            # plot real cpds 
            axs[pred].plot(cpd[:,pred]*100, color = c[region], label = label_region[region])
            axs[pred].set_title(p[pred])
            
            
            p_vals = array_pvals[:,pred] # p-value of 0.001 or 0.05 for loop predictor 
            t05 = t[p_vals == 0.05]; t001 = t[p_vals == 0.001] 
        
            index_95 = indicies_95[0][np.where(indicies_95[1] == pred)[0]]  # between region 95th percentile
            index_99 = indicies_99[0][np.where(indicies_99[1] == pred)[0]]  # between region 99th percentile

            if region == 0:
                axs[pred].plot(t05, np.ones(t05.shape)*ymax + offset, '.', markersize = 2, color = c[region])
                axs[pred].plot(t001, np.ones(t001.shape)*ymax + offset, '.', markersize = 4, color = c[region], label = label_region[region] + ' ' + '< .001')
                axs[pred].plot(index_95, np.ones(index_95.shape)*ymax + offset2, '.', markersize = 2, color = 'grey')
                axs[pred].plot(index_99, np.ones(index_99.shape)*ymax + offset2, '.', markersize = 4, color = 'grey',label = '< .001 diff PFC and HP ')
                axs[pred].set_xticks(_xticks)
                axs[pred].set_xticklabels(['-1','-0.5','Init', 'Ch','R', '+0.5', '+1'])
                axs[pred].set_xlabel('Time in Trial (s)')
                axs[pred].set_ylabel('CPD (%)')
                sns.despine()
            else: 
                axs[pred].plot(t05, np.ones(t05.shape)*ymax +offset3, '.', markersize = 2, color = c[region])
                axs[pred].plot(t001, np.ones(t001.shape)*ymax + offset3, '.', markersize = 4, color = c[region],label = label_region[region] + ' ' + '< .001')
            axs[pred].legend()


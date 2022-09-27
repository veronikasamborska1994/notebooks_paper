import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations 


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


def extract_data(data, perm = False, inds =  np.arange(13,51)):
    '''This function arranges data in a format used in later SVD analysis - finds A reward, A non-rewarded,
    B reward, B non-rewarded average firing rates for each neuron in 3 tasks 
    split by first and second half of the task for cross-validation
    inds =  np.arange(13,51) #-500+500 '''
    
    if perm == False:
        all_subjects = data['DM'][0] # trial data
        all_firing = data['Data'][0]  # firing rates
    else:
        all_subjects = data[0]
        all_firing = data[1]
        
    neurons = 0
    for s in all_firing:
        neurons += s.shape[1] # total # of neurons in each region
    
    n_neurons_cum = 0
    len_trial = len(inds)
    
    # Create matrices to store data from tasks split by first and second half 
    flattened_all_clusters_task_1_first_half =  np.zeros((neurons,len_trial*4))
    flattened_all_clusters_task_1_second_half = np.zeros((neurons,len_trial*4))
    flattened_all_clusters_task_2_first_half = np.zeros((neurons,len_trial*4))
    flattened_all_clusters_task_2_second_half = np.zeros((neurons,len_trial*4))
    flattened_all_clusters_task_3_first_half = np.zeros((neurons,len_trial*4))
    flattened_all_clusters_task_3_second_half = np.zeros((neurons,len_trial*4))
     
    for  s, sess in enumerate(all_firing): #itirate through sessions
            DM = all_subjects[s]
            firing_rates = all_firing[s][:,:,inds]
            n_trials, n_neurons, n_timepoints = firing_rates.shape
            n_neurons_cum += n_neurons 
            
            # extract trial information (reward, task, choice A/B)
            choices = DM[:,1]; reward = DM[:,2]; task =  DM[:,5]; 
            task_1 = np.where(task == 1)[0]; task_2 = np.where(task == 2)[0]; task_3 = np.where(task == 3)[0]
        
            # task indicies of the first/last half
            task_1_1 = task_1[:int(len(task_1)/2)]; task_1_2 = task_1[int(len(task_1)/2):] 
            task_2_1 = task_2[:int(len(task_2)/2)]; task_2_2 = task_2[int(len(task_2)/2):]
            task_3_1 = task_3[:int(len(task_3)/2)]; task_3_2 = task_3[int(len(task_3)/2):]
            
            
            # A rewarded and A non-rewared Task 1 First Half
            flattened_all_clusters_task_1_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,:len_trial] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_1_1))],0)
            flattened_all_clusters_task_1_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial:len_trial*2] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_1_1))],0)
            # B rewarded and B non-rewared Task 1 First Half
            flattened_all_clusters_task_1_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial*2:len_trial*3] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_1_1))],0)
            flattened_all_clusters_task_1_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial*3:len_trial*4] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_1_1))],0)

            # A rewarded and A non-rewared Task 1 Second Half
            flattened_all_clusters_task_1_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,:len_trial]  = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_1_2))],0)
            flattened_all_clusters_task_1_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial:len_trial*2] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_1_2))],0)
            # B rewarded and B non-rewared Task 1 Second Half
            flattened_all_clusters_task_1_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial*2:len_trial*3] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_1_2))],0)
            flattened_all_clusters_task_1_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial*3:len_trial*4] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_1_2))],0)
            
            # A rewarded and A non-rewared Task 2 First Half
            flattened_all_clusters_task_2_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,:len_trial] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_2_1))],0)
            flattened_all_clusters_task_2_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial:len_trial*2] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_2_1))],0)
            # B rewarded and B non-rewared Task 2 First Half
            flattened_all_clusters_task_2_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial*2:len_trial*3] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_2_1))],0)
            flattened_all_clusters_task_2_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial*3:len_trial*4] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_2_1))],0)

            # A rewarded and A non-rewared Task 2 Second Half
            flattened_all_clusters_task_2_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,:len_trial]  = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_2_2))],0)
            flattened_all_clusters_task_2_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial:len_trial*2] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_2_2))],0)
            # B rewarded and B non-rewared Task 2 Second Half
            flattened_all_clusters_task_2_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial*2:len_trial*3] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_2_2))],0)
            flattened_all_clusters_task_2_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial*3:len_trial*4]= np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_2_2))],0)
           
            # A rewarded and A non-rewared Task 3 First Half 
            flattened_all_clusters_task_3_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,:len_trial]  = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_3_1))],0)
            flattened_all_clusters_task_3_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial:len_trial*2]= np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_3_1))],0)
            # B rewarded and B non-rewared Task 3 First Half
            flattened_all_clusters_task_3_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial*2:len_trial*3] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_3_1))],0)
            flattened_all_clusters_task_3_first_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial*3:len_trial*4]= np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_3_1))],0)

            # A rewarded and A non-rewared Task 3 Second Half             
            flattened_all_clusters_task_3_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,:len_trial] = np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 1))[0],(task_3_2))],0)
            flattened_all_clusters_task_3_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial:len_trial*2] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 1))[0],(task_3_2))],0)
            # B rewarded and B non-rewared Task 3 Second Half
            flattened_all_clusters_task_3_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial*2:len_trial*3]= np.mean(firing_rates[np.intersect1d(np.where((reward==1) & (choices == 0))[0],(task_3_2))],0)
            flattened_all_clusters_task_3_second_half[n_neurons_cum-n_neurons:n_neurons_cum ,len_trial*3:len_trial*4] = np.mean(firing_rates[np.intersect1d(np.where((reward==0) & (choices == 0))[0],(task_3_2))],0)
         
      
    return flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
        flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
        flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half
       
def svd(flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
    flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
    flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, diagonal = False, axis = 0):

    # Get Us and Vs for Task 1 Second Half
    u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(flattened_all_clusters_task_1_second_half, full_matrices = False)   
    # Get Us and Vs for Task 2 Second Half
    u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(flattened_all_clusters_task_2_second_half, full_matrices = False)    
    
    t_u_t_1_2 = np.transpose(u_t1_2); t_v_t_1_2 = np.transpose(vh_t1_2) 
    t_u_t_2_2 = np.transpose(u_t2_2); t_v_t_2_2 = np.transpose(vh_t2_2)  
    n_neurons = flattened_all_clusters_task_1_first_half.shape[0]
    
    # Project task 1 first half onto modes from task 1 second half
    s_task_1_2 = np.linalg.multi_dot([t_u_t_1_2, flattened_all_clusters_task_1_first_half, t_v_t_1_2])
    
    if diagonal == False:
        s_1_2 = s_task_1_2.diagonal() # for both cellular and temporal modes look at the diagonal 
    else:
        s_1_2 = np.sum(s_task_1_2**2, axis = axis) # for either cellular and temporal modes look at either 0 or 1 axis
    sum_c_task_1_2 = np.cumsum(abs(s_1_2))/n_neurons #normalise by number of neurons
    
    # Project task 2 first half onto modes from task 1 second half (otherwise same logic)
    s_task_2_1_from_t_1_2 = np.linalg.multi_dot([t_u_t_1_2, flattened_all_clusters_task_2_first_half, t_v_t_1_2])
    if diagonal == False:
        s_2_1_from_t_1_2 = s_task_2_1_from_t_1_2.diagonal()
    else:
        s_2_1_from_t_1_2 = np.sum(s_task_2_1_from_t_1_2**2, axis = axis)
    sum_c_task_2_1_from_t_1_2 = np.cumsum(abs(s_2_1_from_t_1_2))/n_neurons
   
    # Project task 2 first half onto modes from task 2 second half (otherwise same logic)
    s_task_2_1_from_t_2_2 = np.linalg.multi_dot([t_u_t_2_2, flattened_all_clusters_task_2_first_half, t_v_t_2_2])    
    if diagonal == False:
        s_2_1_from_t_2_2 = s_task_2_1_from_t_2_2.diagonal()
    else:
        s_2_1_from_t_2_2 = np.sum(s_task_2_1_from_t_2_2**2, axis = axis)
    sum_c_task_2_1_from_t_2_2 = np.cumsum(abs(s_2_1_from_t_2_2))/n_neurons
    
    #Compare task 3 First Half from Task 1 Last Half 
    s_task_3_1_from_t_1_2 = np.linalg.multi_dot([t_u_t_1_2, flattened_all_clusters_task_3_first_half, t_v_t_1_2])
    if diagonal == False:
        s_3_1_from_t_1_2 = s_task_3_1_from_t_1_2.diagonal()
    else:
        s_3_1_from_t_1_2 = np.sum(s_task_3_1_from_t_1_2**2, axis = axis)
    sum_c_task_3_1_from_t_1_2 = np.cumsum(abs(s_3_1_from_t_1_2))/n_neurons


    average_within_all = np.mean([sum_c_task_1_2, sum_c_task_2_1_from_t_2_2], axis = 0) # average within task projections
    average_between_all = np.mean([sum_c_task_2_1_from_t_1_2, sum_c_task_3_1_from_t_1_2], axis = 0)# average between task projections

    if diagonal == True:
        average_within = average_within_all/average_within_all[-1] # get within task variance explained for cellular OR temporal modes 
        average_between = average_between_all/average_between_all[-1] # get between task variance explained for cellular OR temporal modes 
   
    else: # if taking diagonal for both cellular and temporal modes --> # cumulative weights
        average_within = average_within_all
        average_between = average_between_all
   
    trp = (np.trapz(average_within) - np.trapz(average_between))/average_within.shape[0] # area under the curve
    return trp,average_between,average_within


def real_diff(data, diagonal = False, axis = 0, perm = False, cell = False, inds = np.arange(13,51)):
    flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
    flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
    flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = extract_data(data, inds =  inds)
    
    if perm == True and cell == True: # if you want to shuffle cells within each task
        np.random.shuffle(flattened_all_clusters_task_1_first_half)
        np.random.shuffle(flattened_all_clusters_task_2_first_half)  
        np.random.shuffle(flattened_all_clusters_task_3_first_half)  
        
        np.random.shuffle(flattened_all_clusters_task_1_second_half)
        np.random.shuffle(flattened_all_clusters_task_2_second_half)  
        np.random.shuffle(flattened_all_clusters_task_3_second_half) 
  
    elif perm == True and cell == False: # if you want to shuffle time within each task
        np.random.shuffle(flattened_all_clusters_task_1_first_half.T)
        np.random.shuffle(flattened_all_clusters_task_2_first_half.T)  
        np.random.shuffle(flattened_all_clusters_task_3_first_half.T)  
        
        np.random.shuffle(flattened_all_clusters_task_1_second_half.T)
        np.random.shuffle(flattened_all_clusters_task_2_second_half.T)  
        np.random.shuffle(flattened_all_clusters_task_3_second_half.T) 
        
    trp, average_between_all,average_within_all = svd(flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
    flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
    flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, diagonal = diagonal, axis = axis)
   
    return trp, average_between_all,average_within_all

def cell_time_shuffle_vs_real(data, d = True,  axis = 0, shuffle_cells = True, HP_ = 'False', inds =  np.arange(13,51), shuffle_plot = True):
    
    # temporal and cellular vectors from real data 
    trp,  between,within =  real_diff(data,  diagonal = d,axis = axis, perm = False, cell = False, inds  = inds)
    # temporal and cellular vectors from shuffled data 
    trp_perm,  average_between_all_perm,average_within_all_perm =  real_diff(data, diagonal = d, axis = axis, perm = True, cell = shuffle_cells, inds = inds)
    
    perms_ = np.mean([average_between_all_perm,average_within_all_perm],0)
   
    if d == False:
        perms_norm =  (perms_/within[-1])
        norm_within = within/within[-1]
        norm_between = between/within[-1]
    else:
        perms_norm = perms_/within[-1]
        norm_within = within
        norm_between = between
 
    if HP_ == 'True':
        plt.plot(norm_within*100, label = 'Within HP', color='black')
        if shuffle_plot == True:
            plt.plot(perms_norm*100, label = 'perm HP', color = 'grey')
        plt.plot(norm_between*100, label = 'Between HP', color='black',linestyle = '--')
            
    else:
        plt.plot(norm_within*100, label = 'Within PFC', color='green')
        if shuffle_plot == True:
            plt.plot(perms_norm*100, label = 'perm PFC', color = 'lime')
        plt.plot(norm_between*100, label = 'Between PFC', color='green',linestyle = '--')
            
    sns.despine()
    plt.legend()

def svd_between_brains(HP,PFC, inds = np.arange(13,51)):
    axis = 0
    # HP 
    flattened_all_clusters_task_1_first_half_HP, flattened_all_clusters_task_1_second_half_HP,\
    flattened_all_clusters_task_2_first_half_HP, flattened_all_clusters_task_2_second_half_HP,\
    flattened_all_clusters_task_3_first_half_HP,flattened_all_clusters_task_3_second_half_HP = extract_data(HP, perm = False, inds = inds)
    n_neurons_HP = flattened_all_clusters_task_1_first_half_HP.shape[0]
       
    trp_HP, average_between_all_HP, average_within_all_HP = svd(flattened_all_clusters_task_1_first_half_HP, flattened_all_clusters_task_1_second_half_HP,\
    flattened_all_clusters_task_2_first_half_HP, flattened_all_clusters_task_2_second_half_HP,\
    flattened_all_clusters_task_3_first_half_HP, flattened_all_clusters_task_3_second_half_HP, diagonal = True, axis = axis)
    # Get Us and Vs for Task 1 Second Half HP
    u_t1_2_HP, s_t1_2_HP, vh_t1_2_HP = np.linalg.svd(flattened_all_clusters_task_1_second_half_HP, full_matrices = False)   
    # Get Us and Vs for Task 2 Second Half HP
    u_t2_2_HP, s_t2_2_HP, vh_t2_2_HP = np.linalg.svd(flattened_all_clusters_task_2_second_half_HP, full_matrices = False)    
    t_u_t_1_2_HP = np.transpose(u_t1_2_HP); t_v_t_1_2_HP = np.transpose(vh_t1_2_HP) 
    t_u_t_2_2_HP = np.transpose(u_t2_2_HP); t_v_t_2_2_HP = np.transpose(vh_t2_2_HP)  
    
    # PFC
    flattened_all_clusters_task_1_first_half_PFC, flattened_all_clusters_task_1_second_half_PFC,\
    flattened_all_clusters_task_2_first_half_PFC, flattened_all_clusters_task_2_second_half_PFC,\
    flattened_all_clusters_task_3_first_half_PFC,flattened_all_clusters_task_3_second_half_PFC = extract_data(PFC, perm = False, inds = inds)
    n_neurons_PFC = flattened_all_clusters_task_1_first_half_PFC.shape[0]

    # Get Us and Vs for Task 1 Second Half PFC
    u_t1_2_PFC, s_t1_2_PFC, vh_t1_2_PFC = np.linalg.svd(flattened_all_clusters_task_1_second_half_PFC, full_matrices = False)   
    # Get Us and Vs for Task 2 Second Half PFC
    u_t2_2_PFC, s_t2_2_PFC, vh_t2_2_PFC = np.linalg.svd(flattened_all_clusters_task_2_second_half_PFC, full_matrices = False)    
    
    t_u_t_1_2_PFC = np.transpose(u_t1_2_PFC); t_v_t_1_2_PFC = np.transpose(vh_t1_2_PFC) 
    t_u_t_2_2_PFC = np.transpose(u_t2_2_PFC); t_v_t_2_2_PFC = np.transpose(vh_t2_2_PFC)  
   
    trp_PFC, average_between_all_PFC, average_within_all_PFC = svd(flattened_all_clusters_task_1_first_half_PFC, flattened_all_clusters_task_1_second_half_PFC,\
    flattened_all_clusters_task_2_first_half_PFC, flattened_all_clusters_task_2_second_half_PFC,\
    flattened_all_clusters_task_3_first_half_PFC, flattened_all_clusters_task_3_second_half_PFC, diagonal = True, axis = axis)

    # Projections for HP from PFC between task
    # Project task 2 first half onto modes from task 1 second half (otherwise same logic) using PFC vectors
    s_task_2_1_from_t_1_2_HP = np.linalg.multi_dot([flattened_all_clusters_task_2_first_half_HP, t_v_t_1_2_PFC])
    s_2_1_from_t_1_2_HP = np.sum(s_task_2_1_from_t_1_2_HP**2, axis = axis)
    sum_c_task_2_1_from_t_1_2_HP = np.cumsum(abs(s_2_1_from_t_1_2_HP))/n_neurons_HP
    
    # Compare task 3 First Half from Task 1 Last Half using PFC vectors
    s_task_3_1_from_t_1_2_HP = np.linalg.multi_dot([flattened_all_clusters_task_3_first_half_HP, t_v_t_1_2_PFC])
    s_3_1_from_t_1_2_HP = np.sum(s_task_3_1_from_t_1_2_HP**2, axis = axis)
    sum_c_task_3_1_from_t_1_2_HP = np.cumsum(abs(s_3_1_from_t_1_2_HP))/n_neurons_HP

    # Projections for HP from PFC within task
    # Project task 1 first half onto modes from task 1 second half  using PFC vectors
    s_task_1_2_HP = np.linalg.multi_dot([flattened_all_clusters_task_1_first_half_HP, t_v_t_1_2_PFC])
    s_1_2_HP = np.sum(s_task_1_2_HP**2, axis = axis) # for either cellular and temporal modes look at either 0 or 1 axis
    sum_c_task_1_2_HP = np.cumsum(abs(s_1_2_HP))/n_neurons_HP #normalise by number of neurons
    
    # Project task 2 first half onto modes from task 2 second half (otherwise same logic)  using PFC vectors
    s_task_2_1_from_t_2_2_HP = np.linalg.multi_dot([flattened_all_clusters_task_2_first_half_HP, t_v_t_1_2_PFC])    
    s_2_1_from_t_2_2_HP = np.sum(s_task_2_1_from_t_2_2_HP**2, axis = axis)
    sum_c_task_2_1_from_t_2_2_HP = np.cumsum(abs(s_2_1_from_t_2_2_HP))/n_neurons_HP
    
    average_within_all_HP_from_PFC = np.mean([sum_c_task_1_2_HP, sum_c_task_2_1_from_t_2_2_HP], axis = 0)# average between task projections
    average_within_all_HP_from_PFC = average_within_all_HP_from_PFC/average_within_all_HP_from_PFC[-1] # get between task variance explained for cellular OR temporal modes 

    average_between_all_HP_from_PFC = np.mean([sum_c_task_2_1_from_t_1_2_HP, sum_c_task_3_1_from_t_1_2_HP], axis = 0)# average between task projections
    average_between_all_HP_from_PFC = average_between_all_HP_from_PFC/average_between_all_HP_from_PFC[-1] # get between task variance explained for cellular OR temporal modes 

    # Projections for PFC from HP between task
    # Project task 2 first half onto modes from task 1 second half (otherwise same logic) using HP vectors
    s_task_2_1_from_t_1_2_PFC= np.linalg.multi_dot([flattened_all_clusters_task_2_first_half_PFC, t_v_t_1_2_HP])
    s_2_1_from_t_1_2_PFC = np.sum(s_task_2_1_from_t_1_2_PFC**2, axis = axis)
    sum_c_task_2_1_from_t_1_2_PFC = np.cumsum(abs(s_2_1_from_t_1_2_PFC))/n_neurons_PFC
   
    #Compare task 3 First Half from Task 1 Last Half using HP vectors
    s_task_3_1_from_t_1_2_PFC = np.linalg.multi_dot([flattened_all_clusters_task_3_first_half_PFC, t_v_t_1_2_HP])
    s_3_1_from_t_1_2_PFC = np.sum(s_task_3_1_from_t_1_2_PFC**2, axis = axis)
    sum_c_task_3_1_from_t_1_2_PFC = np.cumsum(abs(s_3_1_from_t_1_2_PFC))/n_neurons_PFC
      
    # Projections for PFC from HP within task
    # Project task 1 first half onto modes from task 1 second half using HP vectors
    s_task_1_2_PFC = np.linalg.multi_dot([flattened_all_clusters_task_1_first_half_PFC, t_v_t_1_2_HP])
    s_1_2_PFC = np.sum(s_task_1_2_PFC**2, axis = axis) # for either cellular and temporal modes look at either 0 or 1 axis
    sum_c_task_1_2_PFC = np.cumsum(abs(s_1_2_PFC))/n_neurons_PFC #normalise by number of neurons
    
    # Project task 2 first half onto modes from task 2 second half (otherwise same logic)  using HP vectors
    s_task_2_1_from_t_2_2_PFC = np.linalg.multi_dot([flattened_all_clusters_task_2_first_half_PFC, t_v_t_1_2_HP])    
    s_2_1_from_t_2_2_PFC = np.sum(s_task_2_1_from_t_2_2_PFC**2, axis = axis)
    sum_c_task_2_1_from_t_2_2_PFC = np.cumsum(abs(s_2_1_from_t_2_2_PFC))/n_neurons_PFC
    
    average_within_all_PFC_from_HP = np.mean([sum_c_task_1_2_PFC, sum_c_task_2_1_from_t_2_2_PFC], axis = 0)# average between task projections
    average_within_all_PFC_from_HP = average_within_all_PFC_from_HP/average_within_all_PFC_from_HP[-1] # get between task variance explained for cellular OR temporal modes 
    average_between_all_PFC_from_HP = np.mean([sum_c_task_2_1_from_t_1_2_PFC, sum_c_task_3_1_from_t_1_2_PFC], axis = 0)# average between task projections
    average_between_all_PFC_from_HP = average_between_all_PFC_from_HP/average_between_all_PFC_from_HP[-1] # get between task variance explained for cellular OR temporal modes 
    
    HP_from_PFC = np.mean([average_between_all_HP_from_PFC, average_within_all_HP_from_PFC], 0)
    PFC_from_HP = np.mean([average_between_all_PFC_from_HP, average_within_all_PFC_from_HP], 0)
    HP_ = np.mean([average_between_all_HP,average_within_all_HP],0)
    PFC_ = np.mean([average_between_all_PFC,average_within_all_PFC],0)
    
    plt.figure(figsize = (6,5))
    plt.plot(HP_*100, color = 'pink', label = 'CA1 from CA1')
    plt.plot(HP_from_PFC*100, linestyle = '--',color = 'pink', label = 'CA1 from PFC')
    plt.plot(PFC_*100, color = 'green', label = 'PFC from PFC')
    plt.plot(PFC_from_HP*100, linestyle = '--', color = 'green', label = 'PFC from CA1')
    plt.legend(); sns.despine()
    plt.xlabel('Number of temporal activity patterns (right singular vectors)')
    plt.ylabel('Variance Explained')
    plt.title('Temporal Modes')
  
def plot_example_patterns(data):    
    flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
    flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
    flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = extract_data(data, inds = np.arange(63))
    
    u_t1_2, s_t1_2, vh_t1_2 = np.linalg.svd(flattened_all_clusters_task_1_second_half, full_matrices = False)   
    u_t2_2, s_t2_2, vh_t2_2 = np.linalg.svd(flattened_all_clusters_task_2_second_half, full_matrices = False)    
    u_t3_2, s_t3_2, vh_t3_2 = np.linalg.svd(flattened_all_clusters_task_3_second_half, full_matrices = False) 
    plt.figure(figsize = (10,3)); vectr = 0
    for vec, temp in enumerate([vh_t1_2,vh_t2_2,vh_t3_2]):
        plt.subplot(1,3,vec+1)
        plt.plot(temp[vectr,:63]*(-1), label = 'A Reward', color = 'pink')
        plt.plot(temp[vectr,63:63*2]*(-1), label = 'A No Reward',color = 'pink', linestyle = '--')
        plt.plot(temp[vectr,63*2:63*3]*(-1), label = 'B Reward',color = 'green')
        plt.plot(temp[vectr,63*3:63*4]*(-1), label = 'B No Reward',color = 'green', linestyle = '--')
        plt.xticks([0,12.5, 25, 35, 42, 49, 63], ['-1','-0.5','Init', 'Ch','R', '+0.5', '+1']) 
        plt.title('Task '+ str(vec+1))

    sns.despine()

    plt.figure(figsize = (10,3)); 
    for vec, temp in enumerate([u_t1_2,u_t2_2,u_t3_2]):
        plt.subplot(1,3,vec+1)
        plt.plot(temp[:,vectr]*(-1), color = 'black')
        plt.ylabel('a.u.')
        plt.xlabel('neuron #')
        plt.title('Task '+ str(vec+1))
    sns.despine()

def permute(PFC, HP, diagonal = False, perm = 2, axis = 0, inds = np.arange(13,51), animal_perm = False):
    '''' Function to permute sessions for the svd analysis and find a distribution in differences between areas
    under the curve between PFC and HP'''

    all_subjects = np.concatenate((HP['DM'][0],PFC['DM'][0]),0)     
    all_subjects_firing = np.concatenate((HP['Data'][0],PFC['Data'][0]),0)     
    n_sessions = np.arange(len(HP['DM'][0])+len(PFC['DM'][0]))
    u_v_area_shuffle = []    
    if animal_perm == True:
        animals_PFC = [0,1,2,3]; animals_HP = [4,5,6]
        m, n = len(animals_PFC), len(animals_HP)
        all_subjects_DM, all_subjects_fr  = animal_exp_permute(HP,PFC)
        for indices_PFC in combinations(range(m + n), m):
            indices_HP = [i for i in range(m + n) if i not in indices_PFC]
            DM_PFC_perm = np.concatenate(np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_PFC)],0)
            firing_PFC_perm = np.concatenate(np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_PFC)],0)
                
            DM_HP_perm = np.concatenate(np.asarray(all_subjects_DM, dtype=object)[np.asarray(indices_HP)],0)
            firing_HP_perm = np.concatenate(np.asarray(all_subjects_fr, dtype=object)[np.asarray(indices_HP)],0)
            HP_shuffle = [DM_HP_perm,firing_HP_perm]; PFC_shuffle = [DM_PFC_perm,firing_PFC_perm]
            u_v_area = []
            for data in [HP_shuffle,PFC_shuffle]: # find area under curve for HP and PFC
                flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
                flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
                flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = extract_data(data, perm = True, inds = inds)

                trp, average_between_all,average_within_all = svd(flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
                flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
                flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, diagonal = diagonal, axis = axis)
                u_v_area.append(trp) 

            u_v_area_shuffle.append(u_v_area)
    else:
        for i in range(perm):
            np.random.shuffle(n_sessions) # Shuffle PFC/HP sessions
            indices_HP = n_sessions[:len(HP['DM'][0])];  indices_PFC = n_sessions[len(HP['DM'][0]):]

            # organise data for permutation test
            PFC_shuffle_dm = all_subjects[np.asarray(indices_PFC)]; HP_shuffle_dm = all_subjects[np.asarray(indices_HP)]
            PFC_shuffle_f = all_subjects_firing[np.asarray(indices_PFC)]; HP_shuffle_f = all_subjects_firing[np.asarray(indices_HP)]
            HP_shuffle = [HP_shuffle_dm,HP_shuffle_f];  PFC_shuffle = [PFC_shuffle_dm,PFC_shuffle_f]

            u_v_area = []
            for data in [HP_shuffle,PFC_shuffle]: # find area under curve for HP and PFC
                flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
                flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
                flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half = extract_data(data, perm = True, inds = inds)

                trp, average_between_all,average_within_all = svd(flattened_all_clusters_task_1_first_half, flattened_all_clusters_task_1_second_half,\
                flattened_all_clusters_task_2_first_half, flattened_all_clusters_task_2_second_half,\
                flattened_all_clusters_task_3_first_half,flattened_all_clusters_task_3_second_half, diagonal = diagonal, axis = axis)
                u_v_area.append(trp) 

            u_v_area_shuffle.append(u_v_area)

    return u_v_area_shuffle

def run_permutations_and_plot(HP, PFC, d = True, p = 1000, axis = 0, animal_perm = False, inds = np.arange(13,51)):
    '''Examine generalisation of singular vectors better between problems in PFC than CA1 by
    calculating the area between the dash and solid lines in earlier plots for CA1 and for PFC separately.
    ution).  Temporal singular vectors generalised equally well between problems in the two regions.'''
    
    u_v_area_shuffle = permute(PFC, HP, diagonal = d, perm = p, axis = axis, inds = inds, animal_perm = animal_perm) # permuted differences
    
    # real differences
    trp_hp,  average_between_hp, average_within_hp =  real_diff(HP,  diagonal = d,axis = axis, perm = False, cell = False, inds = inds)
    trp_pfc,  average_between_pfc, average_within_pfc =  real_diff(PFC,  diagonal = d,axis = axis, perm = False, cell = False, inds = inds)
 
    diff_uv  = []
    for i,ii in enumerate(u_v_area_shuffle):
        diff_uv.append(u_v_area_shuffle[i][0]- u_v_area_shuffle[i][1])
        
    uv_95 = np.percentile(diff_uv,95)*100; real_uv = trp_hp*100 - trp_pfc*100
    diff_uv = np.asarray(diff_uv)*100
    
    plt.hist(diff_uv, color = 'grey', alpha = 0.5)
    plt.vlines(real_uv, ymin = 0, ymax = max(np.histogram(diff_uv)[0]), color = 'pink')
    plt.vlines(uv_95, ymin = 0, ymax = max(np.histogram(diff_uv)[0]), color = 'black')
    plt.ylabel('Count')
    plt.xlabel('Permuted Differences')
    if d == False:
        plt.title('Cellular and temporal singular vectors')
    elif d == True and axis == 1:
        plt.title('Cellular vectors')
    else:
        plt.title('Temporal singular vectors')

    sns.despine()
     

def plot_supplementary(HP_data,PFC_data, inds, animal_perm = False, n_perms = 5):
    #Plot temporal modes 
    plt.figure(figsize = (6,5))
    cell_time_shuffle_vs_real(PFC_data, d = True, axis = 0, shuffle_cells = False, HP_ = 'False', inds = inds, shuffle_plot = False)
    cell_time_shuffle_vs_real(HP_data, d = True, axis = 0, shuffle_cells = False, HP_ = 'True', inds = inds, shuffle_plot = False)
    plt.xlabel('Number of temporal activity patterns (right singular vectors)')
    plt.ylabel('Variance Explained')
    plt.title('Temporal Modes')
    
    #Plot cellular modes 
    plt.figure(figsize = (6,5))
    cell_time_shuffle_vs_real(PFC_data, d = True, axis = 1, shuffle_cells = False, HP_ = 'False', inds = inds, shuffle_plot = False)
    cell_time_shuffle_vs_real(HP_data, d = True, axis = 1, shuffle_cells = False, HP_ = 'True', inds = inds, shuffle_plot = False)
    plt.xlabel('Number of cellular activity patterns (left singular vectors)')
    plt.ylabel('Variance Explained')
    plt.title('Cellular Modes')

    #Plot cellular and temporal modes with a temporal shuffle
    plt.figure(figsize = (6,5))
    cell_time_shuffle_vs_real(PFC_data, d = False, shuffle_cells = True, HP_ = 'False',inds = inds, shuffle_plot = False)
    cell_time_shuffle_vs_real(HP_data, d = False, shuffle_cells = True, HP_ = 'True', inds = inds, shuffle_plot = False)
    plt.xlabel('Number of cellular-temporal activity patterns (right and left singular vectors)')
    plt.ylabel('Cumulative Weight')
    plt.title('Cellular and Temporal Modes with Temporal Shuffle')    

    plt.figure(figsize = (8,2))
    plt.subplot(1,3,1)
    run_permutations_and_plot(HP_data, PFC_data, d = True, p = n_perms, axis = 0, animal_perm = animal_perm, inds = inds)
    plt.subplot(1,3,2)
    run_permutations_and_plot(HP_data, PFC_data, d = True, p = n_perms, axis = 1, animal_perm = animal_perm, inds = inds)
    plt.subplot(1,3,3)
    run_permutations_and_plot(HP_data, PFC_data, d = False, p = n_perms, animal_perm = animal_perm, inds = inds)




      
def plot_main_figure(HP, PFC, HP_dlc, PFC_dlc, n_perms = 5):
    # plot temporal modes 
    plt.figure(figsize = (6,5))
    cell_time_shuffle_vs_real(PFC_dlc, d = True, axis = 0, shuffle_cells = False, HP_ = 'False')
    cell_time_shuffle_vs_real(HP_dlc, d = True, axis = 0, shuffle_cells = False, HP_ = 'True')
    plt.xlabel('Number of temporal activity patterns (right singular vectors)')
    plt.ylabel('Variance Explained')
    plt.title('Temporal Modes')
    
    #Plot cellular modes 
    plt.figure(figsize = (6,5))
    cell_time_shuffle_vs_real(PFC_dlc, d = True, axis = 1, shuffle_cells = True, HP_ = 'False')
    cell_time_shuffle_vs_real(HP_dlc, d = True, axis = 1, shuffle_cells = True, HP_ = 'True')
    plt.xlabel('Number of cellular activity patterns (left singular vectors)')
    plt.ylabel('Variance Explained')
    plt.title('Cellular Modes')

    #Plot cellular and temporal modes with a temporal shuffle
    plt.figure(figsize = (6,5))
    cell_time_shuffle_vs_real(PFC_dlc, d = False, shuffle_cells = True, HP_ = 'False')
    cell_time_shuffle_vs_real(HP_dlc, d = False, shuffle_cells = True, HP_ = 'True')
    plt.xlabel('Number of cellular-temporal activity patterns (right and left singular vectors)')
    plt.ylabel('Cumulative Weight')
    plt.title('Cellular and Temporal Modes with Temporal Shuffle')    

    #Plot cellular and temporal modes with a cellular shuffle
    plt.figure(figsize = (6,5))
    cell_time_shuffle_vs_real(PFC_dlc, d = False, shuffle_cells = False, HP_ = 'False')
    cell_time_shuffle_vs_real(HP_dlc, d = False,  shuffle_cells = False, HP_ = 'True')
    plt.xlabel('Number of cellular-temporal activity patterns (right and left singular vectors)')
    plt.ylabel('Cumulative Weight')
    plt.title('Cellular and Temporal Modes with Cell Shuffle')    

    #Plot Fig 4H
    plt.figure(figsize = (8,2))
    plt.subplot(1,3,1)
    run_permutations_and_plot(HP_dlc, PFC_dlc, d = True, p = n_perms, axis = 0)
    # Plot Fig 4I
    plt.subplot(1,3,2)
    run_permutations_and_plot(HP_dlc, PFC_dlc, d = True, p = n_perms, axis = 1)
    # Plot Fig 4J
    plt.subplot(1,3,3)
    run_permutations_and_plot(HP_dlc, PFC_dlc, d = False, p = n_perms)
    plt.tight_layout()


  
    
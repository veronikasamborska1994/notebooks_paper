import numpy as np
import pylab as plt
import seaborn as sns
from palettable import wesanderson as wes
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
import palettable
from functions.helper_functions import task_ind


def heatplot_sort_init_B(data, ch = 1, title_plot = 'PFC A choice '): 
    ''' Code used to evaluate the relative effects of conflict between problem general and port specific representations.
    Sort neural activity in Layout Type 2 and use this sorting to plot the activity in Layout Type 3. 
    Choice B in Problem Layout 2 is Initiation in Problem Layout 3 but for comparison this function also plots A choices
    that were in the same physical port.
    inputs are data (PFC vs CA1), ch (A or B choice), title_area (PFC/CA1) for plotting''' 
    
    dm = data['DM'][0]; fr = data['Data'][0] # load data
    neurons = 0; n_neurons_cum = 0; 
    for s in fr:
        neurons += s.shape[1]
    ch_2_r = np.zeros((neurons,63));  ch_3_r = np.zeros((neurons,63)) # arrays to store firing rates on rewarded choices 
    ch_2_nr = np.zeros((neurons,63));  ch_3_nr = np.zeros((neurons,63))# arrays to store firing rates on non-rewarded choices

    for  s, sess in enumerate(fr): # loop through sessions
        # load data and trial information 
        DM = dm[s];  firing_rates = fr[s]
        n_trials, n_neurons, n_timepoints = firing_rates.shape; n_neurons_cum += n_neurons
        choices = DM[:,1]; reward = DM[:,2]; task =  DM[:,5]; a_pokes = DM[:,6]; b_pokes = DM[:,7] 
        taskid = task_ind(task, a_pokes, b_pokes)
        
        task_1_r = np.where((taskid == 2) & (choices == ch) & (reward == 1))[0] # Find indicies for task 2
        task_1_nr = np.where((taskid == 2) & (choices == ch & (reward == 0)))[0] # Find indicies for task 2
        task_2_r = np.where((taskid == 3) & (choices == ch)& (reward == 1))[0] # Find indicies for task 3
        task_2_nr = np.where((taskid == 3) & (choices == ch)& (reward == 0))[0] # Find indicies for task 3
        
        # store mean firing rates in each task 
        ch_2_r[n_neurons_cum-n_neurons:n_neurons_cum,:] =  np.mean(firing_rates[task_1_r,:, :],0) 
        ch_2_nr[n_neurons_cum-n_neurons:n_neurons_cum,:] = np.mean(firing_rates[task_1_nr,:, :],0)
        ch_3_r[n_neurons_cum-n_neurons:n_neurons_cum,:] = np.mean(firing_rates[task_2_r,:, :],0)
        ch_3_nr[n_neurons_cum-n_neurons:n_neurons_cum,:] = np.mean(firing_rates[task_2_nr,:, :],0)
            
    ch_2 = np.mean([ch_2_r,ch_2_nr],0);  ch_3 = np.mean([ch_3_r,ch_3_nr],0) # average across rewarded + non-rewarded 
    # normalise by maximum firing rates for heatplots
    ch_2_norm = ch_2/(np.tile(np.max(ch_2,1), [ch_2.shape[1],1]).T+1e-08)
    ch_3_norm = ch_3/(np.tile(np.max(ch_3,1), [ch_3.shape[1],1]).T+1e-08)
    # plotting params
    tick_ = [0, 12, 24, 35, 42, 49, 62]; tick_str = ['-1','-0.5','Init', 'Ch','R', '+0.5', '+1']
    plt.figure(figsize = (10,10)); cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap
    # within task 2 sorting
    peak_ind_2 = np.argmax(ch_2_norm,1) ;ordering_2 = np.argsort(peak_ind_2)
    activity_sorted_2 = ch_2_norm[ordering_2,:]
    plt.subplot(2,2,1); plt.imshow(activity_sorted_2, aspect = 'auto', cmap = cmap)  
    plt.xticks(tick_, tick_str); plt.title(title_plot + 'Sorted Activity in Layout 2')
    plt.ylabel(' # neuron')
    # use task 2 sorting to sort task 3
    activity_sorted_3_sort_by_2 = ch_3_norm[ordering_2,:]
    plt.subplot(2,2,2)
    plt.imshow(activity_sorted_3_sort_by_2, aspect ='auto', cmap = cmap)  
    plt.xticks(tick_, tick_str); plt.title(title_plot + "Activity in Layout 3 sorted \n using soring from Layout 2")
    plt.ylabel(' # neuron')

    # find cells with peaks around choices and initiations 
    init_selective_2 = np.where((np.argmax(ch_2_norm,1)> tick_[2]) & (np.argmax(ch_2_norm,1) < tick_[2]+5))[0]   
    choice_selective_3 = np.where((np.argmax(ch_3_norm,1)> tick_[3]) & (np.argmax(ch_3_norm,1) < tick_[3]+5))[0]
  
    task_2_init = np.mean(ch_2_norm[choice_selective_3],0)
    task_2_init_std = np.std(ch_2_norm[choice_selective_3],0)/np.sqrt(len(ch_2[choice_selective_3]))
    task_3_choice = np.mean(ch_3_norm[init_selective_2],0)
    task_3_choice_std = np.std(ch_3_norm[init_selective_2],0)/np.sqrt(len(ch_3[init_selective_2]))
    return task_2_init, task_2_init_std, task_3_choice, task_3_choice_std

    
def plot_Init_B_remapping(PFC, HP):
    task_2_init_A_PFC, task_2_init_std_A_PFC, task_3_choice_A_PFC, task_3_choice_std_A_PFC = heatplot_sort_init_B(PFC, ch = 1, title_plot = 'PFC A choice ')
    task_2_init_B_PFC, task_2_init_std_B_PFC, task_3_choice_B_PFC, task_3_choice_std_B_PFC = heatplot_sort_init_B(PFC, ch = 0, title_plot = 'PFC B choice ')
    task_2_init_A_HP, task_2_init_std_A_HP, task_3_choice_A_HP, task_3_choice_std_A_HP = heatplot_sort_init_B(HP, ch = 1, title_plot = 'CA1 A choice ')
    task_2_init_B_HP, task_2_init_std_B_HP, task_3_choice_B_HP, task_3_choice_std_B_HP = heatplot_sort_init_B(HP, ch = 0, title_plot = 'CA1 B choice ')
   
    mean_fr_pfc = [[task_3_choice_A_PFC,task_3_choice_B_PFC], [task_2_init_A_PFC,task_2_init_B_PFC]]
    mean_fr_ca1 = [[task_3_choice_A_HP,task_3_choice_B_HP], [task_2_init_A_HP, task_2_init_B_HP]]
    mean_st_pfc = [[task_3_choice_std_A_PFC,task_3_choice_std_B_PFC], [task_2_init_std_A_PFC,task_2_init_std_B_PFC]]
    mean_st_ca1 = [[task_3_choice_std_A_HP,task_3_choice_std_B_HP], [task_2_init_std_A_HP, task_2_init_std_B_HP]]
    titles = ['cells with Init Peak in T2 firing in T3 ', 'cells with Ch Peak in T3 firing in T2']
    ch_ab_str = [' \n choice A', ' \n choice B']; 
    tick_ = [0, 12, 24, 35, 42, 49, 62]; tick_str = ['-1','-0.5','Init', 'Ch','R', '+0.5', '+1']
    isl = wes.Royal2_5.mpl_colors;
    for i, mean in enumerate(mean_fr_pfc):
        plt.figure(figsize = (7,3))
        for ii,mn in enumerate(mean):
            plt.subplot(1,2,ii+1); plt.plot(mean[ii], color = isl[3], label = 'PFC')
            plt.fill_between(np.arange(len(mean[ii])), mean[ii] - mean_st_pfc[i][ii], mean[ii] + mean_st_pfc[i][ii], alpha = 0.1,color =  isl[3])
            plt.plot(mean_fr_ca1[i][ii], color = isl[0],label = 'CA1')
            plt.fill_between(np.arange(len(mean_fr_ca1[i][ii])), mean_fr_ca1[i][ii] - mean_st_ca1[i][ii], mean_fr_ca1[i][ii] + mean_st_ca1[i][ii], alpha = 0.1,color =  isl[0])
            plt.title(titles[i] + ch_ab_str[ii])
            plt.xticks(tick_, tick_str)
            plt.tight_layout()
            sns.despine()
    
        plt.legend()


def trials_surprise(data, compare_tasks = 1):    
    x,y = control_for_time(data, compare_tasks = compare_tasks) # only look at tasks that actually follow each other in time
    surprise_list_neurons_a_a = [];  surprise_list_neurons_b_b = []
    ind_pre_post = 20; ind_base = 10
    for  s, sess in enumerate(x):
        DM = y[s]; choices = DM[:,1]; b_pokes = DM[:,7]; a_pokes = DM[:,6]; task = DM[:,5]
        firing_rates_mean_time = x[s]; n_trials, n_neurons, n_time = firing_rates_mean_time.shape
        taskid = task_ind(task,a_pokes,b_pokes)

        if compare_tasks == 1:
            taskid_1 = 1; taskid_2 = 2
        elif compare_tasks == 2:
            taskid_1 = 2; taskid_2 = 3
        else:
            taskid_1 = 1; taskid_2 = 3
            
        task_1_a = np.where((taskid == taskid_1) & (choices == 1))[0] # Find indicies for task 1 A
        task_1_b = np.where((taskid == taskid_1) & (choices == 0))[0] # Find indicies for task 1 B
        task_2_a = np.where((taskid == taskid_2) & (choices == 1))[0] # Find indicies for task 2 A
        task_2_b = np.where((taskid == taskid_2) & (choices == 0))[0] # Find indicies for task 2 B
        # task 1 --> task 2
        # A choices pre task switch
        task_1_a_pre_baseline = task_1_a[-ind_pre_post:-ind_pre_post+ind_base] # baseline (first 10 of the last 20 trials)
        task_1_a_pre  = task_1_a[-ind_pre_post+ind_base:] # pre task switch 10 last trials  
        # B choices pre task switch
        task_1_b_pre_baseline = task_1_b[-ind_pre_post:-ind_pre_post+ind_base] # baseline (first 10 of the last 20 trials)
        task_1_b_pre  = task_1_b[-ind_pre_post+ind_base:] # pre task switch 10 last trials  
        # A and B choices post task switch
        task_2_b_post = task_2_b[:ind_pre_post] # first 20 trials post task switch Bs
        task_2_a_post = task_2_a[:ind_pre_post] # first 20 trials post task switch As
        # reverse task 2 --> task 1 
        task_2_a_pre_baseline_rev = task_2_a[ind_base:ind_pre_post]
        task_2_a_pre_rev  = task_2_a[:ind_base]; task_1_a_post =  task_1_a[-ind_pre_post:] 
        task_2_b_pre_baseline_rev = task_2_b[ind_base:ind_pre_post]
        task_2_b_pre_rev  = task_2_b[:ind_base]; task_1_b_post =  task_1_b[-ind_pre_post:] 
        
        for neuron in range(n_neurons):
            n_firing = firing_rates_mean_time[:,neuron, :].T  # firing rate of each neuron
            n_firing =  gaussian_filter1d(n_firing.astype(float),2,1); n_firing = n_firing.T; min_std = 2
            # task 1 --> task 2 
            # find means and standard deviations in the baseline
            task_1_mean_a = np.mean(n_firing[task_1_a_pre_baseline], axis = 0); task_1_std_a = np.std(n_firing[task_1_a_pre_baseline], axis = 0)     
            task_1_mean_b = np.mean(n_firing[task_1_b_pre_baseline], axis = 0); task_1_std_b = np.std(n_firing[task_1_b_pre_baseline], axis = 0)
            # calculate surprise on each trial before/after reversal
            a_within =  - norm.logpdf(n_firing[task_1_a_pre], task_1_mean_a, (task_1_std_a + min_std))
            b_within = - norm.logpdf(n_firing[task_1_b_pre], task_1_mean_b,(task_1_std_b + min_std))
            a_between = - norm.logpdf(n_firing[task_2_a_post], task_1_mean_a, (task_1_std_a + min_std))
            b_between = - norm.logpdf(n_firing[task_2_b_post], task_1_mean_b, (task_1_std_b + min_std))
            # task 2 --> task 1 (same as above otherwise)
            task_2_mean_a = np.mean(n_firing[task_2_a_pre_baseline_rev], axis = 0); task_2_std_a = np.std(n_firing[task_2_a_pre_baseline_rev], axis = 0)     
            task_2_mean_b = np.mean(n_firing[task_2_b_pre_baseline_rev], axis = 0); task_2_std_b = np.std(n_firing[task_2_b_pre_baseline_rev], axis = 0)
            a_within_rev =  - norm.logpdf(n_firing[task_2_a_pre_rev], task_2_mean_a, (task_2_std_a + min_std))
            b_within_rev = -  norm.logpdf(n_firing[task_2_b_pre_rev], task_2_mean_b, (task_2_std_b + min_std))
            a_between_rev = - norm.logpdf(n_firing[task_1_a_post], task_2_mean_a, (task_2_std_a + min_std))
            b_between_rev = - norm.logpdf(n_firing[task_1_b_post], task_2_mean_b, (task_2_std_b + min_std))
            # concatenate A within, A between trials and append to a list 
            surprise_array_a = np.concatenate([np.mean([a_within,a_within_rev],0), np.mean([a_between,a_between_rev],0)], axis = 0)                   
            surprise_array_b = np.concatenate([np.mean([b_within,b_within_rev],0), np.mean([b_between,b_between_rev],0)], axis = 0)                   
            surprise_list_neurons_a_a.append(surprise_array_a); surprise_list_neurons_b_b.append(surprise_array_b)
            
    surprise_list_neurons_a_a_all = np.mean(np.asarray(surprise_list_neurons_a_a), axis = 0)
    surprise_list_neurons_b_b_all = np.mean(np.asarray(surprise_list_neurons_b_b), axis = 0)
           
    return surprise_list_neurons_b_b_all, surprise_list_neurons_a_a_all


def control_for_time(data, compare_tasks = 1):
    y = data['DM'][0]; x = data['Data'][0]
    task_time_confound_data = []; task_time_confound_dm = []
    
    for  s, sess in enumerate(x):
        DM = y[s]; b_pokes = DM[:,7];  a_pokes = DM[:,6]; task = DM[:,5]; taskid = task_ind(task,a_pokes,b_pokes)
        if compare_tasks  == 1:
            taskid_1 = 1; taskid_2 = 2

        elif compare_tasks == 2:
            taskid_1 = 2; taskid_2 = 3
        else:
            taskid_1 = 1; taskid_2 = 3
        task_1_rev = np.where(taskid == taskid_1)[0][0]; task_2_rev = np.where(taskid == taskid_2)[0][-1]
        if task_2_rev+1 == task_1_rev:
            task_time_confound_data.append(sess)
            task_time_confound_dm.append(y[s])
            
        if compare_tasks == 3:
            task_1 = np.where(taskid == taskid_1)[0][-1]; task_2 = np.where(taskid == taskid_2)[0][0]
            if task_1+1 == task_2:
                task_time_confound_data.append(sess); task_time_confound_dm.append(y[s])
                
    return task_time_confound_data,task_time_confound_dm

def surprise_plot(HP, PFC):  
    mean_b_b_t1_t2_HP, mean_a_a_t1_t2_HP  = trials_surprise(HP, compare_tasks = 1)
    mean_b_b_t2_t3_HP, mean_a_a_t2_t3_HP = trials_surprise(HP, compare_tasks = 2)
    mean_b_b_t1_t3_HP, mean_a_a_t1_t3_HP = trials_surprise(HP,  compare_tasks = 3)

    mean_b_b_t1_t2_PFC, mean_a_a_t1_t2_PFC = trials_surprise(PFC, compare_tasks = 1)
    mean_b_b_t2_t3_PFC, mean_a_a_t2_t3_PFC  = trials_surprise(PFC,  compare_tasks = 2)
    mean_b_b_t1_t3_PFC, mean_a_a_t1_t3_PFC = trials_surprise(PFC,  compare_tasks = 3)
    
    As_CA1 = [mean_a_a_t1_t2_HP, mean_a_a_t1_t3_HP, mean_a_a_t2_t3_HP]
    Bs_CA1 = [mean_b_b_t1_t2_HP, mean_b_b_t1_t3_HP, mean_b_b_t2_t3_HP]
    As_PFC = [mean_a_a_t1_t2_PFC, mean_a_a_t1_t3_PFC, mean_a_a_t2_t3_PFC]
    Bs_PFC = [mean_b_b_t1_t2_PFC, mean_b_b_t1_t3_PFC, mean_b_b_t2_t3_PFC]
    
    v_min = np.min([np.min(mean_a_a_t1_t2_HP),np.min(mean_a_a_t1_t3_HP), np.min(mean_a_a_t2_t3_HP),\
                    np.min(mean_b_b_t1_t2_HP),np.min(mean_b_b_t1_t3_HP), np.min(mean_b_b_t2_t3_HP),\
                    np.min(mean_a_a_t1_t2_PFC),np.min(mean_a_a_t1_t3_PFC), np.min(mean_a_a_t2_t3_PFC),\
                    np.min(mean_b_b_t1_t2_PFC),np.min(mean_b_b_t1_t3_PFC), np.min(mean_b_b_t2_t3_PFC)])
    v_max = np.max([np.max(mean_a_a_t1_t2_HP),np.max(mean_a_a_t1_t3_HP), np.max(mean_a_a_t2_t3_HP),\
                    np.max(mean_b_b_t1_t2_HP),np.max(mean_b_b_t1_t3_HP), np.max(mean_b_b_t2_t3_HP),\
                    np.max(mean_a_a_t1_t2_PFC),np.max(mean_a_a_t1_t3_PFC), np.max(mean_a_a_t2_t3_PFC),\
                    np.max(mean_b_b_t1_t2_PFC),np.max(mean_b_b_t1_t3_PFC), np.max(mean_b_b_t2_t3_PFC)])
   
    cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap
    tick_ = [0, 12, 24, 35, 42, 49, 62]; tick_str = ['-1','-0.5','Init', 'Ch','R', '+0.5', '+1']
    x_tick = np.hstack([np.arange(3,10,4),10, np.arange(14,32,4)-1])
    xtick_str = np.hstack([-np.flip(np.arange(0,8,4))-4, 'Switch',np.arange(0,20,4)+4])
    fig, ax = plt.subplots(4, 3, figsize=(10,12))
    titles_str = [['CA1 A T1 T2','CA1 A T1 T3', 'CA1 A T2 T3'],\
                 ['CA1 B T1 T2','CA1 B T1 T3', 'CA1 B T2 T3'],\
                 ['PFC A T1 T2','PFC A T1 T3', 'PFC A T2 T3'],\
                 ['PFC B T1 T2','PFC B T1 T3', 'PFC B T2 T3']]
    
    for ab,surprise in enumerate([As_CA1,Bs_CA1,As_PFC,Bs_PFC]):
         for reg,surpr in enumerate(surprise):
            im = ax[ab,reg].imshow(surprise[reg].T,cmap = cmap, aspect = 'auto', vmin = v_min, vmax = v_max)
            ax[ab,reg].set_yticks(tick_)
            ax[ab,reg].set_xticks(x_tick)
            ax[ab,reg].set_xticklabels(xtick_str, rotation = 90)
            ax[ab,reg].set_yticklabels(tick_str)
            ax[ab,reg].set_title(titles_str[ab][reg])
            if ab == 3:
                ax[ab,reg].set_xlabel('Trial Before/After Task Switch')
        
    plt.tight_layout()
  
    fig.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.4])
    fig.colorbar(im, cax=cbar_ax, label = '-log(p(x))')
   
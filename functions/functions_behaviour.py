import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit as fit
from scipy import stats
from palettable import wesanderson as wes
import scipy
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.anova import AnovaRM
from functions.helper_functions import exp_mov_ave
from functions.helper_functions import exponential

def session_plot_moving_average(session):
    '''Plot reward probabilities and moving average of choices for a single session, Figure 1D'''
    
    # extract trial data (choices, rewards, states)
    block = session.trial_data['block']; choices = session.trial_data['choices']; n_trials = session.trial_data['n_trials']
    outcome = session.trial_data['outcomes']; state = session.trial_data['state'];  forced_trials = session.trial_data['forced_trial']
    forced_array = np.where(forced_trials == 1)[0]; forced_trials_sum = sum(forced_trials)

    # exclude without forced trials 
    choices = np.delete(choices, forced_array); block = np.delete(block, forced_array)
    n_trials = n_trials - forced_trials_sum; outcome =  np.delete(outcome, forced_array);  state =  np.delete(state, forced_array)
    exp_average = exp_mov_ave(choices, initValue = 0.5,tau = 8) # calculate moving average for choosing A choice
    
    plt.figure(figsize = [8,5])
    plt.subplot(4,1,1);  plt.plot(choices, color = 'black'); plt.xlim(1,n_trials) # plot choices
    plt.yticks([0,1],['B', 'A']);  plt.ylabel('Choice')
    plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False); 

    plt.subplot(4,1,2)
    plt.plot(outcome, color = 'pink'); plt.xlim(1,n_trials) # plot rewards
    plt.yticks([0,1],['NRew','Rew']);  plt.ylabel('Outcome')
    plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False); 

    plt.subplot(4,1,3);  plt.plot(state, color = 'lightblue') # plot state
    plt.xlim(1,n_trials); plt.yticks([0,1],['B','A']);  plt.ylabel('State')
    plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False); 

    plt.subplot(4,1,4)
    plt.axhline(y = 0.25, color = 'grey', linestyle = '--',  lw = 0.8) # plot thresholds
    plt.axhline(y = 0.75, color = 'grey',  linestyle = '--',  lw = 0.8,label = 'Threshold')
    plt.plot(exp_average, color = 'black') # plot exponential moving average
    plt.ylim(-0,1); plt.xlim(1,n_trials); plt.ylabel('Choice A \n Moving Average')
    plt.xlabel('Trials'); sns.despine(); plt.legend()
    

def trials_till_reversal_plot(experiment):
    '''Plot number of trials taken to reach a threshold for a reversal across tasks, Figure 1E'''
    
    tasks = 10;  reversals = 10; subject_IDs = experiment.subject_IDs; n_subjects = len(subject_IDs)  
    reversal_to_threshold = np.ones(shape=(n_subjects,tasks,reversals)) # subject, task number, reversal number
    reversal_to_threshold[:] = np.NaN  # fill it with NaNs 

    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_IDs = [subject_ID])
        task_number = 0 # reset current task number for each subject
        reversal_number = 0 # reset current rev number for each subject
        previous_session_config = 0 #reset config for each subject 
        trials_from_prev_session = 0
        for j, session in enumerate(subject_sessions):
            sessions_block = session.trial_data['block']
            n_trials = session.trial_data['n_trials']
            prt = (session.trial_data['pre-reversal trials'] > 0).astype(int) # find trials where count till block reversal begins
            
            # delete forced trials
            forced_trials = session.trial_data['forced_trial'] 
            forced_trials_sum = sum(forced_trials)
            forced_array = np.where(forced_trials == 1)[0]
            sessions_block = np.delete(sessions_block, forced_array)
            n_trials = n_trials -  forced_trials_sum
            prt = np.delete(prt, forced_array)
            
            threshold_crossing_trials = np.where((prt[1:] - prt[:-1]) == 1)[0] #find trial indicies where animals crossed a threshold
            block_transitions = sessions_block[1:] - sessions_block[:-1] # find actual block transitions
            reversal_trials = np.where(block_transitions == 1)[0] # indicies of block transitions
            
            configuration = session.trial_data['configuration_i'] # check configuration to detect if it changed from last session
            if configuration[0]!= previous_session_config:
                reversal_number = 0
                task_number += 1 
                trials_from_prev_session = 0
                previous_session_config = configuration[0]  
    
            if len(threshold_crossing_trials) > 0:
                  for i, crossing_trial in enumerate(threshold_crossing_trials): 
                    if reversal_number <= 9:
                        if i == 0: # first element in the threshold_crossing_trials_list 
                            reversal_to_threshold[n_subj, task_number-1, reversal_number] = crossing_trial+trials_from_prev_session
                            trials_from_prev_session = 0
                        elif i > 0: # for other than the first thershold in the session calculate the number of trials since block change
                            reversal_to_threshold[n_subj, task_number-1, reversal_number] = (crossing_trial-reversal_trials[i-1])
                        reversal_number += 1   
            
            # if animals crossed a threshold but session ended before the block change store trials from this session to add to the next
            if len(reversal_trials) != len(threshold_crossing_trials):
                trials_from_prev_session = n_trials - threshold_crossing_trials[i-1]
            # if no threshold crossing occured in the session store trials from this session to add to the next
            elif len(threshold_crossing_trials) == 0:
                trials_from_prev_session = n_trials - forced_trials_sum
   
    # organize data for ANOVAs
    rev = np.tile(np.arange(reversals),n_subjects*tasks); task_n = np.tile(np.repeat(np.arange(tasks),reversals),n_subjects); n_subj = np.repeat(np.arange(n_subjects),tasks*reversals)
    data = np.concatenate(reversal_to_threshold,0); data = np.concatenate(data,0)
     
    anova = {'Data':data,'Sub_id': n_subj,'cond1': task_n, 'cond2':rev}
    anova_pd = pd.DataFrame.from_dict(data = anova)
    aovrm = AnovaRM(anova_pd, depvar = 'Data',within=['cond1','cond2'], subject = 'Sub_id')
    res = aovrm.fit()
     
    # print stats
    print('Problem:' + ' '+ 'df = '+ ' '+ str(np.int(res.anova_table['Num DF'][0])) + ', ' + str(np.int(res.anova_table['Den DF'][0])), 'F' + ' = ' + str(np.round(res.anova_table['F Value'][0],3)),'p' + ' = ' + str(np.round(res.anova_table['Pr > F'][0], 3)));
    print('Reversal:' + ' '+ 'df = '+ ' '+ str(np.int(res.anova_table['Num DF'][1])) + ', ' + str(np.int(res.anova_table['Den DF'][1])), 'F' + ' = ' + str(np.round(res.anova_table['F Value'][1],3)),'p' + ' = ' + str(np.round(res.anova_table['Pr > F'][1], 3)));
   
    # Plot individual subjects
    mean_thre_subj = np.mean(reversal_to_threshold,axis = 2) 
    plt.figure(figsize=(5,5))      
    for i in mean_thre_subj:
        plt.scatter(np.arange(len(i)),i, color = 'grey')
        sns.despine()
        
    mean_threshold = np.mean(reversal_to_threshold, axis = 2) # mean across reversals
    mean_threshold_task = np.mean(mean_threshold, axis = 0) # mean across subjects
    std_threshold_per_task = np.std(mean_threshold, axis = 0) # std across subjects
    sample_size = np.sqrt(n_subjects)
    std_err = std_threshold_per_task/sample_size 
    
    # plot means and standard errors across tasks
    x_pos = np.arange(len(mean_threshold_task))
    plt.errorbar(x = x_pos, y = mean_threshold_task, yerr = std_err, alpha=0.8,  linestyle='None', marker='*', color = 'Black')    

    # plot trend line 
    z = np.polyfit(x_pos, mean_threshold_task,1) # linear fit
    p = np.poly1d(z)
    plt.plot(x_pos,p(x_pos),"--", color = 'grey', label = 'Trend Line')
    plt.xlabel("Problem Number")
    plt.ylabel("Number of Trials Till Threshold")
    plt.xticks(np.arange(10),np.arange(10)+1)
    
    # plot all reversals across all tasks averaged aross subjects 
    x = np.arange(reversals)
    std_proportion = np.std(reversal_to_threshold, axis = 0)
    std_err = std_proportion/sample_size
    mean_thre_subj_all_tasks = np.mean(reversal_to_threshold,axis = 0) 
    
    plt.figure(figsize = (30,5));colors = wes.Moonrise5_6.mpl_colors+wes.Royal2_5.mpl_colors
    for i in range(tasks): 
        plt.plot(i * reversals + x, mean_thre_subj_all_tasks[i],  color = colors[i], label = 'problem '+str(i+1))
        plt.fill_between(i * reversals + x, mean_thre_subj_all_tasks[i] - std_err[i], mean_thre_subj_all_tasks[i]+std_err[i], alpha=0.2 ,color = colors[i])
    plt.xticks(np.arange(100),np.tile(np.arange(10)+1,10)); plt.legend();sns.despine()
    plt.ylabel('Number of Trials Till Threshold');  plt.xlabel('Reversal Number')
   
    
    
def meta_learning_reversals(experiment):
    '''Plot Probability of choosing the new best option 
       (the choice that becomes good after the reversal on the last/first n (tr) trials around the reversal)
       split by the first problem and the last problem, Figure 1F'''
    
    # number of trials before and after switch
    tr = 10; subject_IDs = experiment.subject_IDs;  n_subjects = len(subject_IDs)  
    a_first = [];  a_last = [];  b_first = []; b_last = [] # arrays to store choices on the first last tasks

    for n_subj, subject_ID in enumerate(subject_IDs):
        states_block = []; choices_block = [] # lists for finding states (A/B good) and choices for each subj
        subject_sessions = experiment.get_sessions(subject_ID) # get subjects session
        
        for j, session in enumerate(subject_sessions):
            sessions_block = session.trial_data['block']
            forced_trials = session.trial_data['forced_trial']
            state = session.trial_data['state']
            choices = session.trial_data['choices'] 
            
            # delete forced trials
            forced_array = np.where(forced_trials == 1)[0]
            sessions_block = np.delete(sessions_block, forced_array)
            block_transitions = sessions_block[1:] - sessions_block[:-1] # block transition
            state = np.delete(state,forced_array)
            choices = np.delete(choices,forced_array)   
            
            state_around_block = []; choice_around_block = []
            block_transitions_id = np.where(block_transitions == 1)[0]
                                        
            if len(block_transitions_id) > 1: # if block transition happened during the session
                for b in block_transitions_id:
                    if b > tr and (b + tr) <= len(state): # only take blocks for which all 10 post and 10 pre reversal trials exist in a single session
                        state_around_block.append(state[b - tr: b + tr])
                        choice_around_block.append(choices[b - tr: b + tr])
    
            states_block.append(state_around_block)
            choices_block.append(choice_around_block)
            
        # if session had reversals with enough trials before + after 
        states_block_subj = [x for x in states_block if x != []]; states_block_subj = np.concatenate(states_block_subj,0)
        choices_block_subj = [x for x in choices_block if x != []]; choices_block_subj = np.concatenate(choices_block_subj,0)
        
        # was the block change from A to B or vice versa?
        change_from_a = np.where(states_block_subj[:,tr] == 1)[0]; change_from_b = np.where(states_block_subj[:,tr] == 0)[0]
        a_first.append(np.mean(choices_block_subj[change_from_a[:10]],0)) #first 10 reversals
        a_last.append(np.mean(choices_block_subj[change_from_a[-10:]],0)) #last 10 reversals

        b_first.append(np.mean(choices_block_subj[change_from_b[:10]],0)) #first 10 reversals
        b_last.append(np.mean(choices_block_subj[change_from_b[-10:]],0)) #last 10 reversals

    mean_a_f = np.mean(a_first,0); std_a_f = np.std(a_first,0)/np.sqrt(n_subjects); mean_a_l = np.mean(a_last,0); std_a_l = np.std(a_last,0)/np.sqrt(n_subjects)
    mean_b_f = np.mean(b_first,0); std_b_f = np.std(b_first,0)/np.sqrt(n_subjects); mean_b_l = np.mean(b_last,0); std_b_l = np.std(b_last,0)/np.sqrt(n_subjects)
    
    # fit slopes for choices after block change in the beginning vs late training
    slope_f = []; slope_l = []
    for i,ii in enumerate(a_first):
        after_rev_e = np.mean([1-a_first[i],b_first[i]],0)[tr:] # reverse A to B --> to the same sign as B to A (Correct vs Incorrect)
        after_rev_l = np.mean([1-a_last[i],b_last[i]],0)[tr:] # reverse A to B --> to the same sign as B to A (Correct vs Incorrect)
        sl_f = np.polyfit(np.arange(len(after_rev_e)), after_rev_e,1) # linear fit
        sl_l = np.polyfit(np.arange(len(after_rev_l)), after_rev_l,1) # linear fit
        slope_f.append(sl_f[0]); slope_l.append(sl_l[0]) # slopes for the first/last half of training for each subject
        
    # t-test to compare slopes across animals
    stat = stats.ttest_rel(slope_f,slope_l)
    print('Learning to Learn of the Block Structure ' +'df = ', (len(slope_f)*2)-1, ' t = ' + str(np.round(stat.statistic,2)), ' p = ' + str(np.round(stat.pvalue,3)))
     
    # reverse A to B --> to the same sign as B to A (Correct vs Incorrect)
    correct_first = np.mean([1-mean_a_f,mean_b_f],0);  correct_first_std  = np.mean([std_a_f,std_b_f],0)
    correct_last = np.mean([1-mean_a_l,mean_b_l],0); correct_last_std  = np.mean([std_a_l,std_b_l],0)
    plt.figure(figsize=(5,5))      
    plt.plot(correct_first, color = 'grey')
    plt.fill_between(np.arange(len(correct_first)), correct_first-correct_first_std, correct_first+correct_first_std, alpha=0.1,color ='grey', label = 'first 10 blocks')
    plt.plot(correct_last, color = 'black')
    plt.fill_between(np.arange(len(correct_last)), correct_last-correct_last_std, correct_last+correct_last_std, alpha=0.1,color ='black', label = 'last 10 blocks')
    plt.vlines([tr-0.5],ymin = 0, ymax = 1, color = 'pink', label = 'block switch')
    ticks = np.hstack([-np.flip(np.arange(tr)+1),np.arange(tr)+1])
    plt.xticks(np.arange(20),ticks)
    plt.xlabel('Trial # Before/After Reversal')
    plt.ylabel('Probability of Choosing New Best Option')
    plt.legend()
    sns.despine()

def out_of_sequence(experiment, first_task_only = False):
    '''Plot pokes per trial to a choice port that is no longer available because
    the subject had already chosen the other port, as a function of problem number, Figure 1E;
    
    Plot number of pokes per trial to a choice port that was no longer available as a function of reversal number
    on the first 5 problems and the last 5 problems during training, Figure 1H '''
   
    tasks = 10; reversals = 10; subject_IDs = experiment.subject_IDs; n_subjects = len(subject_IDs)  
    bad_pokes = np.zeros([n_subjects,tasks,reversals])  # subject, task number, reversal number
    bad_pokes[:] = np.NaN # fill it with NaNs 
    
    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_ID)
        previous_session_config = 0 # initiate previous configuration
        task_number = 0 # initiate task number
        all_sessions_wrong_ch = [] # get the list with only trials that were treated as trials in task programme
        all_reversals = []
        all_tasks = []

        for j, session in enumerate(subject_sessions): 
            trials = session.trial_data['n_trials']
            sessions_block = session.trial_data['block']
            forced_trials = session.trial_data['forced_trial']
            forced_array = np.where(forced_trials == 1)[0] 
            trials = trials - len(forced_array) 
            configuration = session.trial_data['configuration_i'] 
            poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
            poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
            
            # delete forced trials
            sessions_block = np.delete(sessions_block, forced_array)
            block_transitions = sessions_block[1:] - sessions_block[:-1] # Block transition
            reversal_trials = np.where(block_transitions == 1)[0]        
            
            # find events of interest
            events = [event.name for event in session.events if event.name in ['a_forced_state','b_forced_state','choice_state', 'init_trial','sound_b_no_reward', 'sound_b_reward','sound_a_no_reward','sound_a_reward',poke_A, poke_B]]
            session_wrong_choice = []
            
            # go through events list and find when how many times the animal poked in the other choice port after making a choice
            wrong_count = 0
            choice_state = False
            prev_choice = 'forced_trial'
            
            for event in events:
                if event == 'choice_state': 
                    session_wrong_choice.append(wrong_count) # append number of wrong pokes on prev trial
                    wrong_count = 0 # reset for each trial
                    choice_state = True # animal needs to choose
                
                elif event == poke_A :  
                    if choice_state == True:
                        prev_choice = 'Poke_A' # animal chose A during the choice phase
                        choice_state = False # indicate the animal made a choice 
                    elif choice_state == False and  prev_choice == 'Poke_B': # animal chose B previously
                        wrong_count += 1  # record if A poke was after an animal already chose B
                                
                elif event == poke_B : 
                    if choice_state == True:
                        prev_choice = 'Poke_B' # animal chose B during the choice phase
                        choice_state = False # indicate the animal made a choice 
                    elif choice_state == False and prev_choice == 'Poke_A': # animal chose A previously
                        wrong_count += 1 # record if A poke was after an animal already chose B
                
                elif event == 'a_forced_state' or event == 'b_forced_state':
                    prev_choice  = 'forced_state' # don't include forced trials
                      
            if j == 0: 
                all_sessions_wrong_ch = session_wrong_choice[:trials]  
            elif j > 0: 
                all_sessions_wrong_ch += session_wrong_choice[:trials] # appends wrong choice counts
                
            if configuration[0]!= previous_session_config: # if configuration changed make sure to reset everything
                reversal_number = 0 
                task_number += 1 
                previous_session_config = configuration[0]  
            
            for i in range(trials):
                for r in reversal_trials:
                    if i == r:
                        reversal_number += 1
                all_reversals.append(reversal_number) # record reversal in task number
                all_tasks.append(task_number)# record task number
                        
        reversals_np = np.asarray(all_reversals)+1 # make all reversals start at 1 (not 0)
        pokes_np = np.asarray(all_sessions_wrong_ch); tasks_np = np.asarray(all_tasks)
        
        for r in range(1,11):
            for t  in range(1,11):
                wrong_p = pokes_np[(tasks_np == t) & (reversals_np == r)] 
                mean_pokes = np.mean(wrong_p) # extract mean number of out of sequence pokes for each task and reversal  
                bad_pokes[n_subj,t-1,r-1] = mean_pokes  
    
    # organize data for ANOVAs
    rev = np.tile(np.arange(reversals),n_subjects*tasks)
    task_n = np.tile(np.repeat(np.arange(tasks),reversals),n_subjects)
    n_subj = np.repeat(np.arange(n_subjects),reversals*tasks)
    data = np.concatenate(bad_pokes,0)
    data = np.concatenate(data,0) 

    anova = {'Data':data,'Sub_id': n_subj,'cond1': task_n, 'cond2':rev}
    anova_pd = pd.DataFrame.from_dict(data = anova)
    aovrm = AnovaRM(anova_pd, depvar = 'Data', within=['cond1','cond2'], subject = 'Sub_id')
    res = aovrm.fit()
     
     # print stats
    print('Problem:' + ' '+ 'df = '+ ' '+ str(np.int(res.anova_table['Num DF'][0])) + ', ' + str(np.int(res.anova_table['Den DF'][0])), 'F' + ' = ' + str(np.round(res.anova_table['F Value'][0],3)),'p' + ' = ' + str(np.round(res.anova_table['Pr > F'][0], 3)));
    print('Reversal:' + ' '+ 'df = '+ ' '+ str(np.int(res.anova_table['Num DF'][1])) + ', ' + str(np.int(res.anova_table['Den DF'][1])), 'F' + ' = ' + str(np.round(res.anova_table['F Value'][1],3)),'p' + ' = ' + str(np.round(res.anova_table['Pr > F'][1], 3)));
    
    if first_task_only: 
        bad_pokes = bad_pokes[:,1:,:]
        tasks = 9
        rev = np.tile(np.arange(reversals),n_subjects*tasks)
        task_n = np.tile(np.repeat(np.arange(tasks),reversals),n_subjects)
        n_subj = np.repeat(np.arange(n_subjects),reversals*tasks)
        data = np.concatenate(bad_pokes,0)
        data = np.concatenate(data,0) 

        anova = {'Data':data,'Sub_id': n_subj,'cond1': task_n, 'cond2':rev}
        anova_pd = pd.DataFrame.from_dict(data = anova)
        aovrm = AnovaRM(anova_pd, depvar = 'Data', within=['cond1','cond2'], subject = 'Sub_id')
        res = aovrm.fit()

        # print stats
        print('Problem excluding 1st task:' + ' '+ 'df = '+ ' '+ str(np.int(res.anova_table['Num DF'][0])) + ', ' + str(np.int(res.anova_table['Den DF'][0])), 'F' + ' = ' + str(np.round(res.anova_table['F Value'][0],3)),'p' + ' = ' + str(np.round(res.anova_table['Pr > F'][0], 3)));
        print('Reversal excluding 1st task:' + ' '+ 'df = '+ ' '+ str(np.int(res.anova_table['Num DF'][1])) + ', ' + str(np.int(res.anova_table['Den DF'][1])), 'F' + ' = ' + str(np.round(res.anova_table['F Value'][1],3)),'p' + ' = ' + str(np.round(res.anova_table['Pr > F'][1], 3)));
       
    # plot individual subjects across tasks 
    mean_bad_pokes_rev = np.mean(bad_pokes,axis = 2)
    task_id = np.arange(10)+1
    plt.figure(figsize=(5,5))      
    for i in mean_bad_pokes_rev:
         plt.scatter(np.arange(len(i)),i, color = 'grey')
    sns.despine()
    plt.xticks(np.arange(10),task_id)
    plt.xlabel("Problem Number")
    plt.ylabel("Mean Number of Out of Sequence Pokes")
    
    # plot mean across subjects 
    mean_bad_pokes_subj = np.mean(mean_bad_pokes_rev, axis = 0)
    std_bad_pokes = np.std(mean_bad_pokes_subj, axis = 0)
    sample_size = np.sqrt(n_subjects)
    std_err = std_bad_pokes/sample_size
    x_pos = np.arange(len(mean_bad_pokes_subj))

    plt.errorbar(x = x_pos, y = mean_bad_pokes_subj, yerr = std_err, alpha=0.8,  linestyle='None', marker='*', color = 'Black')    
    z = np.polyfit(x_pos, mean_bad_pokes_subj,1) # linear fit
    p = np.poly1d(z)
    plt.plot(x_pos,p(x_pos),"--", color = 'grey', label = 'Trend Line')
    plt.xlabel("Problem")
    
    # plot all reversals across all tasks
    x = np.arange(reversals)
    bad_pokes_mean = np.mean(bad_pokes, axis = 0)
    bad_pokes_ser = (np.std(bad_pokes, axis = 0))/np.sqrt(n_subjects)
    
    plt.figure(figsize = (30,5))
    colors = wes.Moonrise5_6.mpl_colors+wes.Royal2_5.mpl_colors
    for i in range(tasks): 
        plt.plot(i * reversals + x, bad_pokes_mean[i],  color = colors[i], label = 'problem '+str(i+1))
        plt.fill_between(i * reversals + x, bad_pokes_mean[i] - bad_pokes_ser[i], bad_pokes_mean[i]+bad_pokes_ser[i], alpha=0.2 ,color = colors[i])
    plt.xticks(np.arange(100),np.tile(np.arange(10)+1,10))
    plt.legend()
    plt.ylabel('Number of Out of Sequence Pokes')
    plt.xlabel('Reversal Number')
    sns.despine()
    
    mice_out = np.mean(bad_pokes,0)
    slope_f = []
    slope_l = []

    for i in bad_pokes:
        f_s = np.mean(i[:5],0)
        l_s = np.mean(i[5:],0)
        x = np.arange(len(f_s))
        subj_p_f, cov_f = fit(exponential, x, f_s,  p0 = [1, 0, 0.5], bounds=(-2, [1, 1, 0.5]), maxfev = 1000000)
        subj_p_l, cov_l = fit(exponential, x, l_s,  p0 = [0, -1, 0.3], bounds=(-2, [1, 1, 0.5]), maxfev = 1000000)
        slope_f.append(np.log(abs(1/(subj_p_f[1]))))
        slope_l.append(np.log(abs(1/(subj_p_l[1]))))
       
    stat = stats.ttest_rel(slope_f,slope_l)
    print('Learning to Learn of the Trial Structure ' + 'df = ', (len(slope_f)*2)-1, ' t = ' + str(np.round(stat.statistic,2)), ' p = ' + str(np.round(stat.pvalue,3)))
    firstH = np.mean(mice_out[:5],0)
    lasttH = np.mean(mice_out[5:],0)
    
    firstH_std = np.std(np.mean(bad_pokes[:,:5,:],1),0)
    firstH_std = firstH_std/np.sqrt(n_subjects)
    
    lasttH_std = np.std(np.mean(bad_pokes[:,5:,:],1),0)
    lasttH_std = lasttH_std/np.sqrt(n_subjects)
    
    plt.figure(figsize=(5,5))      
    popt_exponential, pcov_exponential = fit(exponential, x, firstH, p0 = [1, 0, 0.5], maxfev = 100000)
    a, k, b = popt_exponential; y2 = exponential(x, a, k, b);   # test result
    plt.plot(x, y2, color='black')

    mice_first = np.mean(bad_pokes[:,:5,:],1)
    k = 0
    for i in mice_first:
        plt.scatter(np.arange(len(i)),i, color = 'black')
        sns.despine()
    
    popt_exponential_l, pcov_exponential_l = fit(exponential, x, lasttH, p0 = [0, -1, 0.3], maxfev = 100000)
    a, k, b = popt_exponential_l; y2_l = exponential(x, a, k, b)  # test result
    
    plt.plot(x, y2_l, color='pink')
    mice_last = np.mean(bad_pokes[:,5:,:],1)
    k = 0
    for i in mice_last:
        plt.scatter(np.arange(len(i)),i, color = 'pink')
        sns.despine()
    plt.xlabel("Problem")
    
    plt.errorbar(x = np.arange(len(firstH)), y = firstH, yerr = firstH_std, alpha=0.8,  linestyle='None', marker='*', color = 'Black', label = 'first 5 tasks')    
    plt.errorbar(x = np.arange(len(firstH)), y = lasttH, yerr = lasttH_std, alpha=0.8,  linestyle='None', marker='*', color = 'pink', label = 'last 5 tasks')     
    plt.xticks(np.arange(10), np.arange(10)+1)
    plt.legend(); plt.xlabel('Reversal Number'); plt.ylabel('Out of Sequence Pokes');  sns.despine()
    
   

def policy_training(experiment, subject_IDs ='all'):  
    '''Coefficients from a logistic regression predicting current choices using the history of previous choices (I),
    outcomes (not shown) and choice-outcome interactions, Figure 1I, 1J '''
    
    n = 11 # this number was selected based on significance testing after training 
    subject_IDs = experiment.subject_IDs; n_subjects = len(subject_IDs); coef_subj = [] # list to store regression coefficients
    
    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_IDs=[subject_ID])
        results_task = [];  results_session = [] # regression coefficients for tasks and sessions    
        for j, session in enumerate(subject_sessions):
            choices = session.trial_data['choices']
            all_sessions = len(subject_sessions)-1 # to store last session
            configuration = session.trial_data['configuration_i'] 
            if j == 0:
                previous_session_config = configuration[0] # if first session find poke configuration 
                
            elif configuration[0]!= previous_session_config: # check if configuration changed in this session 
                previous_session_config = configuration[0]  
                results_task.append(np.mean(results_session,0)) # append results if configuration changed
                
            if len(choices) > n*3: # check there is enough trials for regression in that session
                reward = session.trial_data['outcomes'] # rewards
                previous_rewards = scipy.linalg.toeplitz(reward, np.zeros((1,n)))[n-1:-1] # find rewards on n-back trials
                previous_choices = scipy.linalg.toeplitz(choices, np.zeros((1,n)))[n-1:-1] # find choices on n-back trials
                interactions = scipy.linalg.toeplitz((((choices-0.5)*(reward-0.5))*2),np.zeros((1,n)))[n-1:-1] #interactions rewards x choice
                choices_current = (choices[n:]) # current choices need to start at nth trial
                ones = np.ones(len(interactions)).reshape(len(interactions),1) # add constant
                X = (np.hstack([previous_rewards,previous_choices,interactions,ones])) # create design matrix
                model = LogisticRegression(fit_intercept = False) 
                results = model.fit(X,choices_current) # fit logistic regression predicting current choice based on history of behaviour
                results_session.append(results.coef_[0]) # store coefficients
            
                if j == all_sessions:
                    previous_session_config = configuration[0]  
                    results_task.append(np.mean(results_session,0)) # make sure to store last session
                
        coef_subj.append(results_task) # subject coefficients for each task in training 
        
    mean_t = np.mean(coef_subj,0)
    sqrt = np.std(coef_subj,0)/np.sqrt(n_subjects)
    
    isl = wes.Royal2_5.mpl_colors
    
    # plot choice and choice x reward interactions coefficients during training in each task 
    xticks_ = []; j = 0
    for i,ii in enumerate(mean_t):
        j += 1
        xticks_.append((np.arange(len(ii))[:n]+j*n)[0])
        plt.figure(2)
        plt.plot(np.arange(len(ii))[:n]+j*n, ii[n:n*2], color = isl[3])
        plt.fill_between(np.arange(len(ii))[:n]+j*n, ii[n:n*2]+sqrt[i][n:n*2], ii[n:n*2]- sqrt[i][n:n*2],alpha = 0.2, color = isl[3])
        
        plt.figure(3)
        plt.plot(np.arange(len(ii))[:n]+j*n, ii[n*2:-1], color = isl[4])
        plt.fill_between(np.arange(len(ii))[:n]+j*n, ii[n*2:-1]+sqrt[i][n*2:-1], ii[n*2:-1]- sqrt[i][n*2:-1],alpha = 0.2, color = isl[4])

    
    # add ticks to plots
    plt.figure(2)
    plt.xticks(xticks_,np.arange(10)+1)
    sns.despine()
    plt.xlabel('Problem')
    plt.ylabel('Coefficient')
    plt.tight_layout()
    plt.title('Choice History')
    
    plt.figure(3)
    plt.xticks(xticks_,np.arange(10)+1) 
    sns.despine()
    plt.xlabel('Problem')
    plt.ylabel('Coefficient')
    plt.tight_layout()
    plt.title('History of Outcomes x Choices')
  
    return coef_subj

       
   
def plot_policy_during_training(experiment):  
    '''Coefficients from a logistic regression predicting current choices using the history of previous choices (I),
    outcomes (not shown) and choice-outcome interactions, Figure 1I, 1J '''
    
    n = 11 # this number was selected based on significance testing after training 
    coef_subj =  policy_training(experiment, subject_IDs ='all')
    coef_subj = np.asarray(coef_subj)
    choices = coef_subj[:,:,n:n*2]; choices_X_reward = coef_subj[:,:,n*2:-1]
    
    _1_back_ch = choices[:, :, 0];  _other_back_ch = np.mean(choices[:, :,1:],2)
    _1_back_rew_ch = choices_X_reward[:,:,0]; _other_back_rew_ch = np.mean(choices_X_reward[:,:,1:],2)
        
    n_tasks = 10; n_subj = 9
    subject_id = np.tile(np.arange(n_tasks), n_subj)
    fraction_id = np.zeros(n_subj*n_tasks)
    k = 0 
    for n in range(10):
        fraction_id[n*n_subj:n*n_subj+n_subj] = k # fraction in training for later ANOVAs
        k+=1
    
    coef_to_test = [_1_back_ch,_other_back_ch,_1_back_rew_ch, _other_back_rew_ch]
    coef_names = ['Previous Choice:','History of Choices:', 'Previous Outcome x Choice:','History of Outcomes x Choices:']
     
    # caculate ANOVAs for each of the coefficients
    p_vals_list = []
    for c, coef in enumerate(coef_to_test):
        coef = np.concatenate(coef.T,0)
        coef = {'Data':coef,'Sub_id': subject_id,'cond': fraction_id}
        coef = pd.DataFrame.from_dict(data = coef)
        aovrm = AnovaRM(coef, depvar = 'Data',subject = 'Sub_id', within=['cond'])
        res = aovrm.fit()
        coef = res.anova_table
        p_val = np.round(res.anova_table['Pr > F'][0],2)
        print(coef_names[c] + ' '+ 'df = '+ ' '+ str(res.anova_table['Num DF'][0]) + ' ' + str(res.anova_table['Den DF'][0]), 'F = ' + ' ' + str(np.round(res.anova_table['F Value'][0],3)),'p = ' + ' '+ str(np.round(res.anova_table['Pr > F'][0],3)));
        p_vals_list.append(p_val) 

       
    # plot coefficients during each task in training 
    _1_back_ch = np.mean(choices[:, :, 0],0); _1_back_ch_er = np.std(choices[:, :, 0],0)/np.sqrt(9)
    _other_back_ch =  np.mean(np.mean(choices[:, :,1:],2),0);_other_back_ch_err =  np.std(np.mean(choices[:, :, 1:],2),0)/np.sqrt(9)
    
    _1_back_rew_ch = np.mean(choices_X_reward[:,:,0],0);_1_back_rew_ch_err =  np.std(choices_X_reward[:,:,0],0)/np.sqrt(9)
    _all_back_rew_ch = np.mean(np.mean(choices_X_reward[:,:,1:],2),0); _all_back_rew_ch_err =  np.std(np.mean(choices_X_reward[:,:,1:],2),0)/np.sqrt(9)
    
    coefs_to_plot = [_1_back_ch,_other_back_ch,_1_back_rew_ch,_all_back_rew_ch]
    coefs_err = [_1_back_ch_er,_other_back_ch_err,_1_back_rew_ch_err,_all_back_rew_ch_err]
    isl = wes.Royal2_5.mpl_colors

    plt.figure(figsize = (13,4))
    for c,coeff in enumerate(coefs_to_plot):
        plt.subplot(1,4,c+1)
        cl = isl[3]
        if c > 2:
            cl = isl[4]
        plt.errorbar(np.arange(len(coeff)), coeff, yerr=coefs_err[c], fmt='o', color = cl)
        plt.annotate(p_vals_list[c], xy = (10,np.max(coeff)+0.01))
        plt.xlim(-1,10)
        plt.title(coef_names[c])
        plt.xticks(np.arange(10),np.arange(10)+1)
        plt.xlabel('Problem')
        plt.ylabel('Coefficient')

    sns.despine()
    plt.tight_layout()
    
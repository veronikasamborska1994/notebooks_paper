import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import palettable 
from palettable import wesanderson as wes
from functions.helper_functions import task_ind


def plot_neuron(data, session_id = 0, neuron_ID = 0):  
    ''' Simple function to plot firing rates in different tasks and conditions'''
    
    dm = data['DM'][0];  fr = data['Data'][0];  _xticks = [0,12, 24, 35, 42, 49, 63]; isl = wes.Royal2_5.mpl_colors
    x_pos = np.array([149, 68, 231, 0, 149, 298, 68, 231, 149])
    y_pos = -1 * np.array([0, 25, 25, 62, 62, 62, 100, 100, 124])
    for s,session in enumerate(dm):     
        if s == session_id:
            firing_rate = fr[s]; DM = dm[s]; n_trials, n_neurons, n_times = firing_rate.shape
            
            choices = DM[:,1]; b_pokes = DM[:,7]; a_pokes = DM[:,6]; reward = DM[:,2]; task = DM[:,5]; block = DM[:,0]; i_pk = DM[:,8]
            _t_2 = np.where(task == 2)[0];_t_3 = np.where(task == 3)[0]
            taskid = task_ind(task, a_pokes, b_pokes);
            task_arrays = np.zeros(n_trials); task_arrays[:_t_2[0]] = 1;  task_arrays[_t_2[0]:_t_3[0]] = 2;  task_arrays[_t_3[0]:] = 3
            a_all = a_pokes[0]; b_1 = b_pokes[np.where(taskid == 1)[0][0]];  b_2 = b_pokes[np.where(taskid == 2)[0][0]]; b_3 = b_pokes[np.where(taskid == 3)[0][0]]
            i1 = i_pk[np.where(taskid == 1)[0][0]]; i2 = i_pk[np.where(taskid == 2)[0][0]]; i3 = i_pk[np.where(taskid == 3)[0][0]]

            a_rew_1_f = np.mean(firing_rate[np.where((taskid ==1)& (choices ==1) & (reward == 1))[0]],0)
            a_nrew_1_f = np.mean(firing_rate[np.where((taskid ==1)& (choices ==1) & (reward == 0))[0]],0)
            
            a_rew_2_f = np.mean(firing_rate[np.where((taskid ==2)& (choices ==1) & (reward == 1))[0]],0)
            a_nrew_2_f = np.mean(firing_rate[np.where((taskid ==2)& (choices ==1) & (reward == 0))[0]],0)
        
            a_rew_3_f = np.mean(firing_rate[np.where((taskid ==3)& (choices ==1) & (reward == 1))[0]],0)
            a_nrew_3_f = np.mean(firing_rate[np.where((taskid ==3)& (choices ==1) & (reward == 0))[0]],0)
        
        
            b_rew_1_f = np.mean(firing_rate[np.where((taskid ==1)& (choices == 0) & (reward == 1))[0]],0)
            b_nrew_1_f = np.mean(firing_rate[np.where((taskid ==1)& (choices == 0) & (reward == 0))[0]],0)
            
            b_rew_2_f = np.mean(firing_rate[np.where((taskid ==2)& (choices == 0) & (reward == 1))[0]],0)
            b_nrew_2_f = np.mean(firing_rate[np.where((taskid ==2)& (choices == 0) & (reward == 0))[0]],0)
        
            b_rew_3_f = np.mean(firing_rate[np.where((taskid ==3)& (choices == 0) & (reward == 1))[0]],0)
            b_nrew_3_f = np.mean(firing_rate[np.where((taskid ==3)& (choices == 0) & (reward == 0))[0]],0)
            
            
            a_rew_1_std = np.std(firing_rate[np.where((taskid ==1)& (choices ==1) & (reward == 1))[0]],0)/np.sqrt(len(np.where((taskid ==1)& (choices == 1) & (reward == 1))[0]))
            a_nrew_1_std = np.std(firing_rate[np.where((taskid ==1)& (choices ==1) & (reward == 0))[0]],0)/np.sqrt(len(np.where((taskid ==1)& (choices == 1) & (reward == 0))[0]))
            
            a_rew_2_std = np.std(firing_rate[np.where((taskid ==2)& (choices ==1) & (reward == 1))[0]],0)/np.sqrt(len(np.where((taskid ==2)& (choices == 1) & (reward == 1))[0]))
            a_nrew_2_std = np.std(firing_rate[np.where((taskid ==2)& (choices ==1) & (reward == 0))[0]],0)/np.sqrt(len(np.where((taskid ==2)& (choices == 1) & (reward ==0 ))[0]))
        
            a_rew_3_std = np.std(firing_rate[np.where((taskid ==3)& (choices ==1) & (reward == 1))[0]],0)/np.sqrt(len(np.where((taskid ==3)& (choices == 1) & (reward == 1))[0]))
            a_nrew_3_std = np.std(firing_rate[np.where((taskid ==3)& (choices ==1) & (reward == 0))[0]],0)/np.sqrt(len(np.where((taskid ==3)& (choices == 1) & (reward == 0))[0]))
        
        
            b_rew_1_std = np.std(firing_rate[np.where((taskid ==1)& (choices == 0) & (reward == 1))[0]],0)/np.sqrt(len(np.where((taskid ==1)& (choices == 0) & (reward == 1))[0]))
            b_nrew_1_std = np.std(firing_rate[np.where((taskid ==1)& (choices == 0) & (reward == 0))[0]],0)/np.sqrt(len(np.where((taskid ==1)& (choices == 0) & (reward == 0))[0]))
            
            b_rew_2_std = np.std(firing_rate[np.where((taskid ==2)& (choices == 0) & (reward == 1))[0]],0)/np.sqrt(len(np.where((taskid ==2)& (choices == 0) & (reward == 1))[0]))
            b_nrew_2_std = np.std(firing_rate[np.where((taskid ==2)& (choices == 0) & (reward == 0))[0]],0)/np.sqrt(len(np.where((taskid ==2)& (choices == 0) & (reward == 0))[0]))
        
            b_rew_3_std = np.std(firing_rate[np.where((taskid ==3)& (choices == 0) & (reward == 1))[0]],0)/np.sqrt(len(np.where((taskid ==3)& (choices == 0) & (reward == 1))[0]))
            b_nrew_3_std = np.std(firing_rate[np.where((taskid ==3)& (choices == 0) & (reward == 0))[0]],0)/np.sqrt(len(np.where((taskid ==3)& (choices == 0) & (reward == 0))[0]))
        
                        
            for neuron in range(n_neurons):
                if neuron == neuron_ID:
                    trial_length = a_nrew_1_f.shape[1]
                    plt.figure(figsize = (20,5))
                    tasks_fr = [[a_rew_1_f,a_nrew_1_f, b_rew_1_f, b_nrew_1_f], [a_rew_2_f,a_nrew_2_f, b_rew_2_f, b_nrew_2_f], [a_rew_3_f,a_nrew_3_f, b_rew_3_f, b_nrew_3_f]]
                    tasks_fr_std = [[a_rew_1_std,a_nrew_1_std, b_rew_1_std, b_nrew_1_std], [a_rew_2_std, a_nrew_2_std, b_rew_2_std, b_nrew_2_std], [a_rew_3_std,a_nrew_3_std, b_rew_3_std, b_nrew_3_std]]
                    tasks_d = ['Task Layout 1', 'Task Layout 2', 'Task Layout 3']
                    for sub in [1,2,3]:
                        plt.subplot(2,5,sub)
                        plt.title(tasks_d[sub-1])

                        plt.plot(tasks_fr[sub-1][0][neuron], label ='A Rew',  color = 'green')
                        plt.fill_between(np.arange(trial_length), tasks_fr[sub-1][0][neuron]+a_rew_1_std[neuron],\
                                         tasks_fr[sub-1][0][neuron]-a_rew_1_std[neuron],alpha = 0.2,color = 'green')
        
                        plt.plot(tasks_fr[sub-1][1][neuron], linestyle = ':', label ='A NR',  color = 'green')
                        plt.fill_between(np.arange(trial_length), tasks_fr[sub-1][1][neuron]+tasks_fr_std[sub-1][1][neuron],\
                                         tasks_fr[sub-1][1][neuron]-tasks_fr_std[sub-1][1][neuron],alpha = 0.2, color = 'green')
        
                        plt.plot(tasks_fr[sub-1][2][neuron], color = isl[0], label ='B Rew')     
                        plt.fill_between(np.arange(trial_length), tasks_fr[sub-1][2][neuron]+tasks_fr_std[sub-1][2][neuron],\
                                         tasks_fr[sub-1][2][neuron]-tasks_fr_std[sub-1][2][neuron],alpha = 0.2, color =isl[0])
        
                        plt.plot(tasks_fr[sub-1][3][neuron], color = isl[0], linestyle = ':',  label ='B NR')     
                        plt.fill_between(np.arange(trial_length),tasks_fr[sub-1][3][neuron]+tasks_fr_std[sub-1][3][neuron],\
                                        tasks_fr[sub-1][3][neuron]-tasks_fr_std[sub-1][3][neuron],alpha = 0.2, color =isl[0])
                        plt.xticks(_xticks, ['-1','-0.5','Init', 'Ch','R', '+0.5', '+1'], fontsize = 7)
                        plt.ylabel('firing rate', fontsize = 7)
       

                    sns.despine()
                    plt.legend(fontsize = 7)
    
                    # heatmap  
                    plt.subplot(2,5,4); cmap =  palettable.scientific.sequential.Acton_3.mpl_colormap
                    heatplot = firing_rate[:,neuron,:]
                    heatplot = heatplot/(np.tile(np.max(heatplot,1), [heatplot.shape[1],1]).T+1e-08)
          
                    plt.imshow(heatplot,cmap= cmap,aspect = 'auto')
                    plt.xticks(_xticks, ['-1','-0.5','Init', 'Ch','R', '+0.5', '+1'],  fontsize = 7)
                    plt.ylabel('trial number',  fontsize = 7)
                    sns.despine()

                    plt.subplot(2,5,5)
                    plt.scatter(block, np.arange(len(block)), color = 'black', s = 1)
                    plt.scatter(task_arrays+1, np.arange(len(task_arrays)), color = 'red', s = 1)
                    plt.xticks([0,1,2,3,4], ['state A','state B', 'task 1', 'task 2', 'task 3'], rotation = 90, fontsize = 7)
                    

                    plt.yticks([])
                    sns.despine();plt.gca().invert_yaxis()
                    plt.tight_layout()

                    
                    ports_1 = [int(a_all), int(b_1), int(i1)]; ports_2 = [int(a_all), int(b_2), int(i2)]; ports_3 = [int(a_all), int(b_3), int(i3)]
                  
                    for p,ports in enumerate([ports_1,ports_2,ports_3]):
                        plt.subplot(2,5,6+p)
                        plt.scatter(x_pos, y_pos, s=100, c='grey', alpha=0.3)
                        plt.scatter(x_pos[(ports[0]-1)], y_pos[(ports[0]-1)], s=100, c ='green', alpha=0.3)
                        plt.scatter(x_pos[(ports[1]-1)], y_pos[(ports[1]-1)], s=100, c ='pink', alpha=0.3)
                        plt.scatter(x_pos[(ports[2]-1)], y_pos[(ports[2]-1)], s=100, c ='black', alpha=0.3)
                        plt.xlim(-50,600)
                        plt.ylim(-200,100)

                        sns.despine(left=True, right=True, top=True, bottom=True)
                        plt.xticks([])
                        plt.yticks([]);                    
                      
    



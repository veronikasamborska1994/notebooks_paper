import numpy as np

def RSA_physical_rdm():
    a = np.repeat(np.array([bool(i & 0) for i in range(15)]).reshape(1,15),6, axis = 0)
    i1_i3 = np.repeat(np.array([bool(i & 0) for i in range(15)]).reshape(1,15),2, axis = 0)
    i2_b3 = np.repeat(np.array([bool(i & 0) for i in range(15)]).reshape(1,15),3, axis = 0)
    b1 = np.repeat(np.array([bool(i & 0) for i in range(15)]).reshape(1,15),2, axis = 0)
    b2 = np.repeat(np.array([bool(i & 0) for i in range(15)]).reshape(1,15),2, axis = 0)
    a[:,0:6] = True; i1_i3[:,6:8] = True; i2_b3[:,8:11] = True; b1[:,11:13] = True;  b2[:,13:15] = True

    physical_rsa = np.vstack([a,i1_i3, i2_b3, b1, b2])
    return physical_rsa



def RSA_a_b_rdm():
    a = np.repeat(np.array([bool(i & 0) for i in range(15)]).reshape(1,15),6, axis = 0)
    i1_i3_i2 = np.repeat(np.array([bool(i & 0) for i in range(15)]).reshape(1,15),3, axis = 0)
    b = np.repeat(np.array([bool(i & 0) for i in range(15)]).reshape(1,15),6, axis = 0)
    a[:,0:6] = True;  b[:,9:15] = True;
    
    choice_ab_rsa = np.vstack([a,i1_i3_i2,b])
    return choice_ab_rsa
   


def reward_rdm(): 
    a = np.repeat(np.array([bool(i & 0) for i in range(15)]).reshape(1,15),6, axis = 0)
    for i in [0,2,4]:
        a[i,0] = True;  a[i,2] = True;  a[i,4] = True; a[i,9] = True; a[i,11] = True; a[i,13] = True;
    for i in [1,3,5]:
        a[i,1] = True;  a[i,3] = True;  a[i,5] = True;  a[i,10] = True; a[i,12] = True; a[i,14] = True;   
    i1_i3_i2 = np.repeat(np.array([bool(i & 0) for i in range(15)]).reshape(1,15),3, axis = 0)
    
    reward_no_reward = np.vstack([a,i1_i3_i2,a])
    return reward_no_reward


def reward_choice_space():
    reward_no_reward = reward_rdm()
    choice_ab_rsa = RSA_a_b_rdm()
    
    reward_at_choices = reward_no_reward & choice_ab_rsa 
    return reward_at_choices


def choice_vs_initiation():
    a = np.repeat(np.array([bool(i & 0) for i in range(15)]).reshape(1,15),6, axis = 0)
    i1_i3_i2 = np.repeat(np.array([bool(i & 0) for i in range(15)]).reshape(1,15),3, axis = 0)
    b = np.repeat(np.array([bool(i & 0) for i in range(15)]).reshape(1,15),6, axis = 0)
    a[:,0:6] = True; a[:,9:15] = True; b[:,0:6] = True; b[:,9:15] = True; i1_i3_i2[:,6:9] = True
    
    choice_initiation_rsa = np.vstack([a,i1_i3_i2,b])
    return choice_initiation_rsa


def a_bs_task_specific():
    a = np.repeat(np.array([bool(i & 0) for i in range(15)]).reshape(1,15),6, axis = 0)
    i1_i3_i2 = np.repeat(np.array([bool(i & 0) for i in range(15)]).reshape(1,15),3, axis = 0)
    for i in range(len(a)):
        if i%2 == 0:
            a[i,i:i+2] = True;
        else:
            a[i,i-1:i+1] = True;
            
    b = np.flip(a)
    a_bs_task_specific_rsa = np.vstack([a,i1_i3_i2,b])
    return a_bs_task_specific_rsa
        

import numpy as np

def exp_mov_ave(data, tau = 8, initValue = 0.5, alpha = None):
    '''Exponential Moving average for 1d data.  The decay of the exponential can 
    either be specified with a time constant tau or a learning rate alpha.'''
    if not alpha: alpha = 1. - np.exp(-1./tau)
    mov_ave = np.zeros(np.size(data)+1)
    mov_ave[0] = initValue
    for i, x in enumerate(data):
        mov_ave[i+1] = (1.-alpha)*mov_ave[i] + alpha*x 
    return mov_ave[1:]
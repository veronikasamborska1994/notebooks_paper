import numpy as np
from sklearn.linear_model import LinearRegression

def _CPD(X,y):
    '''Helper function to evaluate coefficient of partial determination for each predictor in X'''
    ols = LinearRegression(copy_X = True,fit_intercept= False)
    ols.fit(X,y)
    sse = np.sum((ols.predict(X) - y)**2, axis=0)
    cpd = np.zeros([y.shape[1],X.shape[1]])
    for i in range(X.shape[1]):
        X_i = np.delete(X,i,axis=1)
        ols.fit(X_i,y)
        sse_X_i = np.sum((ols.predict(X_i) - y)**2, axis=0)
        if len(np.where(sse_X_i== 0)[0]) > 0:
            cpd[:,i] = np.NaN
        else:
            cpd[:,i]=(sse_X_i-sse)/sse_X_i
    return cpd

def _cpd(X,y):
    '''Evaluate coefficient of partial determination for each predictor in X'''
    pdes = np.linalg.pinv(X)
    pe = np.matmul(pdes,y)
    Y_predict = np.matmul(X,pe)
    sse = np.sum((Y_predict - y)**2, axis=0)
    cpd = np.zeros([X.shape[1]])
    for i in range(X.shape[1]):
        X_i = np.delete(X,i,axis=1)
        pdes_i = np.linalg.pinv(X_i)
        pe_i = np.matmul(pdes_i,y)
        Y_predict_i = np.matmul(X_i,pe_i)
        sse_X_i = np.sum((Y_predict_i- y)**2, axis=0)
        cpd[i]=(sse_X_i-sse)/sse_X_i
    return cpd


def exp_mov_ave(data, tau = 8, initValue = 0.5, alpha = None):
    'Exponential moving average for 1d data.  The decay of the exponential can either be specified with a time constant tau or a learning rate alpha.'
    if not alpha: alpha = 1. - np.exp(-1./tau)
    mov_ave = np.zeros(np.size(data)+1)
    mov_ave[0] = initValue
    for i, x in enumerate(data):
        mov_ave[i+1] = (1.-alpha)*mov_ave[i] + alpha*x 
    return mov_ave[1:]



def task_ind(task, a_pokes, b_pokes):
    """ Create Task IDs for that are consistent: in Task 1 A and B at left right extremes, in Task 2 B is one of the diagonal ones, 
    in Task 3  B is top or bottom """
    
    taskid = np.zeros(len(task))
    taskid[b_pokes == 10 - a_pokes] = 1     
    taskid[np.logical_or(np.logical_or(b_pokes == 2, b_pokes == 3), np.logical_or(b_pokes == 7, b_pokes == 8))] = 2  
    taskid[np.logical_or(b_pokes ==  1, b_pokes == 9)] = 3
         
    return taskid


def exponential(x, a, k, b):
    'Exponential helper function.'
    return a*np.exp(k*x) + b

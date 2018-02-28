import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func

# minibatch gradient descent
def gradient_descent(g, alpha, max_its, w, num_pts, train_portion,**kwargs):    
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = value_and_grad(g_flat)

    # containers for histories
    weight_hist = []
    train_ind_hist = []
    test_ind_hist = []
    
    # store first weights
    weight_hist.append(unflatten(w))
    
    # pick random proportion of training indecies
    train_num = int(np.round(train_portion*num_pts))
    inds = np.random.permutation(num_pts)
    train_inds = inds[:train_num]
    test_inds = inds[train_num:]
    
    # record train / test inds
    train_ind_hist.append(train_inds)
    test_ind_hist.append(test_inds)
    
    # over the line
    for k in range(max_its):   
        # plug in value into func and derivative
        cost_eval,grad_eval = grad(w,train_inds)
        grad_eval.shape = np.shape(w)

        # take descent step with momentum
        w = w - alpha*grad_eval

        # record weight update
        weight_hist.append(unflatten(w))        
        
        #### pick new train / test split ####
        # pick random proportion of training indecies
        train_num = int(np.round(train_portion*num_pts))
        inds = np.random.permutation(num_pts)
        train_inds = inds[:train_num]
        test_inds = inds[train_num:]
        
        # record train / test inds
        train_ind_hist.append(train_inds)
        test_ind_hist.append(test_inds)
        
    return weight_hist,train_ind_hist,test_ind_hist
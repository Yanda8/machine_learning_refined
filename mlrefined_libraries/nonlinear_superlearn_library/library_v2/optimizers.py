import autograd.numpy as np
from autograd import value_and_grad 
from autograd.misc.flatten import flatten_func

# gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
def gradient_descent(g_train,g_test,alpha_choice,max_its,w_init):
    # flatten the input function to more easily deal with costs that have layers of parameters
    g_flat, unflatten, w = flatten_func(g_train, w_init) # note here the output 'w' is also flattened
    g_flat_test, un, w_test = flatten_func(g_test, w_init) # note here the output 'w' is also flattened

    # compute the gradient function of our input function - note this is a function too
    # that - when evaluated - returns both the gradient and function evaluations (remember
    # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
    # an Automatic Differntiator to evaluate the gradient)
    gradient = value_and_grad(g_flat)

    # run the gradient descent loop
    weight_history = []            # container for weight history
    train_cost_history = []        # container for training cost function history
    test_cost_history = []         # container for testing cost function history

    alpha = 0
    for k in range(1,max_its+1):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice

        # evaluate the gradient, store current (unflattened) weights and train cost function value
        cost_eval,grad_eval = gradient(w)
        weight_history.append(unflatten(w))
        train_cost_history.append(cost_eval)
        
        # store testing history
        test_eval = g_flat_test(w)
        test_cost_history.append(test_eval)

        # take gradient descent step
        w = w - alpha*grad_eval

    # collect final weights
    weight_history.append(unflatten(w))
    # compute final cost function value via g itself (since we aren't computing 
    # the gradient at the final step we don't get the final cost function value 
    # via the Automatic Differentiatoor) 
    train_cost_history.append(g_flat(w))  
    test_cost_history.append(g_flat_test(w))

    return weight_history,train_cost_history,test_cost_history
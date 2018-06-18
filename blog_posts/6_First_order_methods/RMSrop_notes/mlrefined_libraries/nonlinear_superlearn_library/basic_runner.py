import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func
import copy
from inspect import signature

'''
A list of cost functions for supervised learning.  Use the choose_cost function
to choose the desired cost with input data  
'''
class Setup:
    def __init__(self,x,y,feature_transforms,**kwargs):
        normalize = 'standard'
        if 'normalize' in kwargs:
            normalize = kwargs['normalize']
        if normalize == 'standard':
            # create normalizer
            self.normalizer,self.inverse_normalizer = self.standard_normalizer(x)

            # normalize input 
            self.x = self.normalizer(x)
        else:
            self.x = x
            self.normalizer = lambda data: data
            self.inverse_normalizer = lambda data: data
            
        # make any other variables not explicitly input into cost functions globally known
        self.y = y
        self.feature_transforms = feature_transforms
        
        # count parameter layers of input to feature transform
        self.sig = signature(self.feature_transforms)

        # make cost function choice
        self.cost_func = self.least_squares
        self.version = 'None'
        if 'version' in kwargs:
            self.version = kwargs['version']

        self.algo = 'gradient_descent'

        # set parameters by hand
        if 'algo' in kwargs:
            self.algo = kwargs['algo']

    # run optimization
    def fit(self,**kwargs):
        # basic parameters for gradient descent run
        max_its = 500; alpha_choice = 10**(-1);
        w = 0.1*np.random.randn(np.shape(self.x)[0] + 1,1)

        if 'max_its' in kwargs:
            max_its = kwargs['max_its']
        if 'alpha_choice' in kwargs:
            alpha_choice = kwargs['alpha_choice']
        if 'w' in kwargs:
            w = kwargs['w']

        # run gradient descent
        if self.algo == 'gradient_descent':
            self.weight_history, self.cost_history,self.grad_history = self.gradient_descent(self.cost_func,w,alpha_choice,max_its,self.version)
        if self.algo == 'RMSprop':
            gamma = 0.9
            if 'gamma' in kwargs:
                gamma = kwargs['gamma']
            self.weight_history, self.cost_history,self.grad_history, self.h_history = self.RMSprop(self.cost_func,w,alpha_choice,max_its,gamma = gamma)

    ###### cost functions #####
    # compute linear combination of input point
    def model(self,x,w):   
        # feature transformation - switch for dealing
        # with feature transforms that either do or do
        # not have internal parameters
        f = 0
        if len(self.sig.parameters) == 2:
            f = self.feature_transforms(x,w[0])
        else: 
            f = self.feature_transforms(x)    

        # compute linear combination and return
        # switch for dealing with feature transforms that either 
        # do or do not have internal parameters
        a = 0
        if len(self.sig.parameters) == 2:
            a = w[1][0] + np.dot(f.T,w[1][1:])
        else:
            a = w[0] + np.dot(f.T,w[1:])
        return a.T
    
    ###### regression costs #######
    # an implementation of the least squares cost function for linear regression
    def least_squares(self,w):
        cost = np.sum((self.model(self.x,w) - self.y)**2)
        return cost/float(np.size(self.y))
    
    ##### optimizers ####
    def gradient_descent(self,g,w,alpha,max_its,version):    
        # flatten the input function, create gradient based on flat function
        g_flat, unflatten, w = flatten_func(g, w)
        grad = value_and_grad(g_flat)
        cost,grad_eval = grad(w)
        grad_eval.shape = np.shape(w)
        
        ### normalized or unnormalized descent step? ###
        if version == 'normalized':
            grad_norm = np.linalg.norm(grad_eval)
            if grad_norm == 0:
                grad_norm += 10**-6*np.sign(2*np.random.rand(1) - 1)
            grad_eval /= grad_norm
                
        if version == 'component_normalized':
            component_norm = np.abs(grad_eval)
            ind = np.argwhere(component_norm < 10**(-6))
            component_norm[ind] = 10**(-6)
            grad_eval /= component_norm
                
        # record history
        cost = g_flat(w)
        w_hist = []
        w_hist.append(unflatten(w))
        cost_hist = []
        cost_hist.append(cost)
        grad_hist = []
        grad_hist.append(grad_eval)

        # start gradient descent loop        
        for k in range(max_its):   
            # plug in value into func and derivative
            cost,grad_eval = grad(w)
            grad_eval.shape = np.shape(w)

            ### normalized or unnormalized descent step? ###
            if version == 'normalized':
                grad_norm = np.linalg.norm(grad_eval)
                if grad_norm == 0:
                    grad_norm += 10**-6*np.sign(2*np.random.rand(1) - 1)
                grad_eval /= grad_norm
                
            if version == 'component_normalized':
                component_norm = np.abs(grad_eval)
                ind = np.argwhere(component_norm < 10**(-6))
                component_norm[ind] = 10**(-6)
                grad_eval /= component_norm
                
            # take descent step 
            w = w - alpha*grad_eval

            # record weight update
            w_hist.append(unflatten(w))
            cost_hist.append(cost)
            grad_hist.append(grad_eval)

        return w_hist,cost_hist,grad_hist

    def RMSprop(self,g,w,alpha,max_its,**kwargs):        
        # rmsprop params
        gamma=0.9
        eps=10**-8
        if 'gamma' in kwargs:
            gamma = kwargs['gamma']
        if 'eps' in kwargs:
            eps = kwargs['eps']
        
        # flatten the input function, create gradient based on flat function
        g_flat, unflatten, w = flatten_func(g, w)
        grad = value_and_grad(g_flat)
        cost,grad_eval = grad(w)
        grad_eval.shape = np.shape(w)

        # initialize average gradient
        h = np.ones(np.size(w))
        
        # record history
        w_hist = [unflatten(w)]
        cost_hist = [cost]
        grad_hist = [grad_eval]
        h_hist = [h]

        # over the line
        for k in range(max_its):                           
            # evaluate and shape grad
            cost,grad_eval = grad(w)
            grad_eval.shape = np.shape(w)
            
            # update exponential average of past gradients
            h = gamma*h + (1 - gamma)*grad_eval**2 
        
            # take descent step 
            w = w - alpha*grad_eval / (h**(0.5) + eps)

            # record weight update, train costs
            w_hist.append(unflatten(w))
            cost_hist.append(cost)
            
            # record grad and h hist
            grad_hist.append(grad_eval)
            h_hist.append(h)
            
        return w_hist,cost_hist,grad_hist,h_hist
 

    ###### normalizers #####
    # standard normalization function 
    def standard_normalizer(self,x):
        # compute the mean and standard deviation of the input
        x_means = np.mean(x,axis = 1)[:,np.newaxis]
        x_stds = np.std(x,axis = 1)[:,np.newaxis]   

        # create standard normalizer function
        normalizer = lambda data: (data - x_means)/x_stds

        # create inverse standard normalizer
        inverse_normalizer = lambda data: data*x_stds + x_means

        # return normalizer 
        return normalizer,inverse_normalizer

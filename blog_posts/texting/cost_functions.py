# other basic libraries
import time
import copy
import autograd.numpy as np

class Setup:
    def __init__(self,x,y,cost):
        # input data
        self.x = x
        self.y = y
        
        # make cost function choice
        self.cost_func = 0
        if cost == 'least_squares':
            self.cost_func = self.least_squares
        if cost == 'least_absolute_deviations':
            self.cost_func = self.least_absolute_deviations
        if cost == 'softmax':
            self.cost_func = self.softmax
        if cost == 'relu':
            self.cost_func = self.relu
        if cost == 'counter':
            self.cost_func = self.counting_cost
        if cost == 'multiclass_softmax':
            self.cost_func = self.multiclass_softmax
        if cost == 'fusion_rule':
            self.cost_func = self.fusion_rule
        if cost == 'minibatch_softmax':
            self.cost_func = self.minibatch_softmax
        if cost == 'minibatch_multiclass_softmax':
            self.cost_func = self.minibatch_multiclass_softmax
        if cost == 'multiclass_counter':
            self.cost_func = self.multiclass_counter       
        

    ###### cost functions #####
    # an implementation of the least squares cost function for linear regression
    def least_squares(self,w):
        cost = np.sum((np.dot(self.x.T,w) - self.y)**2)
        return cost/float(len(self.y))
    
    # a compact least absolute deviations cost function
    def least_absolute_deviations(self,w):
        cost = np.sum(np.abs(np.dot(self.x.T,w) - self.y))
        return cost/float(len(self.y))

    # the convex softmax cost function
    def softmax(self,w):
        cost = np.sum(np.log(1 + np.exp(-self.y*np.dot(self.x.T,w))))
        return cost/float(len(self.y))
    
    # the convex relu cost function
    def relu(self,w):
        cost = np.sum(np.maximum(0,-self.y*np.dot(self.x.T,w)))
        return cost/float(len(self.y))

    # the counting cost function
    def counting_cost(self,w):
        cost = np.sum((np.sign(np.dot(self.x.T,w)) - self.y)**2)
        return 0.25*cost 
    
    # multiclass softmaax regularized by the summed length of all normal vectors
    def multiclass_softmax(self,W):        
        # pre-compute predictions on all points
        all_evals = W[0,:] + np.dot(self.x.T,W[1:,:])

        # compute cost in compact form using numpy broadcasting
        a = np.log(np.sum(np.exp(all_evals),axis = 1)) 
        b = all_evals[np.arange(len(self.y)),self.y-1]
        cost = np.sum(a - b)
        return cost/float(len(self.y))
    
    # fusion rule for counting number of misclassifications on an input multiclass dataset
    def fusion_rule(self,W):
        # pre-compute predictions on all points
        all_evals = W[0,:] + np.dot(self.x.T,W[1:,:])

        # create predicted labels
        y_predict = np.argmax(all_evals,axis = 1) + 1

        return y_predict
    
    def multiclass_counter(self,W):
        # make predictions
        y_predict = self.fusion_rule(W)
        
        # compare to actual labels
        misclassifications = int(sum([abs(np.sign(a - b)) for a,b in zip(self.y,y_predict)]))
        
        return misclassifications
    
    ####### minibatch cost functions #######

    # the softmax cost - with minibatch input
    def minibatch_softmax(self,w,iter):
        # get subset of points
        x_p = self.x[:,iter]
        y_p = self.y[iter]

        # compute cost over just this subset
        cost  = np.sum(np.log(1 + np.exp((-y_p)*(w[0] + np.dot(x_p,w[1:])))))
        return cost/len(y_p)
    
    # multiclass softmax - minibatch
    def minibatch_multiclass_softmax(self,W,iter):
        # get subset of points
        x_p = self.x[:,iter]
        y_p = self.y[iter]
        
        # pre-compute predictions on all points
        all_evals = W[0,:] + np.dot(x_p.T,W[1:,:])

        # compute cost in compact form using numpy broadcasting
        a = np.log(np.sum(np.exp(all_evals),axis = 1)) 
        b = all_evals[np.arange(len(y_p)),y_p-1]
        cost = np.sum(a - b)
        return cost/float(len(y_p))
        
        
    

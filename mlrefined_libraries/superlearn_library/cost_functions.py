# other basic libraries
import time
import copy
import autograd.numpy as np

class Setup:
    def __init__(self,x,y,cost,**kwargs):
        # input data
        self.x = x
        self.y = y
        self.lam = 0
        if 'lam' in kwargs:
            self.lam = kwargs['lam']
        
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
            
        if cost == 'multiclass_perceptron':
            self.cost_func = self.multiclass_perceptron
        if cost == 'multiclass_softmax':
            self.cost_func = self.multiclass_softmax
        if cost == 'multiclass_counter':
            self.cost_func = self.multiclass_counting_cost

    ###### basic model ######
    # compute linear combination of input point
    def model(self,x,w):
        # tack a 1 onto the top of each input point all at once
        o = np.ones((1,np.shape(x)[1]))
        x = np.vstack((o,x))

        # compute linear combination and return
        a = np.dot(x.T,w)
        return a
            
    ###### cost functions #####
    # an implementation of the least squares cost function for linear regression
    def least_squares(self,w):
        cost = np.sum((self.model(self.x,w) - self.y)**2)
        return cost/float(len(self.y))
    
    # a compact least absolute deviations cost function
    def least_absolute_deviations(self,w):
        cost = np.sum(np.abs(self.model(self.x,w) - self.y))
        return cost/float(len(self.y))

    # the convex softmax cost function
    def softmax(self,w):
        cost = np.sum(np.log(1 + np.exp(-self.y*self.model(self.x,w))))
        return cost/float(len(self.y))
    
    # the convex relu cost function
    def relu(self,w):
        cost = np.sum(np.maximum(0,-self.y*self.model(self.x,w)))
        return cost/float(len(self.y))

    # the counting cost function
    def counting_cost(self,w):
        cost = np.sum((np.sign(self.model(self.x,w)) - self.y)**2)
        return 0.25*cost 
    
    # multiclass perceptron
    def multiclass_perceptron(self,w):        
        # pre-compute predictions on all points
        all_evals = self.model(self.x,w)

        # compute maximum across data points
        a =  np.max(all_evals,axis = 1)        

        # compute cost in compact form using numpy broadcasting
        b = all_evals[np.arange(len(self.y)),self.y.astype(int).flatten()-1]
        cost = np.sum(a - b)

        # add regularizer
        cost = cost + self.lam*np.linalg.norm(w[1:,:],'fro')**2

        # return average
        return cost/float(len(self.y))
    
    # multiclass softmax
    def multiclass_softmax(self,w):        
        # pre-compute predictions on all points
        all_evals = self.model(self.x,w)

        # compute softmax across data points
        a = np.log(np.sum(np.exp(all_evals),axis = 1)) 

        # compute cost in compact form using numpy broadcasting
        b = all_evals[np.arange(len(self.y)),self.y.astype(int).flatten()-1]
        cost = np.sum(a - b)

        # add regularizer
        cost = cost + self.lam*np.linalg.norm(w[1:,:],'fro')**2

        # return average
        return cost/float(len(self.y))
    
    # multiclass misclassification cost function - aka the fusion rule
    def multiclass_counting_cost(self,w):                
        # pre-compute predictions on all points
        all_evals = self.model(self.x,w)

        # compute predictions of each input point
        y_predict = (np.argmax(all_evals,axis = 1) + 1)[:,np.newaxis]
        
        # compare predicted label to actual label
        count = np.sum(np.abs(np.sign(self.y - y_predict)))

        # return number of misclassifications
        return count
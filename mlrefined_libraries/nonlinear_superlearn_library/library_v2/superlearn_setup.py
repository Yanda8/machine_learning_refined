import autograd.numpy as np
from . import optimizers 
from . import cost_functions
from . import normalizers
from . import multilayer_perceptron
from . import history_plotters

class Setup:
    def __init__(self,x,y,**kwargs):
        # link in data
        self.x = x
        self.y = y
        
        # make containers for all histories
        self.weight_histories = []
        self.train_cost_histories = []
        self.train_count_histories = []
        self.test_cost_histories = []
        self.test_count_histories = []
        
    #### define feature transformation ####
    def choose_features(self,name,**kwargs): 
        # select from pre-made feature transforms
        if name == 'multilayer_perceptron':
            transformer = multilayer_perceptron.Setup(**kwargs)
            self.feature_transforms = transformer.feature_transforms
            self.initializer = transformer.initializer
            self.layer_sizes = transformer.layer_sizes
        self.feature_name = name

    #### define normalizer ####
    def choose_normalizer(self,name):
        # produce normalizer / inverse normalizer
        s = normalizers.Setup(self.x_train,name)
        self.normalizer = s.normalizer
        self.inverse_normalizer = s.inverse_normalizer
        
        # normalize input 
        self.x_train = self.normalizer(self.x_train)
        self.x_test = self.normalizer(self.x_test)
        self.normalizer_name = name
        
    #### split data into training and testing sets ####
    def make_train_test_split(self,train_portion):
        # translate desired training portion into exact indecies
        r = np.random.permutation(self.x.shape[1])
        train_num = int(np.round(train_portion*len(r)))
        self.train_inds = r[:train_num]
        self.test_inds = r[train_num:]
        
        # define training and testing sets
        self.x_train = self.x[:,self.train_inds]
        self.x_test = self.x[:,self.test_inds]
        
        self.y_train = self.y[:,self.train_inds]
        self.y_test = self.y[:,self.test_inds]
     
    #### define cost function ####
    def choose_cost(self,name,**kwargs):
        # create training and testing cost functions
        funcs = cost_functions.Setup(name,self.x_train,self.y_train,self.feature_transforms,**kwargs)
        self.train_cost = funcs.cost
        self.model = funcs.model
        
        funcs = cost_functions.Setup(name,self.x_test,self.y_test,self.feature_transforms,**kwargs)
        self.test_cost = funcs.cost
        
        # if the cost function is a two-class classifier, build a counter too
        if name == 'softmax' or name == 'perceptron':
            funcs = cost_functions.Setup('twoclass_counter',self.x_train,self.y_train,self.feature_transforms,**kwargs)
            self.train_counter = funcs.cost
            
            funcs = cost_functions.Setup('twoclass_counter',self.x_test,self.y_test,self.feature_transforms,**kwargs)
            self.test_counter = funcs.cost
            
        if name == 'multiclass_softmax' or name == 'multiclass_perceptron':
            funcs = cost_functions.Setup('multiclass_counter',self.x_train,self.y_train,self.feature_transforms,**kwargs)
            self.train_counter = funcs.cost
            
            funcs = cost_functions.Setup('multiclass_counter',self.x_test,self.y_test,self.feature_transforms,**kwargs)
            self.test_counter = funcs.cost
            
        self.cost_name = name
            
    #### run optimization ####
    def fit(self,**kwargs):        
        # basic parameters for gradient descent run (default algorithm)
        max_its = 500; alpha_choice = 10**(-1);
        self.w_init = self.initializer()
        
        # set parameters by hand
        if 'max_its' in kwargs:
            self.max_its = kwargs['max_its']
        if 'alpha_choice' in kwargs:
            self.alpha_choice = kwargs['alpha_choice']

        # run gradient descent
        weight_history, train_cost_history, test_cost_history = optimizers.gradient_descent(self.train_cost,self.test_cost,self.alpha_choice,self.max_its,self.w_init)
        
         # store all new histories
        self.weight_histories.append(weight_history)
        self.train_cost_histories.append(train_cost_history)
        self.test_cost_histories.append(test_cost_history)
        
        # if classification produce count history
        if self.cost_name == 'softmax' or self.cost_name == 'perceptron' or self.cost_name == 'multiclass_softmax' or self.cost_name == 'multiclass_perceptron':
            train_count_history = [self.train_counter(v) for v in weight_history]
            test_count_history = [self.test_counter(v) for v in weight_history]

            # store count history
            self.train_count_histories.append(train_count_history)
            self.test_count_histories.append(test_count_history)

 
    #### plot histories ###
    def show_histories(self,**kwargs):
        start = 0
        if 'start' in kwargs:
            start = kwargs['start']
        history_plotters.Setup(self.train_cost_histories,self.train_count_histories,self.test_cost_histories,self.test_count_histories,start)
        
    
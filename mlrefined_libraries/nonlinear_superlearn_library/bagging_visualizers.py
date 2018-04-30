# import standard plotting and animation
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.animation as animation
from mlrefined_libraries.JSAnimation_slider_only import IPython_display_slider_only
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

# import autograd functionality
import autograd.numpy as np

# import standard libraries
import math
import time
import copy
from inspect import signature

class Visualizer:
    '''
    Visualize linear regression in 2 and 3 dimensions.  For single input cases (2 dimensions) the path of gradient descent on the cost function can be animated.
    '''
    #### initialize ####
    def __init__(self,csvname):
        # grab input
        data = np.loadtxt(csvname,delimiter = ',')
        self.x = data[:-1,:]
        self.y = data[-1:,:] 
        
        self.colors = [[1,0.8,0.5],[0,0.7,1]]
        self.plot_colors = ['lime','blueviolet','magenta']
        
        # if 1-d regression data make sure points are sorted
        if np.shape(self.x)[1] == 1:
            ind = np.argsort(self.x.flatten())
            self.x = self.x[ind,:]
            self.y = self.y[ind,:]
            
    ########## show boosting crossval on 1d regression, with fit to residual ##########
    def show_runs(self,runs,**kwargs):
        # construct figure
        fig, axs = plt.subplots(figsize=(9,6),nrows=2,ncols = len(runs))
        all_models = []

        for k in range(len(runs)):
            # get current run for cost function history plot
            run = runs[k]

            # pluck out current weights 
            s,t = self.draw_fit_trainval(axs[0,k],run,self.plot_colors[k])
            
            # store model
            all_models.append([s,t])
            
        # plot all models and ave
        self.draw_models(axs[1,1],all_models)
        
        # run off all other axes labels
        axs[1,0].axis('off')
        
        for i in range(2,len(runs)):
            axs[1,i].axis('off')
 
    def show_baggs(self,runs):
        fig, axs = plt.subplots(figsize=(8,4),ncols = 2)
        
        # set plotting limits
        xmax = np.max(copy.deepcopy(self.x))
        xmin = np.min(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0
        xmin -= xgap
        xmax += xgap

        ymax = np.max(copy.deepcopy(self.y))
        ymin = np.min(copy.deepcopy(self.y))
        ygap = (ymax - ymin)*0.5
        ymin -= ygap
        ymax += ygap    

        ####### plot total model on original dataset #######
        # scatter original data - training and validation sets
        ax = axs[0]
        ax.scatter(self.x,self.y,color = 'k',s = 40,edgecolor = 'w',linewidth = 0.9)
        
        ### clean up panels ###             
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        
        # label axes
        ax.set_xlabel(r'$x$', fontsize = 14)
        ax.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 15)
        
        ax = axs[1]
        ax.scatter(self.x,self.y,color = 'k',s = 40,edgecolor = 'w',linewidth = 0.9)
        
        ### clean up panels ###             
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        
        # label axes
        ax.set_xlabel(r'$x$', fontsize = 14)
        ax.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 15)
        
        # plot fit on residual
        s = np.linspace(xmin,xmax,2000)[np.newaxis,:]
        
        # plot fit on residual
        t_ave = np.zeros((s.shape))
        for k in range(len(runs)):
            # get current run
            run = runs[k]
            cost = run.cost
            model = run.model
            feat = run.feature_transforms
            normalizer = run.normalizer
            w = run.weight_histories

            # get best weights                
            t = model(normalizer(s),w)
            ax.plot(s.T,t.T,linewidth = 2,alpha = 0.4)
            t_ave += t
        t_ave /= float(len(runs))
        
        # plot ave model
        ax.plot(s.T,t_ave.T,linewidth = 4,c = 'k',alpha = 1)
        ax.plot(s.T,t_ave.T,linewidth = 3.5,c = 'r',alpha = 1)

        
    def draw_models(self,ax,all_models):
        # set plotting limits
        xmax = np.max(copy.deepcopy(self.x))
        xmin = np.min(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.1
        xmin -= xgap
        xmax += xgap

        ymax = np.max(copy.deepcopy(self.y))
        ymin = np.min(copy.deepcopy(self.y))
        ygap = (ymax - ymin)*0.3
        ymin -= ygap
        ymax += ygap    

        ####### plot total model on original dataset #######
        # scatter original data - training and validation sets
        ax.scatter(self.x,self.y,color = 'k',s = 40,edgecolor = 'w',linewidth = 0.9)
        
        # plot fit on residual
        t_ave = np.zeros((all_models[0][1].shape))
        for k in range(len(all_models)):
            model = all_models[k]
            # plot current model
            s = model[0]
            t = model[1]
            ax.plot(s.T,t.T,linewidth = 2,alpha = 0.4,c = self.plot_colors[k])
            t_ave += t
        t_ave /= float(len(all_models))
        
        # plot ave model
        s = all_models[0][0]
        ax.plot(s.T,t_ave.T,linewidth = 4,c = 'k',alpha = 1)
        ax.plot(s.T,t_ave.T,linewidth = 3.5,c = 'r',alpha = 1)

        ### clean up panels ###             
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        
        # label axes
        ax.set_xlabel(r'$x$', fontsize = 14)
        ax.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 15)
            
    def draw_fit_trainval(self,ax,run,color):
        # set plotting limits
        xmax = np.max(copy.deepcopy(self.x))
        xmin = np.min(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.1
        xmin -= xgap
        xmax += xgap

        ymax = np.max(copy.deepcopy(self.y))
        ymin = np.min(copy.deepcopy(self.y))
        ygap = (ymax - ymin)*0.3
        ymin -= ygap
        ymax += ygap    

        ####### plot total model on original dataset #######
        # scatter original data - training and validation sets
        train_inds = run.train_inds
        valid_inds = run.val_inds
        ax.scatter(self.x[:,train_inds],self.y[:,train_inds],color = self.colors[1],s = 40,edgecolor = 'k',linewidth = 0.9)
        ax.scatter(self.x[:,valid_inds],self.y[:,valid_inds],color = self.colors[0],s = 40,edgecolor = 'k',linewidth = 0.9)
        
        # plot fit on residual
        s = np.linspace(xmin,xmax,2000)[np.newaxis,:]

        # get current run
        cost = run.cost
        model = run.model
        feat = run.feature_transforms
        normalizer = run.normalizer
        w = run.weight_histories

        # get best weights                
        t = model(normalizer(s),w)

        #ax.plot(s.T,t.T,linewidth = 4,c = 'k',alpha = 0.4)
        ax.plot(s.T,t.T,linewidth = 3,c = color,alpha = 0.5)

        ### clean up panels ###             
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        
        # label axes
        ax.set_xlabel(r'$x$', fontsize = 14)
        ax.set_ylabel(r'$y$', rotation = 0,fontsize = 14,labelpad = 15)
        
        return s,t
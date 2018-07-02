# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

# import autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np
import math
import time

class Visualizer:
    '''
    Illustrate a run of your preferred optimization algorithm on a one or two-input function.  Run
    the algorithm first, and input the resulting weight history into this wrapper.
    '''
        
    # plot some partial derivative magnitudes
    def plot_partial_mags(self,run,**kwargs):
        parts = [3]
        if 'parts' in kwargs:
            parts = kwargs['parts']
        
        #colors = ['k','magenta','springgreen','blueviolet','chocolate']
        
        start = 0
        end = len(run.grad_history)
        if 'start' in kwargs:
            start = kwargs['start']
        if 'end' in kwargs:
            end = kwargs['end']
        
        # initialize figure
        fig = plt.figure(figsize = (9,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(len(parts), 1) 
        for n in range(len(parts)):
            ax = plt.subplot(gs[n]); 

            # plot partial deriv mag portion
            grad_hist = run.grad_history
            part = parts[n]
            part = [grad_hist[v][n]**2 for v in range(len(grad_hist))]
            ax.plot(np.arange(start,end,1),part[start:end],linewidth = 2,c='k',label = 'derivative magnitude')

            if hasattr(run, 'h_history'):
                h_hist = run.h_history
                part = parts[n]
                part = [h_hist[v][n] for v in range(len(h_hist))]
                ax.plot(np.arange(start,end,1),part[start:end],linewidth = 3,color = 'r',label = 'exponential average') 
                
            if n == 0:
                ax.set_xlabel('iter')
                ax.set_ylabel('magnitude')
                
                
        anchor = (1,1)
        plt.legend(loc='upper right', bbox_to_anchor=anchor)
                
    # compare cost histories from multiple runs
    def plot_cost_histories(self,runs,start,**kwargs):
        # plotting colors
        colors = ['k','magenta','springgreen','blueviolet','chocolate']
        
        # initialize figure
        fig = plt.figure(figsize = (10,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); 
        
        # any labels to add?        
        labels = [' ',' ']
        if 'labels' in kwargs:
            labels = kwargs['labels']
            
        # plot points on cost function plot too?
        points = False
        if 'points' in kwargs:
            points = kwargs['points']

        # run through input histories, plotting each beginning at 'start' iteration
        for c in range(len(runs)):
            history = runs[c].cost_history
            label = labels[c]
                
            # check if a label exists, if so add it to the plot
            if np.size(label) == 0:
                ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3,color = colors[c]) 
            else:               
                ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3,color = colors[c],label = label) 
                
            # check if points should be plotted for visualization purposes
            if points == True:
                ax.scatter(np.arange(start,len(history),1),history[start:],s = 90,color = colors[c],edgecolor = 'w',linewidth = 2,zorder = 3) 

        # clean up panel
        xlabel = 'step $k$'
        if 'xlabel' in kwargs:
            xlabel = kwargs['xlabel']
        ylabel = r'$g\left(\mathbf{w}^k\right)$'
        if 'ylabel' in kwargs:
            ylabel = kwargs['ylabel']
        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
        if np.size(label) > 0:
            anchor = (1,1)
            if 'anchor' in kwargs:
                anchor = kwargs['anchor']
            plt.legend(loc='upper right', bbox_to_anchor=anchor)
            #leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

        ax.set_xlim([start - 0.5,len(history) - 0.5])
        plt.show()
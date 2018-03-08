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
    animators for time series
    '''
    #### single dimension regression animation ####
    def animate_1d_series(self,x,func,params,**kwargs):
        # produce figure
        fig = plt.figure(figsize = (9,4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,7,1]) 
        ax = plt.subplot(gs[0]); ax.axis('off')
        ax1 = plt.subplot(gs[1]); 
        ax2 = plt.subplot(gs[2]); ax2.axis('off')
        artist = fig
        
        # view limits
        xmin = -3
        xmax = len(x) + 3
        ymin = np.min(x)
        ymax = np.max(x) 
        ygap = (ymax - ymin)*0.15
        ymin -= ygap
        ymax += ygap
            
        # start animation
        num_frames = len(params)+1
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax1.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
                
            # plot x
            ax1.scatter(np.arange(len(x)),x,c = 'k',edgecolor = 'w',s = 60,linewidth = 1,zorder = 2);
            ax1.plot(x,alpha = 0.5,c = 'k',zorder = 2);

            # create y
            if k == 0:
                T = params[0]
                y = func(x,T)
                ax1.scatter(np.arange(len(y)),y,c = 'fuchsia',edgecolor = 'w',s = 60,linewidth = 1,zorder = 3,alpha = 0);
                ax1.set_title(r'Original data')

            if k > 0:
                T = params[k-1]
                y = func(x,T)
                ax1.scatter(np.arange(len(y)),y,c = 'fuchsia',edgecolor = 'w',s = 60,linewidth = 1,zorder = 3);
                ax1.plot(y,alpha = 0.5,c = 'fuchsia',zorder = 3);
                ax1.set_title(r'$T = $ ' + str(T))

            # label axes
            ax1.set_xlabel(r'$p$',fontsize = 13)
            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin,ymax])
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        return(anim)
  

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
        data = data.T
        self.x = data[:,:-1]
        self.y = data[:,-1:]
        self.colors = ['salmon','cornflowerblue','lime','bisque','mediumaquamarine','b','m','g']
        
        # if 1-d regression data make sure points are sorted
        if np.shape(self.x)[1] == 1:
            ind = np.argsort(self.x.flatten())
            self.x = self.x[ind,:]
            self.y = self.y[ind,:]
        
    #### single dimension regression animation ####
    def animate_1d_regression(self,run,frames,**kwargs):
        # select inds of history to plot
        weight_history = run.weight_histories[0]
        cost_history = run.cost_histories[0]
        inds = np.arange(0,len(weight_history),int(len(weight_history)/float(frames)))
        weight_history_sample = [weight_history[v] for v in inds]
        cost_history_sample = [cost_history[v] for v in inds]
        start = inds[0]

        # construct figure
        fig = plt.figure(figsize=(9,4))
        artist = fig
        
        # parse any input args
        scatter = 'none'
        if 'scatter' in kwargs:
            scatter = kwargs['scatter']
        show_history = False
        if 'show_history' in kwargs:
            show_history = kwargs['show_history']

        # create subplot with 1 active panel
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1]) 
        ax = plt.subplot(gs[1]); 
        ax1 = plt.subplot(gs[0]); ax1.axis('off')
        ax3 = plt.subplot(gs[2]); ax3.axis('off')
        
        if show_history == True:
            # create subplot with 2 active panels
            gs = gridspec.GridSpec(1, 2, width_ratios=[2,1]) 
            ax = plt.subplot(gs[0]); 
            ax1 = plt.subplot(gs[1]); 
        
        # start animation
        num_frames = len(inds)
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax.cla()
            
            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            # get current index to plot
            current_ind = inds[k]
            
            # pluck out current weights 
            w_best = weight_history[current_ind]
            
            # produce static img
            show_fit = False
            if k > 0:
                show_fit = True
            self.show_1d_regression(ax,w_best,run,scatter,show_fit)
            
            # show cost function history
            if show_history == True:
                ax1.cla()
                ax1.scatter(current_ind,cost_history[current_ind],s = 60,color = 'r',edgecolor = 'k',zorder = 3)
                self.plot_cost_history(ax1,cost_history,start)
                
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
        
        return(anim)
        
    # 1d regression demo
    def show_1d_regression(self,ax,w_best,runner,scatter,show_fit):
        cost = runner.cost
        predict = runner.model
        feat = runner.feature_transforms
        normalizer = runner.normalizer
        
        # set plotting limits
        xmax = np.max(copy.deepcopy(self.x))
        xmin = np.min(copy.deepcopy(self.x))
        xgap = (xmax - xmin)*0.1
        xmin -= xgap
        xmax += xgap

        ymax = np.max(copy.deepcopy(self.y))
        ymin = np.min(copy.deepcopy(self.y))
        ygap = (ymax - ymin)*0.1
        ymin -= ygap
        ymax += ygap    

        # scatter points or plot continuous version
        if scatter == 'points':
            ax.scatter(self.x.flatten(),self.y.flatten(),color = 'k',s = 40,edgecolor = 'w',linewidth = 0.9)
        elif scatter == 'function':
            ax.scatter(self.x.flatten(),self.y.flatten(),color = 'k',s = 10) 
        else:
            ax.plot(self.x.flatten(),self.y.flatten(),color = 'k',linewidth = 3)

        # clean up panel
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        
        # label axes
        ax.set_xlabel(r'$x$', fontsize = 16)
        ax.set_ylabel(r'$y$', rotation = 0,fontsize = 16,labelpad = 15)
        
        # plot current fit
        if show_fit == True:
            # plot fit
            s = np.linspace(xmin,xmax,2000)[np.newaxis,:]
            if len(np.unique(self.y)) > 2:
                t = predict(normalizer(s),w_best)
            else:
                t = np.tanh(predict(normalizer(s),w_best))

            ax.plot(s.T,t.T,linewidth = 4,c = 'k')
            ax.plot(s.T,t.T,linewidth = 2,c = 'r')
        
        
    #### compare cost function histories ####
    def plot_cost_history(self,ax,history,start):
        # plotting colors
        colors = ['k']
                
        # plot cost function history
        ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 3,color = 'k') 

        # clean up panel / axes labels
        xlabel = 'step $k$'
        ylabel = r'$g\left(\mathbf{w}^k\right)$'
        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
        title = 'cost history'
        ax.set_title(title,fontsize = 18)
        
        # plotting limits
        xmin = 0; xmax = len(history); xgap = xmax*0.05; 

        xmin -= xgap; xmax += xgap;
        ymin = np.min(history); ymax = np.max(history); ygap = ymax*0.05;
        ymin -= ygap; ymax += ygap;
        
        ax.set_xlim([xmin,xmax]) 
        ax.set_ylim([ymin,ymax]) 
    
    ####### animate static_N2_simple run #######
    def animate_static_N2_simple(self,run,frames,**kwargs):      
        # select inds of history to plot
        weight_history = run.weight_histories[0]
        cost_history = run.cost_histories[0]
        inds = np.arange(0,len(weight_history),int(len(weight_history)/float(frames)))
        weight_history_sample = [weight_history[v] for v in inds]
        cost_history_sample = [cost_history[v] for v in inds]
        start = inds[0]
        
        show_history = False
        if 'show_history' in kwargs:
            show_history = kwargs['show_history']
            
        # construct figure
        fig = plt.figure(figsize=(10,5))
        artist = fig
        
        # create subplot with 1 active panel
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,5,1]) 
        ax = plt.subplot(gs[1],aspect = 'equal'); 
        ax1 = plt.subplot(gs[0]); ax1.axis('off')
        ax3 = plt.subplot(gs[2]); ax3.axis('off')

        if show_history == True:
            # create subplot with 2 active panels
            gs = gridspec.GridSpec(1, 3, width_ratios=[3,2,0.01]) 
            ax = plt.subplot(gs[0],aspect = 'equal'); 
            ax1 = plt.subplot(gs[1]); 
            ax2 = plt.subplot(gs[2]); ax2.axis('off');
            
        # start animation
        num_frames = len(inds)
        print ('starting animation rendering...')
        def animate(k):        
            # get current index to plot
            current_ind = inds[k]

            # clear panels
            ax.cla()

            if show_history == True:
                ax1.cla()
                ax1.scatter(current_ind,cost_history[current_ind],s = 60,color = 'r',edgecolor = 'k',zorder = 3)
                self.plot_cost_history(ax1,cost_history,start)

            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()

            # pluck out current weights 
            w_best = weight_history[current_ind]

            # produce static img
            self.static_N2_simple(ax,w_best,run,view = [30,155])
                
            return artist,

        anim = animation.FuncAnimation(fig, animate,frames=num_frames,interval = 25,blit=False)
        
        return(anim)
        
    # show coloring of entire space
    def static_N2_simple(self,ax,w_best,runner,**kwargs):
        cost = runner.cost
        predict = runner.model
        feat = runner.feature_transforms
        normalizer = runner.normalizer
                
        # count parameter layers of input to feature transform
        sig = signature(feat)
        sig = len(sig.parameters)

        # or just take last weights
        self.w = w_best

        ### from above
        ax.set_xlabel(r'$x_1$',fontsize = 15)
        ax.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # plot points in 2d and 3d
        C = len(np.unique(self.y))
        if C == 2:
            ind0 = np.argwhere(self.y == +1)
            ax.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[0], edgecolor = 'k')

            ind1 = np.argwhere(self.y == -1)
            ax.scatter(self.x[ind1,0],self.x[ind1,1],s = 55, color = self.colors[1], edgecolor = 'k')
        else:
            for c in range(C):
                ind0 = np.argwhere(self.y == c)
                ax.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[c], edgecolor = 'k')


        ### create surface and boundary plot ###
        xmin1 = np.min(self.x[:,0])
        xmax1 = np.max(self.x[:,0])
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1

        xmin2 = np.min(self.x[:,1])
        xmax2 = np.max(self.x[:,1])
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2    

        # plot boundary for 2d plot
        r1 = np.linspace(xmin1,xmax1,500)
        r2 = np.linspace(xmin2,xmax2,500)
        s,t = np.meshgrid(r1,r2)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        h = np.concatenate((s,t),axis = 1)
        z = predict(normalizer(h.T),self.w)
        z = np.sign(z)
        
        # reshape it
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))     
        z.shape = (np.size(r1),np.size(r2))
        
        #### plot contour, color regions ####
        ax.contour(s,t,z,colors='k', linewidths=2.5,levels = [0],zorder = 2)
        ax.contourf(s,t,z,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))

    ###### plot plotting functions ######
    def plot_data(self):
        fig = 0
        # plot data in two and one-d
        if np.shape(self.x)[1] < 2:
            # construct figure
            fig, axs = plt.subplots(2,1, figsize=(4,4))
            gs = gridspec.GridSpec(2,1,height_ratios = [6,1]) 
            ax1 = plt.subplot(gs[0],aspect = 'equal');
            ax2 = plt.subplot(gs[1],sharex = ax1); 
            
            # set plotting limits
            xmax = copy.deepcopy(max(self.x))
            xmin = copy.deepcopy(min(self.x))
            xgap = (xmax - xmin)*0.2
            xmin -= xgap
            xmax += xgap
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.5
            ymin -= ygap
            ymax += ygap    

            ### plot in 2-d
            ax1.scatter(self.x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40)

            # clean up panel
            ax1.set_xlim([xmin,xmax])
            ax1.set_ylim([ymin,ymax])
            ax1.axhline(linewidth=0.5, color='k',zorder = 1)
            
            ### plot in 1-d
            ind0 = np.argwhere(self.y == +1)
            ax2.scatter(self.x[ind0],np.zeros((len(self.x[ind0]))),s = 55, color = self.colors[0], edgecolor = 'k',zorder = 3)

            ind1 = np.argwhere(self.y == -1)
            ax2.scatter(self.x[ind1],np.zeros((len(self.x[ind1]))),s = 55, color = self.colors[1], edgecolor = 'k',zorder = 3)
            ax2.set_yticks([0])
            ax2.axhline(linewidth=0.5, color='k',zorder = 1)
        
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
        if np.shape(self.x)[1] == 2:
            # construct figure
            fig, axs = plt.subplots(1, 2, figsize=(9,4))

            # create subplot with 2 panels
            gs = gridspec.GridSpec(1, 2) 
            ax2 = plt.subplot(gs[1],aspect = 'equal'); 
            ax1 = plt.subplot(gs[0],projection='3d'); 

            # scatter points
            self.scatter_pts(ax1,self.x)
            
            ### from above
            ax2.set_xlabel(r'$x_1$',fontsize = 15)
            ax2.set_ylabel(r'$x_2$',fontsize = 15,rotation = 0,labelpad = 20)
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
            # plot points in 2d and 3d
            C = len(np.unique(self.y))
            if C == 2:
                ind0 = np.argwhere(self.y == +1)
                ax2.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[0], edgecolor = 'k')

                ind1 = np.argwhere(self.y == -1)
                ax2.scatter(self.x[ind1,0],self.x[ind1,1],s = 55, color = self.colors[1], edgecolor = 'k')
            else:
                for c in range(C):
                    ind0 = np.argwhere(self.y == c)
                    ax2.scatter(self.x[ind0,0],self.x[ind0,1],s = 55, color = self.colors[c], edgecolor = 'k')
                    
        
            ax1.set_xlabel(r'$x_1$', fontsize = 12,labelpad = 5)
            ax1.set_ylabel(r'$x_2$', rotation = 0,fontsize = 12,labelpad = 5)
            ax1.set_zlabel(r'$y$', rotation = 0,fontsize = 12,labelpad = -3)
        
    # scatter points
    def scatter_pts(self,ax,x):
        if np.shape(x)[1] == 1:
            # set plotting limits
            xmax = copy.deepcopy(max(x))
            xmin = copy.deepcopy(min(x))
            xgap = (xmax - xmin)*0.2
            xmin -= xgap
            xmax += xgap
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.2
            ymin -= ygap
            ymax += ygap    

            # initialize points
            ax.scatter(x,self.y,color = 'k', edgecolor = 'w',linewidth = 0.9,s = 40)

            # clean up panel
            ax.set_xlim([xmin,xmax])
            ax.set_ylim([ymin,ymax])
            
        if np.shape(x)[1] == 2:
            # set plotting limits
            xmax1 = copy.deepcopy(max(x[:,0]))
            xmin1 = copy.deepcopy(min(x[:,0]))
            xgap1 = (xmax1 - xmin1)*0.1
            xmin1 -= xgap1
            xmax1 += xgap1
            
            xmax2 = copy.deepcopy(max(x[:,1]))
            xmin2 = copy.deepcopy(min(x[:,1]))
            xgap2 = (xmax2 - xmin2)*0.1
            xmin2 -= xgap2
            xmax2 += xgap2
            
            ymax = max(self.y)
            ymin = min(self.y)
            ygap = (ymax - ymin)*0.2
            ymin -= ygap
            ymax += ygap    

            # initialize points
            ax.scatter(x[:,0],x[:,1],self.y.flatten(),s = 40,color = 'k', edgecolor = 'w',linewidth = 0.9)

            # clean up panel
            ax.set_xlim([xmin1,xmax1])
            ax.set_ylim([xmin2,xmax2])
            ax.set_zlim([ymin,ymax])
            
            ax.set_xticks(np.arange(round(xmin1), round(xmax1)+1, 1.0))
            ax.set_yticks(np.arange(round(xmin2), round(xmax2)+1, 1.0))
            ax.set_zticks(np.arange(round(ymin), round(ymax)+1, 1.0))
           
            # clean up panel
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            ax.xaxis.pane.set_edgecolor('white')
            ax.yaxis.pane.set_edgecolor('white')
            ax.zaxis.pane.set_edgecolor('white')

            ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
           
    # set axis in left panel
    def move_axis_left(self,ax):
        tmp_planes = ax.zaxis._PLANES 
        ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                             tmp_planes[0], tmp_planes[1], 
                             tmp_planes[4], tmp_planes[5])
        view_1 = (25, -135)
        view_2 = (25, -45)
        init_view = view_2
        ax.view_init(*init_view) 
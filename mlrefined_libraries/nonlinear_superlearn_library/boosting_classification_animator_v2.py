# import standard plotting and animation
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.animation as animation
from mlrefined_libraries.JSAnimation_slider_only import IPython_display_slider_only
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
from matplotlib.ticker import MaxNLocator, FuncFormatter

# import autograd functionality
import autograd.numpy as np
import math
import time
from matplotlib import gridspec
import copy
from matplotlib.ticker import FormatStrFormatter
from inspect import signature

class Visualizer:
    '''
    Visualize cross validation performed on N = 2 dimensional input classification datasets
    '''
    #### initialize ####
    def __init__(self,csvname):
        # grab input
        data = np.loadtxt(csvname,delimiter = ',')
        self.x = data[:-1,:]
        self.y = data[-1:,:] 

        self.colors = ['salmon','cornflowerblue','lime','bisque','mediumaquamarine','b','m','g']
    
    ########## show classification results ##########
    def animate_comparisons(self,frames,runs3,**kwargs):
        pt_size = 55
        if 'pt_size' in kwargs:
            pt_size = kwargs['pt_size']
            
        ### get inds for each run ###
        inds3 = np.arange(0,len(runs3.models),int(len(runs3.models)/float(frames)))
            
        # select inds of history to plot
        num_runs = frames

        # construct figure
        fig = plt.figure(figsize=(9,4))
        artist = fig

        # create subplot with 1 active panel
        gs = gridspec.GridSpec(1, 1) 
        
        ax3 = plt.subplot(gs[2]); ax3.set_aspect('equal'); 
        ax3.axis('off');
        ax3.xaxis.set_visible(False) # Hide only x axis
        ax3.yaxis.set_visible(False) # Hide only x axis
        
        # viewing ranges
        xmin1 = min(copy.deepcopy(self.x[0,:]))
        xmax1 = max(copy.deepcopy(self.x[0,:]))
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1

        xmin2 = min(copy.deepcopy(self.x[1,:]))
        xmax2 = max(copy.deepcopy(self.x[1,:]))
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2

        # start animation
        num_frames = num_runs
        print ('starting animation rendering...')
        def animate(k):
            # clear panels
            ax3.cla()

            # print rendering update
            if np.mod(k+1,25) == 0:
                print ('rendering animation frame ' + str(k+1) + ' of ' + str(num_frames))
            if k == num_frames - 1:
                print ('animation rendering complete!')
                time.sleep(1.5)
                clear_output()
            
            # scatter data
            ind0 = np.argwhere(self.y == +1)
            ind0 = [e[1] for e in ind0]
            ind1 = np.argwhere(self.y == -1)
            ind1 = [e[1] for e in ind1]
            
            ax3.scatter(self.x[0,ind0],self.x[1,ind0],s = pt_size, color = self.colors[0], edgecolor = 'k',antialiased=True)
            ax3.scatter(self.x[0,ind1],self.x[1,ind1],s = pt_size, color = self.colors[1], edgecolor = 'k',antialiased=True)
                
            if k == 0:             
                ax3.set_xlim([xmin1,xmax1])
                ax3.set_ylim([xmin2,xmax2])
                
            # plot fit
            if k > 0:
                # get current run
                a3 = inds3[k-1] 
                steps = runs3.best_steps[:a3+1]
                self.draw_boosting_fit(ax3,steps,a3)
                
            return artist,

        anim = animation.FuncAnimation(fig, animate ,frames=num_frames+1, interval=num_frames+1, blit=True)
        
        return(anim)
    
    

    ### draw boosting fit ###
    def draw_boosting_fit(self,ax,steps,ind):
        # viewing ranges
        xmin1 = min(copy.deepcopy(self.x[0,:]))
        xmax1 = max(copy.deepcopy(self.x[0,:]))
        xgap1 = (xmax1 - xmin1)*0.05
        xmin1 -= xgap1
        xmax1 += xgap1

        xmin2 = min(copy.deepcopy(self.x[1,:]))
        xmax2 = max(copy.deepcopy(self.x[1,:]))
        xgap2 = (xmax2 - xmin2)*0.05
        xmin2 -= xgap2
        xmax2 += xgap2

        ymin = min(copy.deepcopy(self.y))
        ymax = max(copy.deepcopy(self.y))
        ygap = (ymax - ymin)*0.05
        ymin -= ygap
        ymax += ygap

        # plot boundary for 2d plot
        r1 = np.linspace(xmin1,xmax1,30)
        r2 = np.linspace(xmin2,xmax2,30)
        s,t = np.meshgrid(r1,r2)
        s = np.reshape(s,(np.size(s),1))
        t = np.reshape(t,(np.size(t),1))
        h = np.concatenate((s,t),axis = 1).T

        model = lambda x: np.sum([v(x) for v in steps],axis=0)
        z = model(h)
        z = np.sign(z)

        # reshape it
        s.shape = (np.size(r1),np.size(r2))
        t.shape = (np.size(r1),np.size(r2))     
        z.shape = (np.size(r1),np.size(r2))

        #### plot contour, color regions ####
        ax.contour(s,t,z,colors='k', linewidths=2.5,levels = [0],zorder = 2)
        ax.contourf(s,t,z,colors = [self.colors[1],self.colors[0]],alpha = 0.15,levels = range(-1,2))

        ### cleanup left plots, create max view ranges ###
        ax.set_xlim([xmin1,xmax1])
        ax.set_ylim([xmin2,xmax2])
        ax.set_title(str(ind+1) + ' units fit to data',fontsize = 14)
        
        
    def plot_train_valid_errors(self,ax,k,train_errors,valid_errors,num_units):
        num_elements = np.arange(len(train_errors))

        ax.plot([v+1 for v in num_elements[:k+1]] ,train_errors[:k+1],color = [0,0.7,1],linewidth = 1.5,zorder = 1,label = 'training')
        ax.scatter([v+1  for v in num_elements[:k+1]] ,train_errors[:k+1],color = [0,0.7,1],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)

        ax.plot([v+1  for v in num_elements[:k+1]] ,valid_errors[:k+1],color = [1,0.8,0.5],linewidth = 1.5,zorder = 1,label = 'validation')
        ax.scatter([v+1  for v in num_elements[:k+1]] ,valid_errors[:k+1],color= [1,0.8,0.5],s = 70,edgecolor = 'w',linewidth = 1.5,zorder = 3)
        ax.set_title('misclassifications',fontsize = 15)

        # cleanup
        ax.set_xlabel('number of units',fontsize = 12)

        # cleanp panel                
        num_iterations = len(train_errors)
        minxc = 0.5
        maxxc = len(num_elements) + 0.5
        minc = min(min(copy.deepcopy(train_errors)),min(copy.deepcopy(valid_errors)))
        maxc = max(max(copy.deepcopy(train_errors[:10])),max(copy.deepcopy(valid_errors[:10])))
        gapc = (maxc - minc)*0.25
        minc -= gapc
        maxc += gapc
        
        ax.set_xlim([minxc,maxxc])
        ax.set_ylim([minc,maxc])
        
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        #labels = [str(v) for v in num_units]
        #ax.set_xticks(np.arange(1,len(num_elements)+1))
       # ax.set_xticklabels(num_units)


        
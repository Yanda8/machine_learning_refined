import numpy as np
from matplotlib import gridspec
from IPython.display import display, HTML
import copy
import math
import time

# import custom JS animator
from mlrefined_libraries.JSAnimation_slider_only import IPython_display_slider_only
 
# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
from matplotlib import gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

####### K-means functionality #######
# function for updating cluster assignments
def update_assignments(data,centroids):
    P = np.shape(data)[1]
    assignments = []
    for p in range(P):
        # get pth point
        x_p = data[:,p][:,np.newaxis]
        
        # compute distance between pth point and all centroids
        # using numpy broadcasting
        diffs = np.sum((x_p - centroids)**2,axis = 0)
        
        # determine closest centroid
        ind = np.argmin(diffs)
        assignments.append(ind)
    return np.array(assignments)
       
# update centroid locations
def update_centroids(data,assignments):
    K = len(np.unique(assignments))
    # new centroid container
    centroids = []
    for k in range(K):
        # collect indices of points belonging to kth cluster
        S_k = np.argwhere(assignments == k)
        
        # take average of points belonging to this cluster
        c_k = np.mean(data[:,S_k],axis = 1)
        centroids.append(c_k)
    centroids = np.array(centroids)[:,:,0]
    return centroids.T

# main k-means function
def my_kmeans(data,centroids,max_its):
    # collect all assignment and centroid updates - containers below
    all_assignments = []
    all_centroids = [centroids]
    
    # outer loop - alternate between updating assignments / centroids
    for j in range(max_its):
        # update cluter assignments
        assignments = update_assignments(data,centroids)
        
        # update centroid locations
        centroids = update_centroids(data,assignments)
        
        # store all assignments and centroids
        all_assignments.append(assignments)
        all_centroids.append(centroids)
        
    # final assignment update
    assignments = update_assignments(data,centroids)
    all_assignments.append(assignments)

    return all_centroids,all_assignments

####### K-means demo #######
def run_animated_demo(data,centroids,max_its):
    # run K-means algo
    all_centroids,all_assignments = my_kmeans(data,centroids,max_its)
    P = np.shape(data)[1]
    K = len(np.unique(all_assignments[0]))
    
    # with all centroid and assignments in hand we can go forth and animate the process
    colors =  [[1,0,0.4], [ 0, 0.4, 1],[0, 1, 0.5],[1, 0.7, 0.5],[0.7, 0.6, 0.5],'mediumaquamarine']
    
    # determine viewing range for plot
    pt_xmin = np.min(data[0,:])
    cent_xmin = np.min([c[0,:] for c in all_centroids])
    xmin = np.min([pt_xmin,cent_xmin])
    
    pt_xmax = np.max(data[0,:])
    cent_xmax = np.max([c[0,:] for c in all_centroids])
    xmax = np.max([pt_xmax,cent_xmax])    
    
    xgap = (xmax - xmin)*0.2
    xmin -= xgap
    xmax += xgap
    
    pt_ymin = np.min(data[1,:])
    cent_ymin = np.min([c[1,:] for c in all_centroids])
    ymin = np.min([pt_ymin,cent_ymin])
    
    pt_ymax = np.max(data[1,:])
    cent_ymax = np.max([c[1,:] for c in all_centroids])
    ymax = np.max([pt_ymax,cent_ymax])    
    
    ygap = (ymax - ymin)*0.2
    ymin -= ygap
    ymax += ygap
    
    # initialize figure
    fig = plt.figure(figsize = (5,5))
    artist = fig

    # create subplot with 3 panels, plot input function in center plot
    gs = gridspec.GridSpec(1, 1) 
    ax = plt.subplot(gs[0],aspect = 'equal'); 

    # start animation
    num_frames = 3*len(all_centroids)
    print ('starting animation rendering...')
    def animate(j):
        # clear panel
        ax.cla()
        
        # print rendering update
        if j == num_frames - 2:
            print ('rendering animation frame ' + str(j+1) + ' of ' + str(num_frames))
        if j == num_frames - 1:
            print ('animation rendering complete!')
            time.sleep(1.5)
            clear_output()
            
        #### plot first frame - just data #####
        # gather current centroids and assignments 
        c = int(np.floor(np.divide(j,3)))
        centroids = all_centroids[c]
        assignments = all_assignments[c] 

        # draw uncolord points
        ax.scatter(data[0,:],data[1,:],c = 'k',s = 100,edgecolor = 'w',linewidth = 1,zorder = 1) 
        title = 'unclustered data'
        ax.set_title(title,fontsize = 17)
                
        # plot the centroids 
        for k in range(K):
            ax.scatter(centroids[0,k],centroids[1,k],c = colors[k],s = 400,edgecolor ='k',linewidth = 2,marker=(5, 1),zorder = 3)
                
        # draw points and visual guides between points and their assigned cluster centroids
        if np.mod(j,3) == 1: 
            # draw visual guides
            for k in range(K):
                ind = np.argwhere(assignments == k)
                ind = [s[0] for s in ind]
                centroid = centroids[:,k]

                # connect point to cluster centroid via dashed guide line
                for i in ind:
                    pt = data[:,i]
                    ax.plot([pt[0],centroid[0]],[pt[1],centroid[1]],color = colors[k],linestyle = '--',zorder = 0,linewidth = 1)
                        
        # scatter plot each cluster of points
        if np.mod(j,3) == 2: 
            # plot the point assignments 
            for k in range(K):
                ind = np.argwhere(assignments == k)
                ind = [s[0] for s in ind]
                ax.scatter(data[0,ind],data[1,ind],c = colors[k],s = 100,edgecolor = 'k',linewidth = 1,zorder = 2) 
                        
        # set viewing range
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        
        return artist,

    anim = animation.FuncAnimation(fig, animate ,frames=num_frames, interval=num_frames, blit=True)
    return(anim)
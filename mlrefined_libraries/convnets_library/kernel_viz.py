# import standard plotting and animation
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import copy


# import other packages
import numpy as np
import cv2
from scipy import signal as sig
import time
from sklearn.preprocessing import normalize
import seaborn as sns

        
def show_conv(image_path, kernel, **kwargs):
    
    contrast_normalization = True
    if 'contrast_normalization' in kwargs:
        contrast_normalization = kwargs['contrast_normalization']
    
    # load image 
    image = cv2.imread(image_path, 0)
    
    # compute convolution
    conv = sig.convolve2d(image, np.flipud(np.fliplr(kernel)), boundary='fill', fillvalue = 0, mode='same')

    
    # initialize figure
    fig = plt.figure(figsize=(10,4))
    artist = fig
        
    # create subplot with 5 panels
    gs = gridspec.GridSpec(1, 5, width_ratios=[.4, .1, 1, .1, 1]) 
    ax1 = plt.subplot(gs[0]) # kernel 
    ax5 = plt.subplot(gs[4]); ax5.axis('off') # convolution result
    ax3 = plt.subplot(gs[2]); ax3.axis('off') # image
    ax2 = plt.subplot(gs[1]) # convolution symbol 
    ax2.scatter(0, 0, marker="$\star$", s=80, c='k'); ax2.set_ylim([-1, 1]); ax2.axis('off');
    ax4 = plt.subplot(gs[3]) # equal sign
    ax4.scatter(0, 0, marker="$=$", s=80, c='k'); ax4.set_ylim([-1, 1]); ax4.axis('off');       
    
    
    # plot convolution kernel
    cmap_kernel = ["#34495e"]       
    sns.heatmap(kernel, square=True, cbar=False, cmap=cmap_kernel,
                        annot=True, fmt=".1f", linewidths=.1, yticklabels=False, xticklabels=False,
                        annot_kws={"weight": "bold"}, ax=ax1)
    
    
    # plot input image
    ax3.imshow(image, cmap='gray')
    

    # plot convolution
    if contrast_normalization:
        conv = normalize_contrast(conv)
        
    ax5.imshow(conv, cmap=plt.get_cmap('pink'))
    plt.show()
    

def normalize_contrast(image):
    
    # linear transformation 
    a = np.min(image)
    b = np.max(image)        
    image = (image*255/(b-a))-(255*a/(b-a))
    
    # make sure all pixels are integers between 0 and 255
    eps = 1e-4
    image = np.floor(image+eps)
    
    # change data type to uint8
    image = image.astype('uint8')
    
    # equalize histogram using opencv
    image = cv2.equalizeHist(image)
    
    return image
    
   
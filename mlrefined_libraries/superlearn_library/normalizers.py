import autograd.numpy as np

###### standard normalization function ######
def standard_normalizer(x):
    # compute the mean and standard deviation of the input
    x_means = np.mean(x,axis = 1)[:,np.newaxis]
    x_stds = np.std(x,axis = 1)[:,np.newaxis]   

    # create normalizer and input normalizer functions based on mean / std
    normalizer = lambda data: (data - x_means)/x_stds

    # create inverse normalizer function 
    inverse_normalizer = lambda data: data*x_stds + x_means

    # return normalizer and inverse_normalizer
    return normalizer,inverse_normalizer
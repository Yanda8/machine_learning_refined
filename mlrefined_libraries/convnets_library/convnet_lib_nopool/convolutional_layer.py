import autograd.numpy as np
from timeit import default_timer as timer

class Setup:
    def __init__(self,kernel_sizes,**kwargs):
        # select kernel sizes and scale
        self.kernel_sizes = kernel_sizes
        self.scale = 0.1
        self.conv_stride = 2
        if 'scale' in kwargs:
            self.scale = kwargs['scale'] 
        if 'conv_stride' in kwargs:
            self.conv_stride = kwargs['conv_stride']
                
    # convolution function
    def conv_function(self,tensor_window):
        a = np.tensordot(tensor_window,self.kernels.T).T
        return a

    # activation 
    def activation(self,tensor_window):
        return np.maximum(0,tensor_window)
    
    # sliding window for image augmentation
    def sliding_window_tensor(self,tensor,window_size,stride,func):
        # grab image size, set container for results
        image_size = np.shape(tensor)[1]
        results = []
        
        # slide window over input image with given window size / stride and function
        for i in np.arange(0, image_size - window_size + 1, stride):
            for j in np.arange(0, image_size - window_size + 1, stride):
                # take a window of input tensor
                tensor_window =  tensor[:,i:i+window_size, j:j+window_size]
                
                # now process entire windowed tensor at once
                tensor_window = np.array(tensor_window)
                yo = func(tensor_window)

                # store weight
                results.append(yo)
        
        # re-shape properly
        results = np.array(results)
        results = results.swapaxes(0,1)
        if func == self.conv_function:
            results = results.swapaxes(1,2)
        return results 

    # make feature map
    def make_feature_tensor(self,tensor):
        # create feature map via convolution --> returns flattened convolution calculations
        feature_tensor = self.sliding_window_tensor(tensor,self.kernel_size,self.conv_stride,self.conv_function) 

        # shove feature map through nonlinearity
        downsampled_feature_map = self.activation(feature_tensor)

        # return downsampled feature map --> flattened
        return downsampled_feature_map

    # convolution layer
    def conv_layer(self,tensor,kernels):
        #### prep input tensor #####
        # pluck out dimensions for image-tensor reshape
        num_images = np.shape(tensor)[0]
        num_kernels = np.shape(kernels)[0]
        
        # create tensor out of input images (assumed to be stacked vertically as columns)
        tensor = np.reshape(tensor,(np.shape(tensor)[0],int((np.shape(tensor)[1])**(0.5)),int( (np.shape(tensor)[1])**(0.5))),order = 'F')
        
        # pad tensor
        kernel = kernels[0]
        self.kernel_size = np.shape(kernel)[0]
                
        #### prep kernels - reshape into array for more effecient computation ####
        self.kernels = kernels  #np.reshape(kernels,(np.shape(kernels)[0],np.shape(kernels)[1]*np.shape(kernels)[2]))
        
        #### compute convolution feature maps / downsample via pooling one map at a time over entire tensor #####
        # compute feature map for current image using current convolution kernel
        feature_tensor = self.make_feature_tensor(tensor)   
        feature_tensor = feature_tensor.swapaxes(0,1)
        feature_tensor = np.reshape(feature_tensor, (np.shape(feature_tensor)[0],np.shape(feature_tensor)[1]*np.shape(feature_tensor)[2]),order = 'F')
        
        return feature_tensor
        
    def conv_initializer(self):
        '''
        Initialization function: produces initializer to produce weights for 
        kernels and final layer touching fully connected layer
        '''
        # random initialization for kernels
        k0 = self.kernel_sizes[0]
        k1 = self.kernel_sizes[1]
        k2 = self.kernel_sizes[2]
        kernel_weights = self.scale*np.random.randn(k0,k1,k2) 
        return kernel_weights
#IMPORTS#
import numpy as np


class ANN:
    """
        Nueral Network class that will be optimised
        using stochastic optimisation techniques
    """

    def __init__(self,kernels,shape,create=True):
        """
            Initialises the nueral network class
            with convolutional and fully connected layers

            Parameters:
                kernels (list) : information about each convolutional layer
                shape   (list) : dimensions of the weight matrices in the fully connected layers
        """

        #storing information on the dimensions of kernals#
        self.kernels = kernels

        #storing information on the shape of matrices in fully connected layer#
        self.shape = shape

        #weights for the fully connected layers#
        self.W = []

        #weights for filters layers#
        self.filters = []

        #convolutional matrices#
        self.CW = []

        if create == True:

            #creating convolutional layers#
            for i in range(0,len(self.kernels),1):
                filters,CW = self.create_convolution(self.kernels[i])
                self.filters.append(filters)
                self.CW.append(CW)
            
            #creating fully connected layers#
            for i in range(1,len(shape),1):
                temp = self.create_weight(shape[i-1],shape[i])
                self.W.append(temp)
        


    def create_weight(self,u,v):
        """
            creates weight matrices including the bias
        """
        if v !=1:
            W = np.random.randn(v,u+1) * np.sqrt(2/(u+v))
        else:
            W = np.random.randn(u+1) * np.sqrt(2/(u+v))
    
        return W

######################################## CREATING CONVOLUTIONAL LAYER ###############################################################################
    def toeplitz(self,u):
        """
            generates a toeplitz matrix from 
            the vector u
        """

        #size of square matrix#
        n = len(u)

        #the matrix to be generated#
        A = np.zeros((n,n))

        #A[0,:] = u
        A[:,0] = u

        for i in range(1,n,1):
            for j in range(1,n,1):

                if not(j>i):
                    A[i,j] = u[i-j]

        return A

    def conv_matrix(self,K,dim):
        """
            Creates toeplitz matrix so convolution can be performed
            by matrix multiplication

            Parameters:
                K   (array): the filter
                dim (tuple): tuple of image dimensions
        """
        
        #getting shape of kernal#
        m1,n1 = K.shape

        #getting shape of image
        m2,n2 = dim

        K_pad = np.copy(np.pad(K,((m2-1,0),(0,n2-1)),'constant'))
        
        p1,p2 = K_pad.shape


        F_list = []

        for i in range(0,p1,1):

            F = self.toeplitz(K_pad[p1-1-i,:])
            F_list.append(F[:,0:n2])
        
        indices = self.toeplitz(np.arange(1,len(F)+1))

        #where the giant matrix will be stored#
        p1,_ = F_list[0].shape
        A = np.zeros((len(F_list)*p1,n2*n2))


        for j in range(0,n2,1):
            for i in range(j,len(F_list),1):

                M = F_list[int(indices[i,j]-1)]

                u,v = M.shape

                start_i = i*u
                end_i = start_i + u

                start_j = j*v
                end_j = start_j + v

                A[start_i:end_i, start_j:end_j] = np.copy(M)
        
        out = A[:,0:m2*n2]
        return output
    
    def create_convolution(self,kernel_info):
        """
            randomly initialises filters 
        """

        #dimensions of input#
        dim = (kernel_info[0],kernel_info[1])

        #size of filter (square matrix)#
        filer_size = kernel_info[2]

        #depth, number of channels#
        n_channels = kernel_info[3]

        filters = []

        for i in range(0,n_channels,1):
            temp_filter = np.random.randn(filer_size,filer_size)
            filters.append(temp_filter)
        
        CW = self.creat_matrix_conv(filters, dim)
        return filters,CW
        
    def creat_matrix_conv(self,filters,dim):
        """
            creates convolution im matrix form
        """

        n = len(filters)

        A = None
        updated = False

        for i in range(0,n,1):
            temp = self.conv_matrix(filters[i], dim)

            if updated == False:
                A = np.copy(temp)
                updated = True
            else:
                A = np.hstack((A,temp))
        
        return A



#testing#
if __name__ == "__main__":
    kenerls = [[10,10,3,2]]
    shape = []
import numpy as np
from  scipy import linalg as LA


def toeplitz(u):
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

            # if j>=i:
            #     A[i,j] = u[j-i]
            # else:
            #     A[i,j] = u[i-j]
            if not(j>i):
                A[i,j] = u[i-j]
    return A

def conv_matrix(K,dim):
    """
        Creates toeplitz matrix so convolution can be performed
        by matrix multiplication


    """
     
    #getting shape of kernal#
    m1,n1 = K.shape

    #getting shape of image
    m2,n2 = dim

    K_pad = np.copy(np.pad(K,((m2-1,0),(0,n2-1)),'constant'))
    
    p1,p2 = K_pad.shape


    F_list = []

    for i in range(0,p1,1):

        F = toeplitz(K_pad[p1-1-i,:])
        F_list.append(F[:,0:n2])
    
    indices = toeplitz(np.arange(1,len(F)+1))

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
    

if __name__ == "__main__":
    #filter matrix#
    I = np.array([[1,2,3],[4,5,6]])

    #Filter#
    F = np.array([[10,20],[30,40]])

    conv_matrix(F,I.shape)
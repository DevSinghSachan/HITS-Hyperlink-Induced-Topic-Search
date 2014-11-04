# HITS algorithm to compute the authority vector "A" and hub vector "H"
# a general reference for HITS algorthm is
# http://www.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture4/lecture4.html

from math import sqrt
import numpy as np
from scipy.linalg import norm
from scipy.sparse import csc_matrix

# Input to HITS algorithm is consistency matrix where entry (i,j) indicates \
# edge from i->j
# Consistency Matrix PhiMat is assumed to be a sparse matrix in CSC format

def HITS_coherence(MM):
#if __name__ == "__main__":

    # Generating dense adjacency matrix MM
    #MM = np.zeros([3, 3])
    #MM[0] = [0, 0.5, 0.125]
    #MM[1] = [0.5, 0, 0.25]
    #MM[2] = [0.125, 0.25, 0]

    if (MM.sum()==0):
        return np.array([0])

    #print MM
    # Converting dense matrix to sparse matrix
    PhiMat = csc_matrix(MM)

    # epsilon is the tolerance between the successive vectors of
    # hubs abd authorities
    epsilon = 0.0001

    # auth is a vector of authority score of dimension Mx1
    # hub is a vector of hub score of dimension Mx1
    M, N = PhiMat.shape

    # Normalizing the authorities and hubs vector by their L2 norm
    auth0 = np.ones([M, 1])
    hubs0 = np.ones([M, 1])

    hubs1 = PhiMat * auth0
    hubs1 = (1.0/norm(auth0, 2))*hubs1

    auth1 = PhiMat.transpose() * hubs1
    auth1 = (1.0/norm(hubs1, 2))*auth1

    #hubs1 = (1.0/norm(hubs1, 2))*hubs1
    #auth1 = (1.0/norm(auth1, 2))*auth1

    #print auth1
    #print hubs1
    iteration = 0
    # Calculating the hub and authority vectors until convergence
    while((norm(auth1-auth0, 2) > epsilon)or(norm(hubs1-hubs0, 2) > epsilon)):
        iteration += 1
        #print "iteration is ", iteration
        auth0 = auth1
        hubs0 = hubs1

        hubs1 = PhiMat * auth0
        hubs1 = (1.0/norm(auth0, 2))*hubs1

        auth1 = PhiMat.transpose() * hubs1
        auth1 = (1.0/norm(hubs1, 2))*auth1

        #hubs1 = (1.0/norm(hubs1, 2))*hubs1
        #auth1 = (1.0/norm(auth1, 2))*auth1

    # Printing the values of hubs and authorities vectors
    #print "authority vector is ", "\n", auth1, norm(auth1,2)
    #print "hubs vector is ", "\n", hubs1, norm(hubs1, 2)

    return auth1


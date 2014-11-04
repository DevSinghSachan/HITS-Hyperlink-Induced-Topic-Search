from scipy.sparse import csc_matrix
import numpy as np
from collections import deque, Counter
import re

# This is the consistency matrix denoising function         
def DeNoise(matList, window, threshold):

    matListOld = list()
    iteration = 0
 
    while(True):

        iteration += 1
        print "iteration number", iteration
	# Doing row and column sum of the matrices in matList
	# axis=0 is for sum of the column values
	# axis=1 is for sum of the row values
	# By default axis is None, so it sum all the entries of matrix
	rowSum = list()
	colSum = list()
	matSum = list()
	for matrix in matList:
	    rowSum.append(matrix.sum(axis=1))
	    colSum.append(matrix.sum(axis=0))
	    matSum.append(matrix.sum())

	# We will now convert raw counts to probabilities
	probMat = list()
	probRow = list()
	probCol = list() 
	for i in range(0,window-1):
	    probMat.append(matList[i]/matSum[i])
	    probRow.append(rowSum[i]/matSum[i])
	    probCol.append(colSum[i]/matSum[i])


	# We will now output the phi values ( consistency scores)
	NMIlist = list()
	for i in range(0,window-1):
	    # BoolMatrix and IdenMatrix are sparse matrices 
	    BoolMatrix = probMat[i] > 0.0
	    IdenMatrix = BoolMatrix.astype(float)

	    # Computing the denominator of PMI (i.e. RowSum*ColSum) in a sparse way
	    II = IdenMatrix.multiply(csc_matrix(probRow[i])).multiply(csc_matrix(probCol[i]))
	    # Element-wise inverse of the II matrix 
	    II.data = II.data**(-1)
	    # Computing the Pointwise Mututal Information
	    JJ = II.multiply(probMat[i])
	    JJ.data = np.log2(JJ.data)
    
	    # Computing the denominator part of Normalized PMI
	    tempProbMatrix = probMat[i].copy()
	    tempProbMatrix.data **= -1
	    tempProbMatrix.data = np.log2(tempProbMatrix.data)

	    # Computing the actual NPMI
	    tempProbMatrix.data = tempProbMatrix.data**(-1)       
	    NMIlist.append(JJ.multiply(tempProbMatrix))

            del matListOld
	    matListOld = list(matList)

	for j,phi in enumerate(NMIlist):
            
	    Iden = phi > threshold
	    matList[j] = matList[j].multiply(Iden.astype(float))
        
        diff = 0.0
        for mat1, mat2 in zip(matListOld,matList):
            diff += (mat1.sum()-mat2.sum())

        if diff==0.0:
            return NMIlist  


def SEQCONSMATRIX(line, window, _buffer_, _rare_, vocab, NMIList):

    def count_vocab2(doc):
        dictionary = dict()
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        tokens = token_pattern.findall(doc.lower())

        for item in set(tokens):
            dictionary[item] = len(dictionary)

        return dictionary

    # this function is used to generate the PHI matrix of the sequence
    def inc_stats2(q, vocab2):
        if q[0] == _buffer_: return
        token = q[0]
        for i in range(1,len(q)):
            if q[i] == _buffer_: continue
            friend = q[i]

            # Finding the global vocab id of these token and friend words
            idToken = vocab[token] if token in vocab else vocab[_rare_]
            frndToken = vocab[friend] if friend in vocab else vocab[_rare_]
            #print idToken, frndToken

            # assigning values to symmetric Phi matrix 
            PhiMatrix[vocab2[token], vocab2[friend]] += NMIList[i-1][idToken, frndToken]
            PhiMatrix[vocab2[friend], vocab2[token]] += NMIList[i-1][idToken, frndToken]
 
    # Creating the phi matrix of the sequence again
    q = deque([_buffer_ for _ in range(window-1)], window)

    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    toks = token_pattern.findall(line.lower())

    vocab2 = count_vocab2(line)
    PhiMatrix = np.zeros([len( set(toks)),len(set(toks))])

    for tok in toks:
	q.append(tok)
	inc_stats2(q, vocab2)
    for _ in range(window-1):
	q.append(_buffer_)
	inc_stats2(q, vocab2)

    return (vocab2, PhiMatrix)

# This function checks for the local maxima in the 2-D array \
# If a valid local maxima is found it return True else False. 
def IsPhrase(LatticeMatrix, i, j, StartDim, EndDim):
    if ((i-1)>=0): 
        if (LatticeMatrix[i,j] <= LatticeMatrix[i-1,j]):
	    return False
    if ((j-1)>=1):
        if (LatticeMatrix[i,j] <= LatticeMatrix[i,j-1]):
	    return False
    if ((i+1)<=StartDim-1):
        if (LatticeMatrix[i,j] <= LatticeMatrix[i+1,j]):
	    return False
    if ((j+1)<=EndDim-1):
        if (LatticeMatrix[i,j] <= LatticeMatrix[i,j+1]):
	    return False        
    return True

import re
from collections import deque, Counter
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csc_matrix
import numpy as np
from denoise import *
from hits import *

_rare_ = '<?>'
_buffer_ = '<*>'
window = 4
corpus = "news.en-00001-of-00100"
threshold = 0.3

def count_vocab(corpus):
    dictionary = dict()
    mydoclist = []
    line_no = 0

    with open(corpus, "rb") as fp:
        for line in fp:
            line_no += 1
            if(line_no % 100 == 0):
                print "Processing line number:  ", line_no

            mydoclist.append(line.strip())
      
    count_vectorizer = CountVectorizer(lowercase="True", min_df=3, ngram_range=(1, 1))
    count_vectorizer.fit_transform(mydoclist)

    del mydoclist

    for item in count_vectorizer.get_feature_names():
        dictionary[item] = len(dictionary)

    dictionary[_rare_] = len(dictionary)

    return dictionary


if __name__ == "__main__":

    vocab = count_vocab(corpus)
    print "number of words in vocabulary are:  ", len(vocab)

    num_tok = 0

    XYcount = dict()
    for i in range(1,window):
      XYcount[i] = Counter()


    def inc_stats(q):
	if q[0] == _buffer_: return
	token = q[0] if q[0] in vocab else _rare_
	for i in range(1,len(q)):
	    if q[i] == _buffer_: continue
	    friend = q[i] if q[i] in vocab else _rare_
	    XYcount[i][( vocab[token], vocab[friend] )] += 1.0

    with open(corpus, "rb") as fp:
        for line in fp:
            q = deque([_buffer_ for _ in range(window-1)], window)
	    token_pattern = re.compile(r"(?u)\b\w\w+\b")
            toks = token_pattern.findall(line.lower())
	    for tok in toks:
		num_tok += 1
		if num_tok % 1000 is 0: print 'Processed %i tokens' % (num_tok)
		q.append(tok)
		inc_stats(q)
            for _ in range(window-1):
                q.append(_buffer_)
                inc_stats(q)

    matList = list()
    for i in range(1,window):
        countXY = csc_matrix((XYcount[i].values(),
                                   zip(*XYcount[i].keys())),
                                  shape=(len(vocab), len(vocab)))
        matList.append(countXY)
            
    # Denoising the consistency matrix code
    print "Started the denoisng part of the process"
    NMIList = DENOISE(matList, window, threshold)

    # Computing the most important phrases in a line

    with open(corpus, "rb") as fp:
        for line in fp:
            #vocab2, SeqConstMatrix = SEQCONSMATRIX(line, window, _buffer_, vocab, NMIList)
            #AuthVector = HITS(SeqConstMatrix)
            
            # finding the minimum of the auhorities vector
            #AuthMin = min(AuthVector)
            #print AuthMin

            # estimating the local maxima from the consistency matrix
            token_pattern = re.compile(r"(?u)\b\w\w+\b")
            toks = token_pattern.findall(line.lower())
            print toks

            # latticematrix will store the coherence of that length of the phrase
            LatticeMatrix = np.zeros([len(toks)-1, window])

            # We index the tokens starting from 0 onwards
            # i is the index of the first token and \
            # j is the index of the last token in the sunsequence
            for i, start_token in enumerate(toks[0:-1]):
                for j in range(1, window):
                    if (i+j+1 <= len(toks)):
                        newString = " ".join(toks[i:i+j+1])
                        #print newString
                        vocab2, SeqConstMatrix = SEQCONSMATRIX(newString, window, _buffer_, vocab, NMIList)
                        AuthVector = HITS(SeqConstMatrix)

                        # finding the minimum of the auhorities vector
                        AuthMin = min(AuthVector)
                        #print AuthMin
                        LatticeMatrix[i,j] = AuthMin
                                     
            LatticeMatrix = LatticeMatrix*100000
            LatticeMatrix = LatticeMatrix.astype(int)
            print LatticeMatrix

            # Computing the validity of a phrase 
            (StartDim, EndDim) = np.shape(LatticeMatrix)

            for i in range(0, StartDim):
                for j in range(1, EndDim):
                    if(i+j+1 <= len(toks)):
                        temp = IsPhrase(LatticeMatrix, i, j, StartDim, EndDim)
                        if (temp == True):
                            print "phrase is ", " ".join(toks[i:i+j+1]) 

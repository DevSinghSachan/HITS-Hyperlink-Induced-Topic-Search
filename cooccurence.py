import re
import os
import pickle
from collections import deque, Counter
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csc_matrix
from scipy.io import mmwrite
import numpy as np
from phraseIO import *
from hits import *
from hits_coherence import *

_rare_ = '<?>'
_buffer_ = '<*>'
window = 4
corpus = "big_file.txt"
threshold = 0.4

OUTPUT_DIR = "saved_model"
PICKLE_FILE = "variables.pickle"


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

    count_vectorizer = CountVectorizer(lowercase="True", min_df=1,
                                       ngram_range=(1, 1))
    count_vectorizer.fit_transform(mydoclist)

    del mydoclist

    for item in count_vectorizer.get_feature_names():
        dictionary[item] = len(dictionary)

    dictionary[_rare_] = len(dictionary)

    return dictionary


if __name__ == "__main__":

    # Load the pickle file if it exists. It contains vocabulary hashmap and \
    # Normalized Pointwise Mutual Information(NPMI) List vector for k=1,2 and 3
    if os.path.exists(PICKLE_FILE):
        print "pickle file exists"
        with open(PICKLE_FILE, "rb") as pfile:
            pickle_dict = pickle.load(pfile)
            vocab = pickle_dict["VV"]
            NMIList = pickle_dict["UU"]

    else:
        # count_vocab function return a hash-map of unique words in the \
        # sorted in alphabetical order
        vocab = count_vocab(corpus)
        print "number of words in vocabulary are:  ", len(vocab)

        # Saving the above vocabulary in a text file
        tmp_output_dir = OUTPUT_DIR + "/" + "vocabulary"
        with open(tmp_output_dir+"/"+"vocab.txt", "wb") as fp:
            for word in vocab:
                print >> fp, word.encode("utf-8"), "\t", vocab[word]

        # Total number of tokens processed is initialized as 0.
        num_tok = 0

        # Creating the XYcount dictionary. The keys are the various values \
        # of k i.e. 1,2,3. The keys are initialized with Counter() class.
        XYcount = dict()
        for i in range(1, window):
            XYcount[i] = Counter()

        # This function increments the coocurrence counts by 1 for every \
        # when two words co-occur together for k=1,2,3
        def inc_stats(q):
            if q[0] == _buffer_:
                return
            token = q[0] if q[0] in vocab else _rare_
            for i in range(1, len(q)):
                if q[i] == _buffer_:
                    continue
                friend = q[i] if q[i] in vocab else _rare_
                XYcount[i][(vocab[token], vocab[friend])] += 1.0

        # Reading the corpus line by line and tokenizing it. Then adding the \
        # token to an initialized deque() to obtain the co-occurence score   \
        # for k=1,2,3
        with open(corpus, "rb") as fp:
            for line in fp:
                q = deque([_buffer_ for _ in range(window-1)], window)
                token_pattern = re.compile(r"(?u)\b\w\w+\b")
                toks = token_pattern.findall(line.lower())
                for tok in toks:
                    num_tok += 1
                    if num_tok % 1000 is 0:
                        print 'Processed %i tokens' % (num_tok)
                    q.append(tok)
                    inc_stats(q)
                for _ in range(window-1):
                    q.append(_buffer_)
                    inc_stats(q)

        # Initializing a list which will contain the sparse matrix of \
        # coocurrence couts. Then storing the coocurrence counts for  \
        # k=1,2,3 in sparse csc_matrix. Then saving the sparse matrix \
        # in a text file in Matrix Market Format.
        matList = list()
        for i in range(1, window):
            print "converting counter() dict into sparse csc_matrix \
                   for k = ", i
            countXY = csc_matrix((XYcount[i].values(),
                                 zip(*XYcount[i].keys())),
                                 shape=(len(vocab), len(vocab)))
            tmp_out_dir = OUTPUT_DIR + "/" + "coocurrence_counts" + "/"
            mmwrite(tmp_out_dir+"k"+str(i), countXY)
            matList.append(countXY)

        # Denoising the consistency matrix code
        print "Started the denoisng part of the process"
        NMIList = DeNoise(matList, window, threshold)

        # Writing the NMI scores to a file
        for i, item in enumerate(NMIList):
            tmp_out_dir = OUTPUT_DIR + "/" + "NMI" + "/"
            mmwrite(tmp_out_dir+"k"+str(i), item)

        # Saving the NMIList and vocabulary into a pickle file
        pickle_dict = dict(UU=NMIList, VV=vocab)
        with open("variables.pickle", "wb") as pic:
            pickle.dump(pickle_dict, pic)

    # Computing the most important phrases in a line
    with open(corpus, "rb") as fp:
        for line in fp:
            # estimating the local maxima from the consistency matrix
            token_pattern = re.compile(r"(?u)\b\w\w+\b")
            toks = token_pattern.findall(line.lower())
            # print toks

            # latticematrix will store the coherence of that length of \
            # the phrase
            LatticeMatrix = np.zeros([len(toks)-1, window])

            # We index the tokens starting from 0 onwards
            # i is the index of the first token and \
            # j is the index of the last token in the sunsequence
            for i, start_token in enumerate(toks[0:-1]):
                for j in range(1, window):
                    if (i+j+1 <= len(toks)):
                        newString = " ".join(toks[i:i+j+1])
                        # print newString
                        vocab2, SeqConstMatrix = SEQCONSMATRIX(newString,
                                window, _buffer_, _rare_, vocab, NMIList)

                        # calculating the authority vector of the \
                        # above sequence (i,j)
                        AuthVector = HITS_coherence(SeqConstMatrix)

                        # finding the minimum of the auhorities vector
                        AuthMin = min(AuthVector)
                        # print AuthMin
                        LatticeMatrix[i, j] = AuthMin

            LatticeMatrix = LatticeMatrix*100000
            LatticeMatrix = LatticeMatrix.astype(int)
            # print LatticeMatrix

            # Computing the validity of a phrase
            (StartDim, EndDim) = np.shape(LatticeMatrix)

            for i in range(0, StartDim):
                for j in range(1, EndDim):
                    if(i+j+1 <= len(toks)):
                        temp = IsPhrase(LatticeMatrix, i, j, StartDim, EndDim)
                        if temp is True:
                            print "phrase is ", " ".join(toks[i:i+j+1])

"""
The file preprocesses the files/train.txt and files/test.txt files.

I requires the dependency based embeddings by Levy et al.. Download them from his website and change 
the embeddingsPath variable in the script to point to the unzipped deps.words file.
"""
from __future__ import print_function
import numpy as np
import gzip
import os
import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

import networkx as nx
import spacy
nlp = spacy.load('en')

outputFilePath = 'pkl/sem-relations-low-dim-pi.pkl.gz'

#We download English word embeddings from here https://www.cs.york.ac.uk/nlp/extvec/
#embeddingsPath = 'embeddings/wiki_extvec.gz'
embeddingsPath = 'embeddings/glove.6B.50d.txt'

folder = 'files/'
files = [folder+'train.txt', folder+'test.txt']

#Mapping of the labels to integers
labelsMapping = {'Other':0, 
                 'Message-Topic(e1,e2)':1, 'Message-Topic(e2,e1)':2, 
                 'Product-Producer(e1,e2)':3, 'Product-Producer(e2,e1)':4, 
                 'Instrument-Agency(e1,e2)':5, 'Instrument-Agency(e2,e1)':6, 
                 'Entity-Destination(e1,e2)':7, 'Entity-Destination(e2,e1)':8,
                 'Cause-Effect(e1,e2)':9, 'Cause-Effect(e2,e1)':10,
                 'Component-Whole(e1,e2)':11, 'Component-Whole(e2,e1)':12,  
                 'Entity-Origin(e1,e2)':13, 'Entity-Origin(e2,e1)':14,
                 'Member-Collection(e1,e2)':15, 'Member-Collection(e2,e1)':16,
                 'Content-Container(e1,e2)':17, 'Content-Container(e2,e1)':18}

words = {}
maxSentenceLen = [0,0]

distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
minDistance = -30
maxDistance = 30
for dis in range(minDistance,maxDistance+1):
    distanceMapping[dis] = len(distanceMapping)

def shortestDependencyPath(pos1, pos2, sentence):
    document = nlp(unicode(sentence))
    edges = []
    sdp = None
    for token in document:
        for child in token.children:
            edges.append(('{0}'.format(token.i),'{0}'.format(child.i)))
    graph = nx.Graph(edges)
    try:
        sdp = nx.shortest_path(graph, source=str(pos1), target=str(pos2))
    except e:
        print(sentence)
    finally:
        if sdp is None:
            return []
        else:
            return map(int, sdp)


def createMatrices(file, word2Idx, maxSentenceLen=100):
    """Creates matrices for the events and sentence for the given file"""
    labels = []
    positionMatrix1 = []
    positionMatrix2 = []
    positionIndex = []
    tokenMatrix = []

    positionIndex = []
    sdpMatrix = []
    
    for line in open(file):
        splits = line.strip().split('\t')
        
        label = splits[0]
        pos1 = splits[1]
        pos2 = splits[2]
        sentence = splits[3]
        tokens = sentence.split(" ")
        
        tokenIds = np.zeros(maxSentenceLen+4)
        positionValues1 = np.zeros(maxSentenceLen)
        positionValues2 = np.zeros(maxSentenceLen)

        positionIndex1 = np.zeros(maxSentenceLen)
        positionIndex2 = np.zeros(maxSentenceLen)
        positionIndex3 = np.zeros(maxSentenceLen)
        positionIndex4 = np.zeros(maxSentenceLen)
        positionIndex5 = np.zeros(maxSentenceLen)
        positionIndex6 = np.zeros(maxSentenceLen)
        positionIndex1[int(pos1)] = 1
        positionIndex2[int(pos2)] = 1
        positionIndex3[int(pos1)-1] = 1
        positionIndex4[int(pos1)+1] = 1
        positionIndex5[int(pos2)-1] = 1
        positionIndex6[int(pos2)+1] = 1
        positionIndex.append(np.concatenate((positionIndex1, positionIndex2, positionIndex3, \
            positionIndex4, positionIndex5, positionIndex6), axis=0))

        sdpWeight = np.zeros(maxSentenceLen, dtype=np.float32)
        sdp = shortestDependencyPath(pos1, pos2, sentence)
        token_idx = 0
        for idx in range(0, min(maxSentenceLen, len(tokens))):
            if idx == int(pos1) or idx == int(pos2):
                tokenIds[token_idx] = getWordIdx("PADDING_TOKEN", word2Idx)
                token_idx += 1
                tokenIds[token_idx] = getWordIdx(tokens[idx], word2Idx)
                token_idx += 1
                tokenIds[token_idx] = getWordIdx("PADDING_TOKEN", word2Idx)
            else:
                tokenIds[token_idx] = getWordIdx(tokens[idx], word2Idx)
            
            distance1 = idx - int(pos1)
            distance2 = idx - int(pos2)
            
            if distance1 in distanceMapping:
                positionValues1[idx] = distanceMapping[distance1]
            elif distance1 <= minDistance:
                positionValues1[idx] = distanceMapping['LowerMin']
            else:
                positionValues1[idx] = distanceMapping['GreaterMax']
                
            if distance2 in distanceMapping:
                positionValues2[idx] = distanceMapping[distance2]
            elif distance2 <= minDistance:
                positionValues2[idx] = distanceMapping['LowerMin']
            else:
                positionValues2[idx] = distanceMapping['GreaterMax']
            sdpWeight[idx] = 0.3
            token_idx += 1
            
        sdpWeight[sdp] = 0.8
        sdpWeight[int(pos1)] = 0.3
        sdpWeight[int(pos2)] = 0.3
        tokenMatrix.append(tokenIds)
        positionMatrix1.append(positionValues1)
        positionMatrix2.append(positionValues2)
        
        labels.append(labelsMapping[label])
        sdpMatrix.append(sdpWeight)
    
    return np.array(labels, dtype='int64'), np.array(tokenMatrix, dtype='int64'), np.array(positionMatrix1, dtype='int64'), np.array(positionMatrix2, dtype='int64'), np.array(positionIndex, dtype='float32'), np.array(sdpMatrix, dtype='float32')


def getWordIdx(token, word2Idx): 
    """Returns from the word2Idex table the word index for a given token"""       
    if token in word2Idx:
        return word2Idx[token]
    elif token.lower() in word2Idx:
        return word2Idx[token.lower()]
    
    return word2Idx["UNKNOWN_TOKEN"]

for fileIdx in range(len(files)):
    file = files[fileIdx]
    for line in open(file):
        splits = line.strip().split('\t')
        
        label = splits[0]
        
        sentence = splits[3]        
        tokens = sentence.split(" ")
        maxSentenceLen[fileIdx] = max(maxSentenceLen[fileIdx], len(tokens))
        for token in tokens:
            words[token.lower()] = True
            

print("Max Sentence Lengths: ", maxSentenceLen)
        
# :: Read in word embeddings ::
# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []

# :: Downloads the embeddings from the York webserver ::
if not os.path.isfile(embeddingsPath):
    basename = os.path.basename(embeddingsPath)
    if basename == 'wiki_extvec.gz':
           print("Start downloading word embeddings for English using wget ...")
           #os.system("wget https://www.cs.york.ac.uk/nlp/extvec/"+basename+" -P embeddings/")
           os.system("wget https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_english_embeddings/"+basename+" -P embeddings/")
    else:
        print(embeddingsPath, "does not exist. Please provide pre-trained embeddings")
        exit()
        
# :: Load the pre-trained embeddings file ::
#fEmbeddings = gzip.open(embeddingsPath, "r") if embeddingsPath.endswith('.gz') else open(embeddingsPath, encoding="utf8")
fEmbeddings = open(embeddingsPath, "r")
	
print("Load pre-trained embeddings file")
for line in fEmbeddings:
    split = line.decode('utf-8').strip().split(" ")
    word = split[0]
    
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)
        
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)

    if word.lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[word] = len(word2Idx)

wordEmbeddings = np.array(wordEmbeddings)

print("Embeddings shape: ", wordEmbeddings.shape)
print("Len words: ", len(words))

# :: Create token matrix ::
train_set = createMatrices(files[0], word2Idx, max(maxSentenceLen))
test_set = createMatrices(files[1], word2Idx, max(maxSentenceLen))

data = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx, 
        'train_set': train_set, 'test_set': test_set}

f = gzip.open(outputFilePath, 'wb')
pkl.dump(data, f)
f.close()

print("Data stored in pkl folder")        

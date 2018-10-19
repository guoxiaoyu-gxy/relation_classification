# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import gzip

import os
import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl


batch_size = 16
nb_epoch = 300

embedding_dims = 300
position_dims = 50
class_dims = 1000

lstm_input_dims = embedding_dims
lstm_output_dims = 300

conv_dims = 1*embedding_dims + 0*position_dims + 2*lstm_output_dims

log_interval = 10

penalty = 0

print("Load dataset")
f = gzip.open('./pkl/sem-relations-sdp-2018-1.2.pkl.gz', 'rb')
data = pkl.load(f)
f.close()

embeddings = data['wordEmbeddings']
yTrain, sentenceTrain, resentenceTrain, positionTrain1, positionTrain2, \
        repositionTrain1, repositionTrain2, sdpTrain, resdpTrain = data['train_set']
yTest, sentenceTest, resentenceTest, positionTest1, positionTest2, \
        repositionTest1, repositionTest2, sdpTest, resdpTest = data['test_set']

n_out = 6
max_sentence_len = sentenceTrain.shape[1]
max_position = int(max(np.max(positionTrain1), np.max(positionTrain2)) + 1)

print("sentenceTrain: ", sentenceTrain.shape)
print("positionTrain2: ", positionTrain2.shape)
print("yTrain: ", yTrain.shape)

print("sentenceTest: ", sentenceTest.shape)
print("positionTest1: ", positionTest1.shape)
print("positionTest2: ", positionTest2.shape)
print("yTest: ", yTest.shape)

print("Embeddings: ", embeddings.shape)

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D
import random
embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1]).cuda()
embedding.weight.data.copy_(torch.from_numpy(embeddings))

class CnnOneAttNet(nn.Module):
    def __init__(self):
        super(CnnOneAttNet, self).__init__()

        #bi-lstm layer
        self.lstm = nn.GRU(lstm_input_dims, lstm_output_dims, batch_first=True, dropout=0, bidirectional=True)

        #conv
        self.conv1 = nn.Conv1d(conv_dims, class_dims, kernel_size=1, padding=0)
        self.norm1 = nn.BatchNorm1d(class_dims)
        nn.init.xavier_normal(self.conv1.weight, gain=np.sqrt(1.0))
        nn.init.constant(self.conv1.bias, 0)
        
        self.drope = nn.Dropout(0.5)
        self.drop = nn.Dropout(0.5)
        self.norm = nn.BatchNorm1d(max_sentence_len)

        self.U = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(embedding_dims, class_dims)), requires_grad=True)
        self.Up = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(class_dims, class_dims)), requires_grad=True)
        self.W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(class_dims, n_out)), requires_grad=True)
        self.M = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(max_sentence_len, n_out)), requires_grad=True)


    def forward(self, words, pos1, pos2, sdp, labels, mask, idx, indices):
        embedw = self.drope(embedding(words))
        
        ##lstm
        output, (hn, cn) = self.lstm(embedw)
        output1, output2 = output.chunk(2, dim=2)
        
        sdp = F.relu((sdp.float()*0.8 + 0.8) / 1.0)
        u = sdp.view(-1, max_sentence_len, 1)
        
        H = torch.cat((u*output1, u*embedw, u*output2), 2).permute(0, 2, 1)
        x1 = F.tanh(self.conv1(H))
        G = torch.matmul(x1.permute(0, 2, 1), self.Up)
        G = torch.matmul(G, self.W)
        A = F.softmax(G, 1)
        x1 = torch.max(torch.bmm(x1, A), 2)[0].view(x1.shape[0], -1)
        x1 = torch.max(F.tanh(self.conv1(H)), 2)[0].view(x1.shape[0], -1)

        x = self.drop(x1)

        s1 = self.W[:,labels]
        s1 = torch.diag(torch.matmul(x, s1))
        s1 = torch.log(1 + torch.exp(2 * (2.5 - s1)))
        #s1 = s1.masked_fill_(idx.byte(), 0)
        
        s2 = torch.zeros_like(s1)

        x = torch.matmul(x, self.W)
        s4 = torch.masked_select(x, mask.byte()).view(x.shape[0], -1)
        #s3 = torch.index_select(s4, 1, indices)
        #s3 = s3 * (1 - idx).view(-1, 1).float()
        s3 = torch.max(s4, 1)[0]
        s3 = torch.log(1 + torch.exp(2 * (0.5 + s3)))
        s2 = s2.add_(s3)

        #s3 = s4 * idx.view(-1, 1).float()
        #s3 = torch.max(s3, 1)[0]
        #s3 = torch.log(1 + torch.exp(2 * (0.5 + s3)))
        #s2 = s2.add_(s3)
        return s1, s2, x

model = CnnOneAttNet().cuda()
#optimizer = optim.Adadelta(model.parameters(), lr=1, weight_decay=1e-4)
optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-2)
#optimizer = optim.Adagrad(model.parameters(), lr=1e-2, lr_decay=0.5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[280,290], gamma=0.1)
print(model)

train_cat = torch.from_numpy(np.concatenate((sentenceTrain, resentenceTrain, positionTrain1, positionTrain2, \
        repositionTrain1, repositionTrain2, sdpTrain, resdpTrain), axis=1))
train_datasets = D.TensorDataset(data_tensor=train_cat, target_tensor=torch.LongTensor(yTrain))
train_dataloader = D.DataLoader(train_datasets, batch_size, True)

test_cat = torch.from_numpy(np.concatenate((sentenceTest, resentenceTest, positionTest1, positionTest2, \
        repositionTest1, repositionTest2, sdpTest, resdpTest), axis=1))
test_datasets = D.TensorDataset(data_tensor=test_cat, target_tensor=torch.LongTensor(yTest))
test_dataloader = D.DataLoader(test_datasets, batch_size, False)


def data_unpack(cat_data, target):
    list_x = np.split(cat_data.numpy(), [max_sentence_len, max_sentence_len*2, \
            max_sentence_len*3, max_sentence_len*4, max_sentence_len*5, max_sentence_len*6, \
            max_sentence_len*7, max_sentence_len*8], 1)
    sen = Variable(torch.from_numpy(list_x[0])).cuda()
    resen = Variable(torch.from_numpy(list_x[1])).cuda()
    pos1 = Variable(torch.from_numpy(list_x[2])).cuda()
    pos2 = Variable(torch.from_numpy(list_x[3])).cuda()
    repos1 = Variable(torch.from_numpy(list_x[4])).cuda()
    repos2 = Variable(torch.from_numpy(list_x[5])).cuda()
    sdp = Variable(torch.from_numpy(list_x[6])).cuda()
    resdp = Variable(torch.from_numpy(list_x[7])).cuda()
    list_y = np.split(target.numpy(), [1, 2], 1)
    label = Variable(torch.from_numpy(list_y[0]).squeeze(1)).cuda()
    relabel = Variable(torch.from_numpy(list_y[1]).squeeze(1)).cuda()

    idx = np.ones((sen.shape[0], n_out))
    idx[np.arange(idx.shape[0]), list_y[0].reshape(1,-1)] = 0
    mask = Variable(torch.from_numpy(idx)).cuda()
    idx = np.zeros((sen.shape[0]))
    idx[np.where(np.array(list_y[0]).reshape(-1)==0)] = 1
    idx = Variable(torch.from_numpy(idx)).cuda()
    indices = Variable(torch.LongTensor([1, 2, 3, 4, 5])).cuda()
    return sen, resen, pos1, pos2, repos1, repos2, sdp, resdp, label, relabel, mask, idx, indices


def data_unpack_reverse(cat_data, target):
    list_x = np.split(cat_data.numpy(), [max_sentence_len, max_sentence_len*2, \
            max_sentence_len*3, max_sentence_len*4, max_sentence_len*5, max_sentence_len*6, \
            max_sentence_len*7, max_sentence_len*8], 1)
    sen = Variable(torch.from_numpy(np.concatenate((list_x[0], list_x[1]), axis=0))).cuda()
    resen = Variable(torch.from_numpy(list_x[1])).cuda()
    pos1 = Variable(torch.from_numpy(np.concatenate((list_x[2], list_x[4]), axis=0))).cuda()
    pos2 = Variable(torch.from_numpy(np.concatenate((list_x[3], list_x[5]), axis=0))).cuda()
    repos1 = Variable(torch.from_numpy(list_x[4])).cuda()
    repos2 = Variable(torch.from_numpy(list_x[5])).cuda()
    sdp = Variable(torch.from_numpy(np.concatenate((list_x[6], list_x[7]), axis=0))).cuda()
    resdp = Variable(torch.from_numpy(list_x[7])).cuda()
    list_y = np.split(target.numpy(), [1], 1)
    label = Variable(torch.from_numpy(np.concatenate((list_y[0], list_y[1]), axis=0)).squeeze(1)).cuda()
    relabel = Variable(torch.from_numpy(list_y[1]).squeeze(1)).cuda()
    
    idx = np.ones((sen.shape[0], n_out))
    idx[np.arange(idx.shape[0]), np.array(target.numpy()).reshape(-1)] = 0
    mask = Variable(torch.from_numpy(idx)).cuda()
    idx = np.zeros((sen.shape[0]))
    idx[np.where(np.array(list_y).reshape(-1)==1)] = 1
    idx = Variable(torch.from_numpy(idx)).cuda()
    indices = Variable(torch.LongTensor([1, 2, 3, 4, 5])).cuda()
    return sen, resen, pos1, pos2, repos1, repos2, sdp, resdp, label, relabel, mask, idx, indices

def train(epoch):
    model.train()
    correct = 0
        
    for i, (sen, lab) in enumerate(train_dataloader):
        sentences, resentences, pos1, pos2, repos1, repos2, \
                sdp, resdp, labels, relabels, mask, idx, indices = data_unpack(sen, lab)
        output1, output2, output = model(sentences, pos1, pos2, sdp, labels, mask, idx, indices)
        #print(output1.view(1, -1), output2.view(1, -1))
        loss = torch.mean(output1 + output2)
        #loss = torch.sum(torch.log(1 + torch.exp(2*(2-output1))) + torch.log(1 + torch.exp(2*(0.5+output2))))
        #print(output1.view(1, -1), output2.view(1, -1))
        #loss = F.cross_entropy(output1, labels)# + F.cross_entropy(output2, relabels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #pred = output.data.max(1, keepdim=True)[1]
        value, pred = output.data.max(1, keepdim=True)
        #for idx, num in enumerate(value):
        #    if num.cpu().numpy() <= penalty:
        #        pred[idx] = 0
        correct += pred.eq(labels.data.view_as(pred)).long().cpu().sum()
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, int(i) * len(sentences), sentenceTrain.shape[0],
                100. * int(i) * len(sentences) / len(sentenceTrain), loss.data[0]))
    print('\nTrain set: Accuracy:{:.4f}%'.format(100. * correct / sentenceTrain.shape[0]))

#from sklearn.metrics import precision_recall_fscore_support
max_prec, max_rec, max_acc, max_f1 = 0, 0, 0, 0
max_epoch = 1
#labelsMapping = {0:'ANY', 1:'USAGE', 2:'USAGE', 3:'PART_WHOLE', 4:'PART_WHOLE', 
#	5:'MODEL-FEATURE', 6:'MODEL-FEATURE', 7:'RESULT', 8:'RESULT', 9:'COMPARE', 
#	10:'COMPARE', 11:'TOPIC', 12:'TOPIC'}

#labelsMapping = {0:'ANY', 1:'USAGE', 2:'PART_WHOLE', 3:'MODEL-FEATURE', 4:'RESULT', 5:'COMPARE', 6:'TOPIC'}

labelsMapping = {0:'USAGE', 1:'PART_WHOLE', 2:'MODEL-FEATURE', 3:'RESULT', 4:'COMPARE', 5:'TOPIC'}

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    prediction = np.zeros(yTest.shape[0])
       
    for i, (sen, lab) in enumerate(test_dataloader):
        sentences, resentences, pos1, pos2, repos1, repos2, \
                sdp, resdp, labels, relabels, mask, idx, indices = data_unpack(sen, lab)
        output1, output2, output = model(sentences, pos1, pos2, sdp, labels, mask, idx, indices)
        #test_loss += F.cross_entropy(output, labels).data[0]
        value, pred = output.data.max(1, keepdim=True)
        #for idx, num in enumerate(value):
        #    if num.cpu().numpy() <= penalty:
        #        pred[idx] = 0
        correct += pred.eq(labels.data.view_as(pred)).long().cpu().sum()
        prediction[i*batch_size:min((i+1)*batch_size,yTest.shape[0])] = pred.squeeze(1).cpu().numpy()

    #test_loss /= sentenceTest.shape[0]
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, sentenceTest.shape[0],
    #    100. * correct / sentenceTest.shape[0]))
    global max_acc, max_rec, max_prec, max_f1, max_f1_epoch, max_prec_epoch
    
    #err = 0
    #for i in range(len(prediction)-1):
    #    if yTest[:,0].squeeze()[i] != prediction[i] and prediction[i] == 0:
    #        err += 1
    #if max_prec > 0.85:
    #    print(err, prediction[err])
    #print("Err Other:", err)

    idx = np.where(yTest[:,0]!=0)
    #prec, rec, f1, acc = precision_recall_fscore_support(yTest[idx,0].squeeze(), prediction[idx], average='macro')
    #prec, rec, f1, acc = precision_recall_fscore_support(yTest[:,0].squeeze(), prediction, average='macro')
    #if max_f1 < f1:
    #    max_f1_epoch = epoch + 1
    #if max_prec < prec:
    #    max_prec_epoch = epoch + 1
    #max_acc = max(max_acc, acc)
    #max_prec = max(max_prec, prec)
    #max_rec = max(max_rec, rec)
    #max_f1 = max(max_f1, f1)
    #print(prec, f1)
    entity_pair = []
    with open('./answer/1.2.test.relations.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            entity_pair.append(line.strip())
    with open('./answer/propose_answer.txt', 'w') as f:
        for i, num in enumerate(prediction):
            f.write(str(labelsMapping[num])+entity_pair[i]+'\n')

print("Start training")
if not os.path.isfile('./answer/keys.test.1.2.txt'):
    with open('./answer/answer_keys.txt', 'w') as f:
        for i, num in enumerate(yTest[:,0].squeeze()):
            f.write(str(int(i+1))+'\t'+str(labelsMapping[num])+'\n')

import re
for epoch in range(nb_epoch):
    scheduler.step()
    train(epoch)
    test(epoch)
    #print("Max f1 epoch: %d" % max_f1_epoch)
    #print("Max pre epoch: %d" % max_prec_epoch)
    #print("Max f1: %.4f" % max_f1)
    #print("Max precision: %.4f" % max_prec)
    #print("Max recall: %.4f" % max_rec)
    
    os.system('./answer/eval.pl ./answer/propose_answer.txt ./answer/keys.test.1.2.txt > ./answer/result_2018.txt')
    best = 0
    with open('./answer/result_2018_best.txt', 'r') as f:
        lines = f.readlines()
        last = lines[-2]
        best = re.compile(r'[1-9]\d*\.\d*|0\.\d*[1-9]\d*$').findall(last)[0]
    with open('./answer/result_2018.txt', 'r') as f:
        lines = f.readlines()
        last = lines[-2]
        acc = re.compile(r'[1-9]\d*\.\d*|0\.\d*[1-9]\d*$').findall(last)[0]
        if float(acc) > float(best):
            os.system('mv ./answer/result_2018.txt ./answer/result_2018_best.txt')
        max_acc = max(max_acc, float(acc))
    print("Max Official: %.2f\n" % max_acc)

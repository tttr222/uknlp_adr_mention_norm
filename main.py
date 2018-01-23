#!/usr/bin/env python
import sys, os, random, pickle, json, codecs, re
import numpy as np
import pandas as pd
from CharLSTM import CharacterLSTM
import sklearn.metrics as skm

num_ensembles = 10
num_epoch = 40
batch_size = 8

def main(args):
    trainset = load_dir('data_train',load_labels=True)
    X_train = [ text.split(' ') for text in trainset['text'].tolist() ]
    y_train = trainset['label'].tolist()
    print "Training", len(X_train), len(y_train)
    
    testset = load_file('data_test/task_3_test1_amiaformat_to_release.txt',load_labels=False)
    X_test = [ text.split(' ') for text in testset['text'].tolist() ]
    id_test = testset['id'].tolist()
    print "Testing", len(X_test)
    
    labels = sorted(set(y_train))
    
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    proba_cumulative = np.zeros((len(X_test),len(labels)))
    
    for j in range(num_ensembles):
        trainset = random.sample(zip(X_train, y_train), len(X_train))
        Mx, My = zip(*trainset)
        X_trainf, y_trainf = Mx[:-1000], My[:-1000]
        X_devf, y_devf = Mx[-1000:], My[-1000:]
        model_proba = train_task(labels, X_trainf, y_trainf, X_devf, y_devf, X_test, j)
        proba_cumulative += model_proba
    
    proba_cumulative = proba_cumulative/num_ensembles
    y_pred = np.argmax(proba_cumulative,axis=1)
    
    prediction = [ labels[y] for y in y_pred ]
    print "Predictions:", len(prediction)
    print "id_test:", len(id_test)
    with open('test_hiercharlstm.txt','w') as f:
        for pid, label in zip(id_test,prediction):
            print >> f, "{}\t{}".format(pid,label)

def train_task(labels, X_train, y_train, X_dev, y_dev, X_test, seed_state):
    model = CharacterLSTM(labels,learning_rate=0.01,decay_rate=0.95)
    model.fit(X_train,y_train, X_dev, y_dev, num_epoch=num_epoch,seed=seed_state,batch_size=batch_size)
    
    probas = []
    for i in range(0,len(X_test),batch_size):
        probas.append(model.predict_proba(X_test[i:i+batch_size]))
    
    return np.concatenate(probas,axis=0)

def load_file(fname, load_labels = False):
    if load_labels:
        columns = ('id','text','label')
    else:
        columns = ('id','text')
    
    frame = pd.read_csv(fname,header=None,names=columns,
                        usecols=columns,delimiter='\t')
                        
    return frame
    
def load_dir(dirname, load_labels = True):
    frames = []
    for name in os.listdir(dirname):
        if name.endswith('amiaformat.txt'):
            frames.append(load_file(os.path.join(dirname,name),load_labels))
            
    return pd.concat(frames)

if __name__ == '__main__':
    random.seed(0)
    main(sys.argv)

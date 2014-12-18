# -*- coding: utf-8 -*-
"""
Created on Sat Dec 06 21:04:43 2014

@author: Ken

hpocf: hyper parameter optimization code fold
"""


import hyperopt
from hyperopt import hp
import mlr.learn.codefold as codefold
from mlr.datagen.treevector import TreeVector
import pylearn2.train as train
from mlr.utils.notheano import SimpleList
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import EpochCounter
from pylearn2.training_algorithms.learning_rule import AdaDelta
from hyperopt.pyll.stochastic import sample
from hyperopt import  fmin, tpe, hp, STATUS_OK, Trials
import time


def cfGen(data,frae,tv,depth):
    cfm = codefold.CodeFoldModel(frae,tv)    
    cfc = codefold.CodeFoldCost(data,depth)
    return (cfm,cfc)    

def trainFrae(space,data,frae,tv,depth):
    cfm,cfc = cfGen(data,frae,tv,depth)
    trainds = SimpleList(1000,space['samples'])
    trainms = SimpleList(1000,1100)    
    testms = SimpleList(0,100)
    validms = SimpleList(900,1000)
    ms = {'train':trainms,'test':testms,'valid':validms}
    tc = EpochCounter(space['epoch'])
    lr = AdaDelta()
    sgd = SGD(
                learning_rate=space['lrate'],
                batch_size=space['bsize'],
                learning_rule=lr,
                monitoring_dataset=ms,
                cost=cfc,
                termination_criterion=tc
             )
    trainobj = train.Train(
        dataset=trainds,
        model=cfm,
        algorithm=sgd
    )
    trainobj.main_loop()
    return trainobj

def opt(space):
    print space
    global W
    global data
    global frae
    global tv
    global depth
    t0 = time.time()

    frae.fc.W[:] = W[:]     
    trainobj = trainFrae(space,data,frae,tv,depth)

    testobj = trainobj.algorithm.monitor.channels['test_objective'].val_record
    t1 = time.time()
    improved = testobj[-1]
    elapsed = t1-t0
    loss = improved*improved*elapsed
    #loss = testobj[-1]
    print 'loss,elapsed,improved: ', loss,elapsed,improved
    status = STATUS_OK
    return {'loss':loss,'status':status,'elapsed':elapsed,'improved':improved}

def optCodeFold(numEvals):


    space = {
                'epoch': 3+hp.randint('epoch',6),
                'samples': 10000+5000*hp.randint('samps',3),
                'lrate': 10**-(7+hp.randint('lrate',3)),
                'bsize': 2+(hp.randint('bsize',3)),

            }


    trials = Trials()
    best = fmin(opt,space=space,algo=tpe.suggest,max_evals=numEvals,trials=trials)
    print 'best: ',best
        
    return trials
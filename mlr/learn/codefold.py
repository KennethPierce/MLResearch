# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:03:18 2014

@author: Ken
"""
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX

import mlr.utils.notheano as notheano
from mlr.utils.tree import Tree
import mlr.costs.unfolder as unfolder
from mlr.datagen.treevector import TreeVector

from scipy.optimize import check_grad,approx_fprime
import numpy

from scipy.optimize import minimize
import time

   


class CodeFoldModel(Model):
    """
    This class is serialized to disk.
    """
    def __init__(self,toInput,treeVector):
        """
        toInput: Frae used to make input a proper tree
        toLearn: Frae being learned.  If not given, learn with toInput
        """
        super(CodeFoldModel,self).__init__()
        
        self.toInput = toInput
        self.toLearn = toInput
        self.tv = treeVector

        self.W= sharedX(self.toLearn.fc.W,name='W',borrow=True)
        self._params=[self.W]
        self.input_space=VectorSpace(dim=1)


class CodeFoldCost(notheano.Cost):
    """
    Fold algo can't use theano directly.  
    """
    def __init__(self,dataset,treeDepth):
        """
        dataset: list of trees<ints> to be trained with
        treeVector: ints->vectors
        treeDepth: all inputed trees must be exactly this depth
        """
        assert isinstance(dataset,list)
        self.ds = dataset
        self.treeDepth = treeDepth

    def prepInput(self,model,tree):        
        assert tree.depth == self.treeDepth
        tvtree = model.tv.convertTree(tree)
        return model.toLearn.prepInput(tvtree)
        
    
    def wrap(self,model,dataIdxs,funct):
        dataIdxs = notheano.SliceData(dataIdxs)
        trees = [self.ds[i] for i in dataIdxs]
        pis =  [self.prepInput(model,i) for i in trees]
        val = [funct(*i) for i in pis]
        s =  sum(val)/len(dataIdxs)
        return s
        
    
    def cost(self,model,dataIdxs):
        fc,fg = model.toLearn.costTrees[self.treeDepth]
        c =  self.wrap(model,dataIdxs,fc)
        assert type(c) == numpy.float64        
        return numpy.array(c)
        
    def grad(self,model,dataIdxs):
        fc,fg = model.toLearn.costTrees[self.treeDepth]
        g =  self.wrap(model,dataIdxs,fg)
        assert g.shape==model.toLearn.fc.W.shape 
        return g
        
class CodeFoldCostNumpy(notheano.Cost):
    """
    Fold algo can't use theano directly.  
    """
    def __init__(self,dataset):
        """
        dataset: list of trees<ints> to be trained with
        treeVector: ints->vectors
        """
        assert isinstance(dataset,list)
        self.ds = dataset
        self.grad = self.fast_grad
    
    def prepInput(self,model,idx):
        assert idx < len(self.ds)
        tree,meta = self.ds[idx]
        assert(isinstance(tree,Tree))
        btree = unfolder.TreeToFraeTree(model.toInput.fc).binarySplit(tree)
        vtree = model.tv.convertTree(btree)        
        return vtree
    
    def cost(self,model,dataIdxs): 
        dataIdxs = notheano.SliceData(dataIdxs)
        btrees = [self.prepInput(model,i) for i in dataIdxs]
        costs = [model.toLearn.costTree(i) for i in btrees]
        return sum(costs)/len(dataIdxs)
        
    def fast_grad(self,model,dataIdxs):    
        dataIdxs = notheano.SliceData(dataIdxs)
        btrees = [self.prepInput(model,i) for i in dataIdxs]
        grads = [model.toLearn.d_costTree(i)[0] for i in btrees]
        g = sum(grads)/len(dataIdxs)
        assert g.shape==model.toLearn.fc.W.shape
        return g
        
        
    def verify_grad(self,model,dataIdxs):        
        def c(w):            
            f = unfolder.FraeNumpy(unfolder.MatrixFold(model.toLearn.fc.size))
            m = CodeFoldModel(f,model.tv)
            m.toLearn.fc.W[:] = w[:]
            return self.cost(m,dataIdxs)
        def g(w):
            f = unfolder.FraeNumpy(unfolder.MatrixFold(model.toLearn.fc.size))
            m = CodeFoldModel(f,model.tv)
            m.toLearn.fc.W[:] = w[:]
            return self.fast_grad(m,dataIdxs)
            
        err = check_grad(c,g,model.toLearn.fc.W)        
        grad = approx_fprime(model.toLearn.fc.W,c,1e-8)
        return grad
        
def minCodeFold_(data,size=50,mi=10,cfm=None):
    """
    Quick test of l-bfgs-b performance
    """
    depth = data[0].depth
    if not cfm:
        print 'time: ',time.time()    
        toInput = unfolder.Frae(unfolder.MatrixFold(size),depth+1)
        print 'time: ',time.time()
        tv = TreeVector(size)
        cfm = CodeFoldModel(toInput,tv)
    frae = cfm.toLearn
    mf = frae.fc
    cfc = CodeFoldCost(data,depth)
    dl = [[float(i)] for i in range(len(data))]
    def cost(w):
        mf.W[:] = w[:]
        return cfc.cost(cfm,dl)
        pass
    def grad(w):
        mf.W[:] = w[:]        
        return cfc.grad(cfm,dl)
        pass
    def cb(w):
        print '.',
        pass
    #print "cost: ",cost(mf.W)
    t1 = time.time()
    print 'time: ', time.ctime()
    res = minimize(cost, mf.W, method='L-BFGS-B', jac = grad, options = {'maxiter':mi},callback=cb)
    print
    print "cost: ",res.fun
    t2 = time.time()
    print 'elapsed: ', t2-t1
    return res,cfm
    

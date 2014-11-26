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

from scipy.optimize import check_grad,approx_fprime
import numpy

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
            f = unfolder.Frae(unfolder.MatrixFold(model.toLearn.fc.size))
            m = CodeFoldModel(f,model.tv)
            m.toLearn.fc.W[:] = w[:]
            return self.cost(m,dataIdxs)
        def g(w):
            f = unfolder.Frae(unfolder.MatrixFold(model.toLearn.fc.size))
            m = CodeFoldModel(f,model.tv)
            m.toLearn.fc.W[:] = w[:]
            return self.fast_grad(m,dataIdxs)
            
        err = check_grad(c,g,model.toLearn.fc.W)        
        grad = approx_fprime(model.toLearn.fc.W,c,1e-8)
        return grad
        
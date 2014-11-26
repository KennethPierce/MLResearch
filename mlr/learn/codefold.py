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
    
    def prepInput(self,model,dataIdxs):
        dataIdxs = notheano.SliceData(dataIdxs)
        assert len(dataIdxs)==1 # batchsize one only
        idx = dataIdxs[0]
        assert idx < len(self.ds)
        tree,meta = self.ds[idx]
        assert(isinstance(tree,Tree))
        btree = unfolder.TreeToFraeTree(model.toInput.fc).binarySplit(tree)
        vtree = model.tv.convertTree(btree)        
        return vtree
    
    def cost(self,model,dataIdxs): 
        btree = self.prepInput(model,dataIdxs)
        return  model.toLearn.costTree(btree)
        
    def grad(self,model,dataIdxs):    
        btree = self.prepInput(model,dataIdxs)
        g,_,_ =  model.toLearn.d_costTree(btree)
        assert g.shape==model.toLearn.fc.W.shape
        return g
        
        
        
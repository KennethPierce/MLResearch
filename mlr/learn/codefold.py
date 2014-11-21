# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:03:18 2014

@author: Ken
"""
from pylearn2.models.model import Model
import mlr.utils.notheano as notheano
import mlr.costs.unfolder as unfolder

class CodeFoldModel(Model):
    """
    This class is serialized to disk.
    """
    def __init__(self,toInput):
        """
        toInput: Frae used to make input a proper tree
        toLearn: Frae being learned.  If not given, learn with toInput
        """
        super(CodeFoldModel,self).__init__()
        self.toInput = toInput
        self.toLearn = toInput

        
class CodeFoldCost(notheano.Cost):
    """
    Fold algo can't use theano directly.  
    """
    def __init__(self,dataset,treeVector):
        """
        dataset: list of trees<ints> to be trained with
        treeVector: ints->vectors
        """
        assert isinstance(dataset,list)
        self.ds = dataset
        self.tv = treeVector
    
    def prepInput(self,model,dataIdxs):
        dataIdxs = notheano.SliceData(dataIdxs)
        assert len(dataIdxs)==1 # batchsize one only
        tree = self.ds[dataIdxs[0]]
        vtree = self.tv.convertTree(tree)
        btree = unfolder.TreeToFraeTree(model.toInput.fc).Greedy(vtree)
        return btree
    
    def cost(self,model,dataIdxs): 
        btree = self.prepInput(model,dataIdxs)
        return  model.toLearn.costTree(btree)
        
    def grad(self,model,dataIdxs):    
        btree = self.prepInput(model,dataIdxs)
        return model.toLearn.d_costTree(btree)
        
        
        
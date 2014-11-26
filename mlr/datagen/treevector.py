# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 14:08:54 2014

@author: Ken
"""

from mlr.utils.tree import Tree     
import numpy
class TreeVector():
    def __init__(self,numFeat,numEx=0):
        self.rand = numpy.random.rand
        self.numFeat = numFeat
        self.vectors = self.rand(1,self.numFeat)
        self.growVectors(numEx)
        
    def convertTree(self,inTree):
        """
        convert inTree<int> to inTree<vectors>
        inTree: tree to convert
        returns: Tree() with same structure
        """
        ns=None
        if not inTree.isLeaf:
            ns = [self.convertTree(i) for i in inTree.ns]
        return Tree(self.getVector(inTree.v),ns)
    
    def getVector(self,i):
        """get i-th vector"""
        if i == None:
            return None
        assert i >= 0
        self.growVectors(i)
        return self.vectors[i:(i+1),:]
    
    def growVectors(self,num):
        """grow vectors to accomodate num"""
        rows = self.vectors.shape[0]
        if num >= rows:
            r = self.rand(1+num-rows,self.numFeat)            
            self.vectors = numpy.concatenate([self.vectors,r])
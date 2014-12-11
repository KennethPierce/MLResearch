# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:07:59 2014

@author: Ken
"""

import unittest
import mlr.learn.codefold as codefold
import mlr.costs.unfolder as unfolder
from mlr.datagen.treevector import TreeVector
from mlr.utils.tree import Tree
import numpy

class TestCodeFold(unittest.TestCase):
    
    def setUp(self):
        self.data = [Tree(3,[Tree(2,None),Tree(5,None)]),
                     Tree(3,[Tree(2,None),Tree(5,None)]),
                     Tree(4,[Tree(7,None),Tree(6,None)])]
        self.depth = self.data[0].depth
        self.isize =3
        self.tvi = TreeVector(self.isize)
        self.mf = unfolder.MatrixFold(self.isize)
    
    def cgNumpy(self,frae,idx):
        #numpy and theano impls don't match with same mf due to different cost function
        #numpy doesn't score intermediate nodes while unfolding - theano does
        model = codefold.CodeFoldModel(frae,self.tvi)
        cost = codefold.CodeFoldCostNumpy([(i,None) for i in self.data])
        return(cost.cost(model,idx),cost.grad(model,idx))
        
    def cgTheano(self,frae,idx):

        model = codefold.CodeFoldModel(frae,self.tvi)
        cost = codefold.CodeFoldCost(self.data,self.depth)
        return(cost.cost(model,idx),cost.grad(model,idx))

    def allClose(self,n,t):
        print n[0],t[0],' =costs'
        print n[1],t[1],' =grads'        
        return numpy.allclose(n[0],t[0]) and numpy.allclose(n[1],t[1])

    def batches(self,frae,cg):
        idx = numpy.array([[0.0]])
        b1= cg(frae,idx)
        idx = numpy.array([[0.0],[1.0]])
        b2 = cg(frae,idx)
        self.assertTrue(self.allClose(b1,b2))
        idx = numpy.array([[0.0],[1.0],[2.0]])
        b3 = cg(frae,idx)
        self.assertFalse(self.allClose(b2,b3))
        
    def testBatchTheano(self):
        self.batches(unfolder.Frae(self.mf,self.depth+1),self.cgTheano)

    def testBatchNumpy(self):
        self.batches(unfolder.FraeNumpy(self.mf),self.cgNumpy)
        
if __name__ == '__main__':
    unittest.main()        
    
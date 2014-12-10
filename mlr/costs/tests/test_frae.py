# -*- coding: utf-8 -*-
"""
Created on Wed Dec 03 18:08:00 2014

@author: Ken
"""
import unittest
from mlr.costs import unfolder
from mlr.costs.unfolder import Frae
from mlr.utils.tree import Tree as BinTree
import numpy

class TestFrae(unittest.TestCase):
    def setUp(self):
        self.size  = 1
        self.mf = unfolder.MatrixFold(self.size)
        self.frae = Frae(self.mf,maxDepth=3)
        l = BinTree(self.r(),None)
        n1 = BinTree(None,[l,l])
        n2 = BinTree(None,[n1,n1])
        n3 = BinTree(None,[n1,l])
        n4 = BinTree(None,[l,n1])
        self.bts = [n2,n3,n4]
        pass
    def r(self):
        return numpy.random.rand(1,self.size)
    
    def testTheanoNumpyCostTree(self):
        bts = self.bts
        frae = self.frae
        pis = [frae.prepInput(i) for i in bts]
        d = bts[0].depth
        cts = [(frae.costTree(i),frae.costTrees[d][0](*i)) for i in pis]
        for ct1,ct2 in cts:
            self.assertTrue(numpy.allclose(ct1,ct2))

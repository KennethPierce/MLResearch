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
    def test_BasicCodeFoldModelNumpy(self):
        isize = 3
        toInputMF = unfolder.MatrixFold(isize)
        toInput = unfolder.FraeNumpy(toInputMF)
        tvi = TreeVector(isize)
        model = codefold.CodeFoldModel(toInput,tvi)
        data = [(Tree(3,[Tree(2,None),Tree(5,None),Tree(0,None)]),'metadata')]
        cost = codefold.CodeFoldCostNumpy(data)
        idx0 = numpy.array([[0.0]])
        cost.cost(model,idx0)
        cost.grad(model,idx0)
        

if __name__ == '__main__':
    unittest.main()        
    
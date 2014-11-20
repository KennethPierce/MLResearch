# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 14:09:56 2014

@author: Ken
"""
import unittest
import numpy
from mlr.datagen.treevector import TreeVector

class TestTreeVector(unittest.TestCase) :
    def test_grow(self):
        numFeat=10
        tv =TreeVector(numFeat)
        vals = [1,numFeat-1,0]
        vs = [tv.getVector(i) for i in vals]                
        for i in vs:
            self.assertEqual(i.shape,(1,numFeat))
            
        self.assertEqual(tv.vectors.shape,(numFeat,numFeat))
        vs1 = [tv.getVector(i) for i in vals]     
        self.assertTrue(numpy.allclose(vs,vs1))
    def setUp(self):
        pass        


if __name__ == '__main__':
    unittest.main()         

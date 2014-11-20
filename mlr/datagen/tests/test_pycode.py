# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 09:57:46 2014

@author: Ken
"""
import unittest
from mlr.datagen.pycode import DictValToInt,PyCode
import ast

class TestDictValToInt(unittest.TestCase):
    def test_Basic(self):
        d = DictValToInt()
        val = d.mapTo('foo')        
        self.assertEqual(d.mapFrom(val),'foo')
    
class TestPyCode(unittest.TestCase):
    def setUp(self):
        pass        

    def parseToTree(self,xs):
        data=[]
        pyc = PyCode(data)
        return [pyc.toTree(ast.parse(i)) for i in xs]        

    def test_VarNamesDifferentIsSameTree(self):       
        cf1='''a,b = c;print [k for k in j if k is not i]'''
        cf2='''x,b = c;print [k for k in j if k is not i]'''
        xs = self.parseToTree([cf1,cf2])
        self.assertEqual(*xs)

    def test_DiffStructureIsDifferentTree(self):
        cf1 ='''x=3;a=b,c'''
        cf2 ='''x=3;a,b=c'''            
        xs = self.parseToTree([cf1,cf2])
        self.assertNotEqual(*xs)
        

if __name__ == '__main__':
    unittest.main()        
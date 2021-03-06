# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:33:02 2014

@author: Ken
"""

from mlr.utils.tree import Tree   
import ast
import os
import random
import collections
from pylearn2.utils import serial    
from mlr.utils.bidict import BiDict

class MetaData(collections.namedtuple('MetaData',['fn','astDump'])):
    pass

class DictValToInt:
    def __init__(self):
        self.ndict = BiDict()


    def mapTo(self,key):
        if slice(key,None) not in self.ndict:
            self.ndict[key:]=len(self.ndict)
        return self.ndict[key:]

    def mapFrom(self,val):
        return self.ndict[:val]     
     
class PyCode:
    """
    Converts python files -> abstract syntax trees -> to utils.tree structures.
    """
    def __init__(self,data,nodeType=ast.FunctionDef,fileSuffix='.py'):
        self.dict = DictValToInt()
        self.data=data
        self.nodeType=nodeType
        self.fileSuffix=fileSuffix

    def toTree(self,node):
        def branches(node):
            trees = []
            for name,val in ast.iter_fields(node):
                if type(val).__module__ == "_ast":
                    ns = [self.toTree(val)]
                elif isinstance(val,list):
                    ns = [self.toTree(i) for i in val]
                else:
                    #ns = [Tree(self.dict.mapTo(val),None)]
                    ns = [Tree(self.dict.mapTo(name),None)]
                trees.extend(ns)
            return trees
        t = branches(node)
        v = type(node).__name__
        vid = self.dict.mapTo(v)
        return Tree(vid,tuple(t) if t else None)

    def addExample(self,node,metaData):
        tree = self.toTree(node)
        self.data.append((tree,metaData))

    def addFromFile(self,fn):

        with open(fn) as f:
            try:
                p=ast.parse(f.read(),fn)
                funs = [i for i in ast.walk(p) if type(i) == self.nodeType]
                for i in funs:
                    m = MetaData(fn,ast.dump(i,False))
                    self.addExample(i,m)
            except Exception:
                p = None

    def addFromDir(self,dn):
        for root,dirs,files in os.walk(dn):
            fns =[os.path.join(root, fn) for fn in files if fn.endswith(self.fileSuffix)]
            print '.',
            for i in fns:
                self.addFromFile(i)
        print




def genData1(srcDir,fn='pycode_',maxNode=400):
    """
    Script to generate data for pylearn2 training
    """
    dl = []
    pyc = PyCode(dl)
    pyc.addFromDir(srcDir)
    dl = [i for i in dl if i[0].nodeCnt < maxNode]
    uniq = {i for i in dl}
    dl = list(uniq)
    random.shuffle(dl)
    l = len(dl)
    serial.save(fn+str(l)+'.pkl',dl)
        
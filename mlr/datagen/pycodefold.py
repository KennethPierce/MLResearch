# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 10:53:03 2014

@author: Ken
"""
import mlr.costs.unfolder as unfolder
import mlr.datagen.pycode as pycode
import random

class PyCodeFold():
    def __init__(self,fc=None,tv=None):
        self.fc = fc
        self.tv = tv
        pass
    def binarySplitTrees(self,data,depth):       
        trees = [i[0] for i in data]
        binsplit = unfolder.TreeToFraeTree(self.fc).binarySplit
        btrees = [binsplit(i) for i in trees]
        dtrees = [i for i in pycode.getTrees(btrees,depth)]
#        uniq = {i for i in dtrees}
#        dtrees = list(uniq)
        random.shuffle(dtrees)
        return dtrees

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 17:44:04 2014

@author: Ken
"""
import collections

class Tree(collections.namedtuple('Tree',['v','ns'])):
    """
    Tree structure
    v contains node's value
    ns contains list of BinTrees
    """
    __slots__=()
    @property
    def isLeaf(self):
        return self.ns == None
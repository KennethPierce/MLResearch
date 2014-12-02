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
    @property
    def nodeCnt(self):
        if self.isLeaf:
            return 1
        return 1 + sum([i.nodeCnt for i in self.ns])
    @property
    def depth(self):
        if self.isLeaf:
            return 0
        return 1 + max([i.depth for i in self.ns])
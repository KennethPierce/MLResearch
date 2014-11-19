# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 07:38:37 2014

@author: Ken
"""
import collections 

class BiDict(collections.Mapping):
    """
    intended to be compatible with pypi-BiDict interface.
    Only supports __get_item__ with slices
    example
        d[a:]=4
        d[:a] returns 4
    """
    def __init__(self):
        self._mapTo = {}
        self._mapFrom = {}

    def __getitem__(self,k):
        assert isinstance(k,slice)
        if k.start is not None:
            return self._mapTo[k.start]
        if k.stop is not None: 
            return self._mapFrom[k.stop]

    def __setitem__(self,k,v):
        assert isinstance(k,slice)
        if k.start is not None:
            key,val = k.start,v
        elif k.stop is not None: 
            val,key = k.stop,v
        else:
            assert False
        self._mapTo[key]=val
        self._mapFrom[val]=key        
    
    def __iter__(self):
       return iter(self._mapTo)
       
    def __len__(self):
       return len(self._mapTo)
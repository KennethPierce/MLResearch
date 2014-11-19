# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 07:42:34 2014

@author: Ken
"""

from mlr.utils.bidict import BiDict

def test_bidict():
    d = BiDict()
    d['foo':]='bar'
    assert d['foo':]=='bar'
    d[:'foobar'] = 123
    assert d[123:] == 'foobar'
    assert len(d) ==2
import unittest
from costs import unfolder
from costs.unfolder import BinTree as BinTree

class ToyFold(unfolder.Fae):
    def unfoldHelper(self,x,y):
        for i in range(x,y):
            if (i*(i+1)/2) > y:
                return i-1
        assert(False)
        return 0
    def enfold(self,nv):
        x=nv[0]
        y=nv[1]
        z = x+y
        return z*(z+1)/2+y

    def unfold(self,a):
        foo = self.unfoldHelper(0,a)
        bar = self.enfold([foo,0])
        y = a-bar
        x = foo-y
        return [x,y]

    def cost(self,nv):
        x=nv[0]
        y=nv[1]
        return (x-y)**2

def test_toyfold():
    mybt0 = BinTree(0,[BinTree(1,None),BinTree(2,None)])
    mybt1 = BinTree(0,[BinTree(3,None),mybt0])
    mybt2 = BinTree(0,[mybt0,mybt1])
    mybt3 = BinTree(0,[mybt2,mybt1])
    mybt4 = BinTree(0,[mybt1,mybt1])
    mybt5 = BinTree(0,[mybt1,mybt2])
    mybt6 = BinTree(0,[mybt2,mybt0])
    mybt7 = BinTree(0,[mybt0,mybt2])
    bts = [mybt0,mybt1,mybt2,mybt3,mybt4,mybt5,mybt6,mybt7]
    for bt in bts:
        frae = unfolder.Frae(ToyFold())
        ct = frae.costTree(bt)
        assert ct == 0
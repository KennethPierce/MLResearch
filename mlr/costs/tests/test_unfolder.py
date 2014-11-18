import unittest
import numpy
from mlr.costs import unfolder
from mlr.costs.unfolder import BinTree as BinTree

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
        
def test_FraeMatrixFold():
    size =2
    lcnt =9
    mf = unfolder.MatrixFold(size)
    frae = unfolder.Frae(mf)    
    def r():
        return 0.5-numpy.random.rand(1,size)
    def numGrad(bt,w):

        s = w.shape
        dw = w.copy()
        epsilon=1e-6
        for i in range(s[0]):
            for j in range(s[1]):
                save = w[i,j]
                w[i,j] = save+epsilon
                c1 = frae.costTreeFlat(bt)
                w[i,j] = save-epsilon
                c2 = frae.costTreeFlat(bt)
                w[i,j] = save
                dw[i,j] = (c1-c2)/(2*epsilon)
        return dw
                
    def binTreeFromList(l):
        cnt = len(l)
        if cnt == 1:
            return BinTree(l[0],None)
        return BinTree(None,[binTreeFromList(l[:(cnt/2)]),binTreeFromList(l[(cnt/2):])])

    bt = binTreeFromList([r() for i in range(lcnt)])
    dwu = numGrad(bt,mf.wu)
    dwe = numGrad(bt,mf.we)
    bte = frae.enfolder(bt)
    btu = frae.unfolder(bt,bte.v)
    bterroru = frae.d_erroru(bte,btu)
    bterrore = frae.d_errore(bte,bterroru.v)
    assert numpy.allclose(dwu,mf.dwu)
    assert numpy.allclose(dwe,mf.dwe)


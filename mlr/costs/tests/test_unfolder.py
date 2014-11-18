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

    def r():
        return 0.5-numpy.random.rand(1,size)
    def numGrad(bt,w,costfun):

        s = w.shape
        dw = w.copy()
        epsilon=1e-6
        for i in range(s[0]):
            for j in range(s[1]):
                save = w[i,j]
                w[i,j] = save+epsilon
                c1 = costfun(bt)
                w[i,j] = save-epsilon
                c2 = costfun(bt)
                w[i,j] = save
                dw[i,j] = (c1-c2)/(2*epsilon)
        return dw
    def numGradW(bt,w,costfun):
        dw = w.copy()
        epsilon=1e-6
        for i,j in enumerate(w):
            save = w[i]
            w[i] = save+epsilon
            c1 = costfun(bt)
            w[i] = save-epsilon
            c2 = costfun(bt)
            w[i] = save
            dw[i] = (c1-c2)/(2*epsilon)
        return dw

                
    def binTreeFromList(l):
        cnt = len(l)
        if cnt == 1:
            return BinTree(l[0],None)
        return BinTree(None,[binTreeFromList(l[:(cnt/2)]),binTreeFromList(l[(cnt/2):])])
        

    def test_Grad(bte,mf,cost,grad):        
        dw = numGradW(bte,mf.W,cost)
        dW,(bterroru,dwu1),(bterrore,dwe1) = grad(bte)
        assert numpy.allclose(dw,dW)


    mf = unfolder.MatrixFold(size)
    frae = unfolder.Frae(mf)    
    bt = binTreeFromList([r() for i in range(lcnt)])
    bte = frae.enfolder(bt)
    test_Grad(bte,mf,frae.costTreeFlat,frae.d_costTreeFlat)
    test_Grad(bte,mf,frae.costTree,frae.d_costTree)

if __name__ == "__main__":
    test_FraeMatrixFold()
    
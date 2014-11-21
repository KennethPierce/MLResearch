import unittest
import numpy
from mlr.costs import unfolder
from mlr.utils.tree import Tree as BinTree

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

class TestToyFold(unittest.TestCase) :
    def test_toyfold(self):
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

class TestMatrixFold(unittest.TestCase):
    
    def setUp(self):
        self.size=2    
        self.lcnt=9

    def r(self):
        return 0.5-numpy.random.rand(1,self.size)

    def numGradW(self,bt,w,costfun):
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

    def binTreeFromList(self,l):
        cnt = len(l)
        if cnt == 1:
            return BinTree(l[0],None)
        return BinTree(None,[self.binTreeFromList(l[:(cnt/2)]),self.binTreeFromList(l[(cnt/2):])])

    def CheckGrad(self,bte,mf,cost,grad):        
        dw = self.numGradW(bte,mf.W,cost)
        dW,(bterroru,dwu1),(bterrore,dwe1) = grad(bte)
        assert numpy.allclose(dw,dW)

        
    def test_FraeMatrixFold(self):   
        mf = unfolder.MatrixFold(self.size)
        frae = unfolder.Frae(mf)    
        bt = self.binTreeFromList([self.r() for i in range(self.lcnt)])
        bte = frae.enfolder(bt)
        self.CheckGrad(bte,mf,frae.costTreeFlat,frae.d_costTreeFlat)
        self.CheckGrad(bte,mf,frae.costTree,frae.d_costTree)

    def test_TreeToFraeTree(self):
        mf = unfolder.MatrixFold(self.size)
        ns = [BinTree(self.r(),None) for i in range(10)]
        bt = BinTree(self.r,ns)
        ttft = unfolder.TreeToFraeTree(mf)
        bt1 = ttft.Greedy(bt)
        self.assertEqual(len(bt1.ns),2)
        
        

if __name__ == '__main__':
    unittest.main()        
    
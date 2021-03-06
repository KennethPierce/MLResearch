import unittest
import numpy

from scipy.optimize import check_grad,approx_fprime
from mlr.costs import unfolder
from mlr.utils.tree import Tree as BinTree
from mlr.datagen.treevector import TreeVector



class ToyFold(unfolder.Fae):
    def __init__(self):        
        self.W =numpy.zeros((1,1))
        
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


class TestMatrixFold(unittest.TestCase):
    
    def setUp(self):
        self.size=2    
        self.lcnt=9
        mybt0 = BinTree(0,[BinTree(1,None),BinTree(2,None)])
        mybt1 = BinTree(0,[BinTree(3,None),mybt0])
        mybt2 = BinTree(0,[mybt0,mybt1])
        mybt3 = BinTree(0,[mybt2,mybt1])
        mybt4 = BinTree(0,[mybt1,mybt1])
        mybt5 = BinTree(0,[mybt1,mybt2])
        mybt6 = BinTree(0,[mybt2,mybt0])
        mybt7 = BinTree(0,[mybt0,mybt2])
        self.bts = [mybt0,mybt1,mybt2,mybt3,mybt4,mybt5,mybt6,mybt7]
        self.bt = self.binTreeFromList([self.r() for i in range(self.lcnt)])
        self.mf = unfolder.MatrixFold(self.size)


    def r(self):
        return 0.5-numpy.random.rand(1,self.size)

    def numGradW(self,bt,w,costfun):
        dw = w.copy()
        epsilon=1e-4
        for i,j in enumerate(w):
            save = w[i]
            w[i] = save+epsilon
            c1 = costfun(bt)
            c1 = numpy.float(c1) #cast up from 32 to 64 bits
            w[i] = save-epsilon
            c2 = costfun(bt)
            c2 = numpy.float(c2)
            w[i] = save
            dw[i] = (c1-c2)/(2*epsilon) #may down cast to 32 bits
        return dw

    def binTreeFromList(self,l):
        cnt = len(l)
        if cnt == 1:
            return BinTree(l[0],None)
        return BinTree(None,[self.binTreeFromList(l[:(cnt/2)]),self.binTreeFromList(l[(cnt/2):])])

    def CheckGrad(self,bte,mf,cost,grad):        
        dw = self.numGradW(bte,mf.W,cost)
        dW,(bterroru,dwu1),(bterrore,dwe1) = grad(bte)
        print dw,' =dw'
        print dW,' =dW'
        assert numpy.allclose(dw,dW,rtol=1e-3,atol=1e-4)


    def test_toyfold(self):
        for bt in self.bts:
            frae = unfolder.FraeNumpy(ToyFold())
            ct = frae.costTree(bt)
            self.assertEqual(ct,0)
        
    def test_FraeMatrixFold(self):   
        mf = self.mf
        frae = unfolder.FraeNumpy(mf)    
        bt = self.bt
        bte = frae.enfolder(bt)
        self.CheckGrad(bte,mf,frae.costTreeFlat,frae.d_costTreeFlat)
        self.CheckGrad(bte,mf,frae.costTree,frae.d_costTree)

        
    def test_GradViaScipy(self):  
        mf = self.mf
        bt = self.bt
        def c(w):
            f = unfolder.FraeNumpy(unfolder.MatrixFold(self.size))
            f.fc.W[:] = w[:]
            c = f.costTree(bt)
            return c
        def g(w):
            f = unfolder.FraeNumpy(unfolder.MatrixFold(self.size))
            f.fc.W[:] = w[:]
            g,_,_ = f.d_costTree(bt)
            return g
            
        #err = check_grad(c,g,mf.W)
        approx = approx_fprime(mf.W,c,1e-4)
        grad = g(mf.W)
        print approx," =approx"
        print grad," =grad"
        self.assertTrue(numpy.allclose(approx,grad,rtol=1e-3,atol=1e-4))

    def inOrder(self,bt,acc):
        if bt.v <> None:
            acc.append(bt.v)
        if bt.isLeaf:
            return acc
        for i in bt.ns:
            self.inOrder(i,acc)
        return acc           
        
    def getLeafs(self,bt,acc):
        if bt.isLeaf:
            acc.append(bt.v)
        else:
            for i in bt.ns:
                self.getLeafs(i,acc)
        return acc
            
    def test_TreeToFraeTree_MiddleSplit(self):        
        ttft = unfolder.TreeToFraeTree(None)        
        t = BinTree(44,[BinTree(4,self.bts)]+self.bts)
        bt = ttft.middleSplit(t)
        a = self.inOrder(t,[])
        b = self.getLeafs(bt,[])
        #assert same number of original node.v values == number leafs 
        self.assertEqual(len(a),len(b))
        
    def test_TreeToFraeTree_GreedySplit(self):      
        size=self.size
        ttft = unfolder.TreeToFraeTree(unfolder.MatrixFold(size))        
        t = BinTree(4,[BinTree(5,self.bts)]+self.bts)
        tv = TreeVector(size)
        t = tv.convertTree(t)        
        bt = ttft.greedySplit(t)
        a = self.inOrder(t,[])
        b = self.getLeafs(bt,[])
        #assert same number of original node.v values == number leafs 
        self.assertEqual(len(a),len(b))



if __name__ == '__main__':
    unittest.main()        
    
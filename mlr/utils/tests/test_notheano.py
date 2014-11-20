import unittest
import mlr.utils.notheano as notheano
import numpy
import theano
import theano.tensor as T
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
class AECost(notheano.Cost):
    """
    Notheano Cost
    """
    def __init__(self,ds):
        self.ds = ds        
        pass
    
    def SliceData(self,data):
        indexes = notheano.SliceData(data)
        d = [self.ds[i] for i in indexes]
        return numpy.array(d)
    
    
    def GetFuns(self,model):
        """
        on first call create the fcost and fgrad theano functions
        otherwise just return them
        """
        if hasattr(self,'fcost') :
            return self.fcost,self.fgrad
        X = T.dmatrix('X')
        W = T.dmatrix('W')
        a1 = T.tanh(X.dot(W))
        a2 = T.tanh(a1.dot(W.T))
        cost = T.mean(a2.dot(a2.T),axis=0).sum()
        self.fcost = theano.function([X,W],cost)
        self.fgrad = theano.function([X,W],T.grad(cost,W))
        return self.GetFuns(model)
        
    def cost(self,model,data):    
        cf,gf = self.GetFuns(model)
        X = self.SliceData(data)
        W = model.W.get_value()
        ans = cf(X,W)
        return ans

        
    def grad(self,model,data):
        cf,gf = self.GetFuns(model)
        X = self.SliceData(data)
        W = model.W.get_value()
        return gf(X,W)

        
class AEModel(Model):
    def __init__(self,nvis,nhid):
        super(AEModel,self).__init__()
        self.W= sharedX(numpy.random.uniform(low=-.1,high=.1,size=(nvis,nhid)),'W')
        self._params=[self.W]
        self.input_space=VectorSpace(dim=1)       
        

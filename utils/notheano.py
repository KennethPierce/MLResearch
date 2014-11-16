# create a simple pylearn2 model of an AE
import theano
import theano.tensor as T
from pylearn2.costs.cost import Cost,DefaultDataSpecsMixin
from pylearn2.utils import CallbackOp
from theano.tensor.shared_randomstreams import RandomStreams
from theano.compat.python2x import OrderedDict
from theano.compat.six.moves import zip as izip
import numpy
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX

class NoTheanoCost(DefaultDataSpecsMixin,Cost):
    supervised = False
        
    def expr(self, model, data, **kwargs):
        space,source = self.get_data_specs(model)
        space.validate(data)
        srng = RandomStreams(seed=234)
        return OverwriteOp(self.cost,model)(srng.uniform(low=0.0,high=1000.0),data)
    def get_gradients(self, model, data, ** kwargs):
        srng = RandomStreams(seed=232)
        params = list(model.get_params())
        grads = [OverwriteOp(self.grad,model)(srng.uniform(size=i.shape),data) for i in params]
        gradients = OrderedDict(izip(params, grads))
        updates = OrderedDict()
        return gradients, updates        
    def cost(self,model,data):
        raise NotImplementedError(str(type(self)) + " doesn't implement CallbackCost")
    def grad(self,model,data):
        raise NotImplementedError(str(type(self)) + " doesn't implement CallbackGrad")


class OverwriteOp(theano.gof.Op):
    """
    A Theano Op that implements the identity transform but also does an
    arbitrary (user-specified) side effect.

    Parameters
    ----------
    callback : WRITEME
    """
    view_map = {0: [0]}

    def __init__(self,callback,model):
        self.callback = callback
        self.model = model

    def make_node(self, xin,xin_data):
        """
        .. todo::

            WRITEME
        """
        xout = xin.type.make_variable()
        return theano.gof.Apply(op=self, inputs=[xin,xin_data], outputs=[xout])

    def perform(self, node, inputs, output_storage):
        """
        .. todo::

            WRITEME
        """
        xin,xin_data = inputs
        xout, = output_storage
        xout[0] = self.callback(self.model,xin_data)
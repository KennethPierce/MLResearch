import theano
from theano.tensor.shared_randomstreams import RandomStreams
from theano.compat.python2x import OrderedDict
from theano.compat.six.moves import zip as izip

from pylearn2.costs.cost import Cost as PyLearn2Cost
from pylearn2.costs.cost import DefaultDataSpecsMixin 


def SliceData(data):
    """
    Helper function.  Converts a ndarray of float indexes to a list of ints
        ie. [[1.0],[2.0]] -> [1,2]
    """
    return [int(i) for [i] in data]

class Cost(DefaultDataSpecsMixin,PyLearn2Cost):
    """
    Subclass this class instead of Cost to remove theano dependencies
        overwrite cost instead of expr.
        without theano you have to supply the gradient with grad()
        Your init function can load your arbitrary data.  The trainer should give you just indexes
        The grad callback is made for each model.param.  Suggest having just one param that you
            flatten and unflatten as needed.
    """
    supervised = False
    
    def expr(self, model, data, **kwargs):
        """
        Overwrites the Cost.expr so we can inject our theano.Op.  
        """
        space,source = self.get_data_specs(model)
        space.validate(data)
        #really no point to using these random values.  Could be zeros
        srng = RandomStreams(seed=234)
        return OverwriteOp(self.cost,model)(srng.uniform(low=0.0,high=1000.0),data)
    def get_gradients(self, model, data, ** kwargs):
        """
        Overwrites the Cost.get_gradients so we can inject our theano.Op
        This will do a separate call back for each model.param
            Consider rewriting your model to have one param 
        """
        srng = RandomStreams(seed=232)
        params = list(model.get_params())
        grads = [OverwriteOp(self.grad,model)(srng.uniform(size=i.shape),data) for i in params]
        gradients = OrderedDict(izip(params, grads))
        updates = OrderedDict()
        return gradients, updates        
    def cost(self,model,data):
        """
            data: this is the data from the trainer.  Likely just a list of indexes to the real data.
            return the cost.
        """
        raise NotImplementedError(str(type(self)) + " doesn't implement cost")
    def grad(self,model,data):
        """
            data: this is the data from the trainer.  Likely just a list of indexes to the real data.
            return the gradient.
        """
        raise NotImplementedError(str(type(self)) + " doesn't implement grad")
    

        

class OverwriteOp(theano.gof.Op):
    """
    A Theano Op that takes two inputs.  It has same output type as first input type.  The callback return value will update the output.
    Note: Code modified from CallbackOp in pylearn2
    """
    view_map = {0: [0]}

    def __init__(self,callback,usrData):
        self.callback = callback
        self.usrData = usrData

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
        xout[0] = self.callback(self.usrData,xin_data)
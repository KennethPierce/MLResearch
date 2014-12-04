"""folding recursive auto-encoder"""
import numpy
from mlr.utils.tree import Tree as BinTree



class Fae:
    """Folding autoencoder"""
    def enfold(self,vs):
        """
        vs: list of x objects 
        return: 1 x object
        """
        raise 

    def unfold(self,act):
        """
        act: x object
        returns: list of x reconstructed objects
        """
        raise

    def cost(self,vs):
        """ 
        vs: list of objects.  vs[0] is target.  vs[1:] are reconstructions
        return cost of reconstruction
        """
        raise
        
    def d_enfold(self,acts,err):
        """
        acts: 
        err: previous layer errors
        returns: error term for this layer
        """
        raise
        
    def d_unfold(self,act,errs):
        """
        act:
        errs: previous layer errors
        returns: error term for this layer
        """
        raise

    def d_cost(self,vs):
        """
        return: error term of cost
        """
        raise
        
    def d_input(self,v,err):
        """
        return: error term for inputs
        """
        raise
    def deltaW(self,de,du):
        """
        de: enfold delta terms
        du: unfold delta terms
        return: W-shaped delta terms
        """
        raise        
       
class MatrixFold(Fae) :
    def __init__(self,s):
        self.size =s;
        r = (6.0/(3*s))**.5
        w = 2*s*s
        self.W = r-(2*r*numpy.random.rand(2*w))
        self.we = self.W[:w]
        self.we.shape = (2*s,s) #will raise instead of copy
        self.wu = self.W[w:]
        self.wu.shape = (s,2*s) #will raise instead of copy
#        self.we = r-(2*r*numpy.random.rand(2*s,s))
#        self.wu = r-(2*r*numpy.random.rand(s,2*s))

    def enfold(self,vs):
        j = numpy.concatenate(vs,axis=1)
        return numpy.tanh(j.dot(self.we))

    def unfold(self,act):
        s = self.size
        x = numpy.tanh(act.dot(self.wu))
        return (x[:,:s],x[:,s:])

    def d_enfold(self,acts,err):
        s=self.size
        act = numpy.concatenate(acts,axis=1)
        delta = act.T.dot(err)
        e = err.dot(self.we.T)*(1-act*act)       
        return (e[:,:s],e[:,s:]),delta

    def d_unfold(self,act,errs):
        j = numpy.concatenate(errs,axis=1)
        delta = act.T.dot(j)
        ret = j.dot(self.wu.T)*(1-act*act)
        return ret,delta

    def deltaW(self,de,du):
        return numpy.concatenate([de.flatten(),du.flatten()])

    def cost(self,vs):
        v=vs[0]-vs[1]
        return v.dot(v.T)/2
        
    def d_cost(self,vs):
        return -(vs[0]-vs[1])*(1-vs[1]*vs[1])

    def d_input(self,v,err):
        """error to input"""
        pass
       
class TreeToFraeTree:
    """
    Methods to convert arbitrary trees with data in both leafs and non-leafs
    """
    def __init__(self,fc):
        """
        fc: fae class used to score fold options        
        """
        self.fc = fc      
        self.frae = Frae(self.fc)
    
    def binarySplit(self,node):
        """
        fast but stupid split.  baseline performance test cost of greedy split
        """
        if node.isLeaf:
            return BinTree(node.v,None)
        def collapse(ns):
            l = len(ns)            
            assert l <> 0
            if l==1:
                return ns
            if l==2:
                return [BinTree(None,ns)]
            hl = int(l/2)
            return collapse(collapse(ns[:hl])+collapse(ns[hl:]))
        
        ng = [self.binarySplit(i) for i in node.ns]        
        n = collapse(ng)
        return BinTree(None,[BinTree(node.v,None)]+n)                  
            

        
        
        
            
            
    def greedy(self,node):
        """
        Preserve existing structure but fix nodes with too many and too few leaves
        When fixing, select best scoring pair to merge
        """
        if node.isLeaf:
            return node
        
        def collapse(ng):
            if len(ng)==1:
                return ng
            else:
                z = zip(ng,ng[1:])
                trees = [BinTree(self.fc.enfold([l.v,r.v]),[l,r]) for l,r in z]
                cts = [self.frae.costTree(i) for i in trees]
                idx,_ = min(enumerate(cts),key=lambda x:x[1])
                ngc = ng[:idx] + [trees[idx]] + ng[idx+2:]
                assert len(ngc) + 1 == len(ng)
                return collapse(ngc)
        ng = [self.greedy(i) for i in node.ns]        
        n = collapse(ng)[0]
        e = self.fc.enfold([node.v,n.v])
        return BinTree(e,[BinTree(node.v,None),n])
  
class Frae:
    """
    Expects all BTree nodes to have form of either 0 or n children.  
    fc: instance of Fae
    """
    def __init__(self,fc):
        """
        fc: fold class instance.  Should inherit from Fae class 
        """
        self.fc = fc
        
    def enfolder(self,bt):
        """
        bt: input btree to enfold
        returns enfolded btree
        """
        if bt.isLeaf :
            assert(bt.v <> None)
            return bt
        else:
            ns = [self.enfolder(i) for i in bt.ns]
            vs = [i.v for i in ns]  
            v = self.fc.enfold(vs)
            return BinTree(v,ns)
            
    def unfolder(self,bt,act):
        """
        bt: btree used to "shape" the unfold
        act: value to unfold to a btree
        returns: reconstructed btree
        """
        if bt.isLeaf:
            return BinTree(act,None)
        else:
            xs = self.fc.unfold(act)
            z = zip(bt.ns,xs)
            bts = [self.unfolder(i,j) for i,j in z]
            return BinTree(act,bts)
            
    def coster(self,bts):
        """
        bts: list of btrees. bts[0] is target. bts[1:] are the reconstructions
        return: cost of reconstruction error.
        """
        if bts[0].isLeaf:
            assert all([i.isLeaf for i in bts])
            return self.fc.cost([i.v for i in bts])
        else:
            ret = 0#self.fc.cost([i.v for i in bts])
            z= zip (*[i.ns for i in bts]) 
            for i in z:
                ret += self.coster(i)
            return ret

    def costTreeFlat(self,bt):
        """
        bt:  input btree with inputs at leaves. Non-leaf node values are ignored.
        return: returns the cost value of the tree
        """
        bte = self.enfolder(bt)
        btu = self.unfolder(bt,bte.v)
        return self.coster([bte,btu])
            
    def costTree(self,bt):
        """
        bt: input btree with inputs at leaves. Non-leaf node values are ignored.
        return: returns the cost value of the tree and subtrees
        """
        def costTree_rec(bt):
            if bt.isLeaf:
                return 0
            ret = self.coster([bt,self.unfolder(bt,bt.v)])
            for i in bt.ns:
                ret += costTree_rec(i)
            return ret
        bte = self.enfolder(bt)
        return costTree_rec(bte)

    def d_costTree(self,bt):
        """
        return: W like array of the delta terms
        """        
        def d_costTree_rec(bte):
            if bte.isLeaf:
                return None
            dW,_,_ =self.d_costTreeFlat(bte)
            deltaList = [d_costTree_rec(i) for i in bte.ns]
            deltaList = [i for i in deltaList if i is not None]
            return sum(deltaList,dW)
            
        bte = self.enfolder(bt)
        return d_costTree_rec(bte),(None,None),(None,None)


    def d_costTreeFlat(self,bte):
        btu = self.unfolder(bte,bte.v)
        bterroru,dwu = self.d_erroru(bte,btu)
        bterrore,dwe = self.d_errore(bte,bterroru.v)
        dW = self.fc.deltaW(dwe,dwu)
        return dW,(bterroru,dwu),(bterrore,dwe)

    def d_errore(self,bte,err):
        """
        bte: enfolded tree
        err: error term from layer above 
        returns: backprop errors for the folded reconstruction
        """
        if bte.isLeaf:
            self.fc.d_input(bte.v,err) 
            return BinTree(err,None),None           
        else:           
            errs,delta = self.fc.d_enfold([i.v for i in bte.ns],err)
            z = zip(errs,bte.ns)
            btsDelta=[self.d_errore(bt,e) for (e,bt) in z]
            bts,deltaList = zip(*btsDelta)
            deltaList = [i for i in deltaList if i is not None]
            return BinTree(err,bts),sum(deltaList,delta)
            
    def d_erroru(self,bte,btu):
        """
        bte: enfolded tree
        btu: unfolded tree
        returns: backprop errors for the unfold reconstruction
        """
        if bte.isLeaf:
            assert btu.isLeaf
            # bte.v is an input while btu.v is the unfolded value
            return BinTree(self.fc.d_cost([bte.v,btu.v]),None),None
        else: 
            #errc = self.fc.d_cost([bte.v,btu.v])
            z = zip(bte.ns,btu.ns)
            errTreeDelta = [self.d_erroru(e,u) for e,u in z]
            errTree,deltaList = zip(*errTreeDelta)
            err = [i.v for i in errTree]
            erru,delta = self.fc.d_unfold(btu.v,err)
            deltaList = [i for i in deltaList if i is not None]
            return BinTree(erru,errTree),sum(deltaList,delta)

import scipy.sparse as sparse
import theano
import theano.tensor as T

class Frae_():
    def __init__(self,fc,maxDepth=6):
        self.fc = fc
        self.We = fc.we
        self.Wu = fc.wu
        self.maxDepth = maxDepth
        self.tW = theano.shared(fc.W,"W",borrow=True)
        s = fc.size
        w = 2*s*s
        self.tWe = self.tW[:w]
        self.tWe = T.reshape(self.tWe,(2*s,s))
        self.tWu = self.tW[w:]
        self.tWu = T.reshape(self.tWu,(s,2*s))
        self.costTrees = [self.tCostTree(i) for i in range(1,maxDepth)]        
        self.costTrees[0:0] = [None]

    
    def tCostTree(self,cnt):
        tInputs = [T.dmatrix() for i in range(cnt)]
        c = self.costTreeTheano(tInputs)
        f = theano.function(tInputs,c)
        g = theano.function(tInputs,theano.grad(c,self.tW))
        return f,g

    
    def tCoster(self,cnt):
        tInputs = [T.dmatrix() for i in range(cnt)]
        e = self.enfolderTheano(tInputs)
        u = self.unfolderTheano(e)
        c = self.costerTheano(e,u)
        f = theano.function(tInputs,c)
        g = theano.function(tInputs,theano.grad(c,self.tW))
        return f,g
    
    def prepInput(self,bt):
        def getLeafs(bt,state):
            if bt.isLeaf:
                yield (state,bt.v)
            else:
                for idx,t in enumerate(bt.ns):
                    for i in getLeafs(t,state + [idx]):
                        yield i
        def getIdx(l):
            return reduce(lambda prev,i : prev*2+i,l,0)
        def nanMatrix(rows):
            ret = numpy.zeros((rows,self.fc.size))
            ret[:] = numpy.nan
            return ret
            
        ret = [nanMatrix(2**i) for i in range(bt.depth+1)]
        for l,v in getLeafs(bt,[]):
            idx = getIdx(l)
            ret[len(l)][idx] = v
        ret = ret[1:]
        ret.reverse()
        
        return ret
        
    
    def enfolder(self,listInputs):
        prev = numpy.zeros_like(listInputs[0])
        ret = [listInputs[0]]
        for i in listInputs:
            prev = numpy.where(numpy.isnan(i),prev,i)
            j = numpy.concatenate((prev[::2],prev[1::2]),axis=1)
            act = numpy.tanh(j.dot(self.We))
            prev = act
            ret.append(act)
        return ret

    def enfolderTheano(self,listInputs):
        prev = T.zeros_like(listInputs[0])
        ret = [listInputs[0]]
        for i in listInputs:
            prev = T.where(T.isnan(i),prev,i)
            j = T.concatenate((prev[::2],prev[1::2]),axis=1)
            act = T.tanh(j.dot(self.tWe))
            prev = act
            ret.append(act)
        return ret

        
    def unfolder(self,listActs):
        ret = []
        prev = listActs[-1] 
        for _ in listActs[0:-1]:
            prev = numpy.tanh(prev.dot(self.Wu))
            hl = self.Wu.shape[1]/2
            val = numpy.zeros((2*prev.shape[0],hl))
            val[::2] = prev[:,:hl]
            val[1::2] = prev[:,hl:]
            prev=val
            ret.append(prev)
        ret.reverse()
        return ret     

    def unfolderTheano(self,listActs):
        ret = []
        prev = listActs[-1] 
        for _ in listActs[0:-1]:
            prev = T.tanh(prev.dot(self.tWu))
            hl = self.Wu.shape[1]/2
            #val = T.zeros((2*prev.shape[0],hl))
            val = T.zeros((2*prev.shape[0],hl))
            
            val = T.set_subtensor(val[::2,:],prev[:,:hl])
            val = T.set_subtensor(val[1::2,:],prev[:,hl:])
            prev=val
            ret.append(prev)
        ret.reverse()
        return ret     

    def coster(self,enf,unf):        
        def cost(e,u):
            diff = e-u
            diff = numpy.where(numpy.isnan(e),0,diff)
            sq = diff*diff
            return numpy.sum(sq)
        
        z=zip(enf,unf)
        costs = [cost(*i) for i in z]        
        return sum(costs)


    def costerTheano(self,enf,unf):        
        def cost(e,u):
            diff = e-u
            diff = T.where(T.isnan(e),0,diff)
            sq = diff*diff
            return T.sum(sq)
        
        z=zip(enf,unf)
        costs = [cost(*i) for i in z]        
        return sum(costs)


    def costTree(self,listInputs):
        l=[]
        enf = self.enfolder(listInputs)                
        for i in range(len(enf),0,-1):            
            e = enf[:i]
            u = self.unfolder(e)
            c = self.coster(e,u)
            l.append(c)
        return sum(l)
        

    def costTreeTheano(self,listInputs):
        l=[]
        enf = self.enfolderTheano(listInputs)                
        for i in range(len(enf),0,-1):            
            e = enf[:i]
            u = self.unfolderTheano(e)
            c = self.costerTheano(e,u)
            l.append(c)
        return T.sum(l)
        

        
#### Sparse not fully implemented

    def unfolderSparse(self,listActs):
        ret = []
        prev = listActs[-1] 
        Wu = sparse.csr_matrix(self.Wu)
        la = listActs[:]
        la.reverse()
        for act in la[0:-1]:
            
            prev = (prev.dot(Wu)).tanh()
            hl = Wu.shape[1]/2
            val = sparse.lil_matrix((2*prev.shape[0],hl))
            val[::2] = prev[:,:hl]
            val[1::2] = prev[:,hl:]
            prev=val.tocsr()
            ret.append(prev)
        ret.reverse()
        return ret             


    def enfolderSparse(self,listInputs):
        prev = sparse.csr_matrix((listInputs[0].shape))
        ret = [listInputs[0]]
        We = sparse.csr_matrix(self.We)
        for i in listInputs:
            prev = prev+i
            j = sparse.hstack([prev[::2],prev[1::2]])
            act = (j.dot(We)).tanh()
            prev = act
            ret.append(act)
        return ret
        
    def prepInputSparse(self,bt):
        def leafsAtDepth(bt,dep):
            res = sparse.lil_matrix((2**dep,self.fc.size))            
            def leafAtDepth(bt,s): 
                if bt.isLeaf:
                    if s.start +1 == s.stop:
                        res[s]=bt.v
                else:
                    if s.start +1 <> s.stop:
                        assert len(bt.ns) == 2
                        assert bt.v==None
                        leafAtDepth(bt.ns[0],slice(s.start,s.stop/2))
                        leafAtDepth(bt.ns[1],slice(s.stop/2,s.stop))
            leafAtDepth(bt,slice(0,res.shape[0]))
            return res.tocsr()
        d = bt.depth
        ret = [leafsAtDepth(bt,i) for i in range(d,0,-1)]
        return ret
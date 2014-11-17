"""folding recursive auto-encoder"""
import collections
import numpy

class BinTree(collections.namedtuple('BinTree',['v','ns'])):
    """
    Tree structure
    v contains node's value
    ns contains list of BinTrees
    """
    __slots__=()
    @property
    def isLeaf(self):
        return self.ns == None

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

    def d_errore(self,bte,err):
        """
        bte: enfolded tree
        err: error term from layer above 
        returns: backprop errors for the folded reconstruction
        """
        if bte.isLeaf:
            self.fc.d_input(bte.v,err) 
            return BinTree(err,None)           
        else:           
            errs = self.fc.d_enfold([i.v for i in bte.ns],err)
            z = zip(errs,bte.ns)
            bts=[self.d_errore(bt,e) for (e,bt) in z]
            return BinTree(err,bts)
            
    def d_erroru(self,bte,btu):
        """
        bte: enfolded tree
        btu: unfolded tree
        returns: backprop errors for the unfold reconstruction
        """
        if bte.isLeaf:
            assert btu.isLeaf
            # bte.v is an input while btu.v is the unfolded value
            return BinTree(self.fc.d_cost([bte.v,btu.v]),None)
        else: 
            #errc = self.fc.d_cost([bte.v,btu.v])
            z = zip(bte.ns,btu.ns)
            errTree = [self.d_erroru(e,u) for e,u in z]
            err = [i.v for i in errTree]
            erru = self.fc.d_unfold(btu.v,err)
            return BinTree(erru,errTree)

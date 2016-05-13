import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D




def closestPlot(centers,points):
    cl = centers.shape[0]
    pl = points.shape[0]
    a=np.repeat(centers,pl,axis=0)
    b=np.concatenate([points for i in range(cl)],axis=0)
    d = np.linalg.norm(a-b,axis=1).reshape(cl,pl)
    m = np.argmin(d,axis=0)
    return m


def closestPolar(c,p):
    pc = toCartM(c)
    pp = toCartM(p)
    return closestPlot(pc,pp)
    
def closestSpiral(centers,points):
    c=toPolarM(centers).dot(np.ones((2,2)))%(2*np.pi)
    p=toPolarM(points).dot(np.ones((2,2)))%(2*np.pi)
    return closestPlot(c,p)

def randPlot(alg,numc,nump,scale=10):
    c = np.random.rand(numc,2)*scale-(.5*scale)
    p = np.random.rand(nump,2)*scale-(.5*scale)
    cats=alg(c,p)
    scatterPlot(p,cats)

def toUnitCircle(m):
    return m / np.linalg.norm(m,axis=1,keepdims=True)


def toCart(r,theta):
    return (r*np.sin(theta),r*np.cos(theta))
    
def toPolar(x,y):
    return ((x*x+y*y)**0.5,np.arctan2(x,y))

def toCartM(m):
   r = m[:,0:1]
   t = m[:,1:2]
   x = r*np.sin(t)
   y = r*np.cos(t)
   return np.concatenate((x,y),axis=1)    
    
def toPolarM(m):
    dis = np.linalg.norm(m,axis=1,keepdims=True)
    an = np.arctan2(m[:,0:1],m[:,1:2])
    return np.concatenate((dis,an),axis=1)
    
def spiralPlot():
    px = np.linspace(0,1*np.pi*2)
    fig= plt.figure()
    ax = fig.add_subplot(111)
    ps = [toCart(i,i+1) for i in px]
    ax.plot([i for i,j in ps],[j for i,j in ps],marker='+')
    ps = [toCart(i,i+2) for i in px]
    ax.plot([i for i,j in ps],[j for i,j in ps],marker='o')
    ps = [toCart(i,i+3) for i in px]
    ax.plot([i for i,j in ps],[j for i,j in ps],marker='x')
    ps = [toCart(i,i+4) for i in px]
    ax.plot([i for i,j in ps],[j for i,j in ps],marker='*')    
    plt.show()  



def scatterPlot(p,c):
    fig= plt.figure()
    ax = fig.add_subplot(111)
    
    ax.scatter([i for i,j in p],[j for i,j in p],c=c)    
    plt.show()
    

def simplePlot():
    t = np.arange(0.0, 2.0, 0.01)
    s = np.sin(2*np.pi*t)
    plt.plot(t, s)
    plt.xlabel('time (s)')
    plt.ylabel('voltage (mV)')
    plt.title('About as simple as it gets, folks')
    plt.grid(True)
    plt.savefig("test.png")
    plt.show()

def simple3dLine():
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    ax.plot(x, y, z, label='parametric curve')
    ax.legend()
    plt.show()
    
def matrixPlot(m):
    fig = plt.figure()
    if m.shape[1]==3 :
        ax = fig.add_subplot(111, projection='3d')    
        for i in range(0,m.shape[0]):
          ax.text(m[i,0],m[i,1],m[i,2],'%s'%i,None)
    if m.shape[1] ==2 :
        ax = fig.add_subplot(111)    
        for i in range(0,m.shape[0]):
          ax.text(m[i,0],m[i,1],'%s'%i,None)

            
    plt.show()
    
class SparseFilter :
    def __init__(self,w):
        self.w = tf.Variable(tf.convert_to_tensor(w, dtype=tf.float32))
        pass
    def sess(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            

    def applyW(self,s0):
        return tf.matmul(s0,self.w)
    def abso(self,s1):
        return tf.abs(s1)
    def normL2Row(self,s2):
        return tf.nn.l2_normalize(s2,0)
    def normL2Col(self,s3):
        return tf.nn.l2_normalize(s3,1)
    def score(self,s4):
        return tf.reduce_sum(s4)

def runSF():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    
    ex1 =np.array( [[0.0,0.0],[-1.0,-3.0],[-2.0,-2.0],[-2.0,-5.0],[-1.2,2.0],[-2.2,9.0],[2.0,-1.0],[5.0,5.5]])
        
    
    
    with tf.Session() as sess:
        #s = SparseFilter(np.repeat(np.identity(50),16,0)[0:784,:])
        s = SparseFilter(tf.random_normal([784,50]))
        #s =SparseFilter(np.identity(2))  
        #s0 = tf.constant(ex1,dtype=tf.float32)
        s1 = s.applyW(x)
        s2 = s.abso(s1)
        s3 = s.normL2Row(s2)
        s4 = s.normL2Col(s3)
        s5 = s.score(s4)
        optimizer = tf.train.AdamOptimizer(0.1)
        train = optimizer.minimize(s5)
        sess.run(tf.initialize_all_variables())
        
        np.set_printoptions(precision=2,suppress=True)
        for i in range(10000):
            batch = mnist.train.next_batch(5000)
            train.run(feed_dict={x: batch[0]})
            #print (sess.run(s5,feed_dict={x: batch[0]})) 
        fullset = mnist.train.next_batch(55000)
        result = (sess.run(s4,feed_dict={x: fullset[0]}))
        W = sess.run(s.w)

    
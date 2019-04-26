import numpy as np
import GPy
import mnist

def getMNISTexample(scalingfactor=7,Ntraining=500,splittype='01'):
    """
    Get a scaled MNIST example
    splittype must be:
     'fiveormore'
     or a pair of digits, e.g. '35'
    """
    X = mnist.train_images()
    Y = mnist.train_labels()
    
    if splittype=="fiveormore":
        Y = Y<5
        print("Comparing Y<5 to Y>5")
    else:
        digitA = int(splittype[0])
        digitB = int(splittype[1])
        keep = (Y==digitA) | (Y==digitB)
        X = X[keep,:]
        Y = Y[keep]
        Y=(Y==digitB)
        print("Comparing %d vs %d" % (digitA, digitB))

    def scale(X,res):
        newX = []
        for x in X:
            if res==1:
                newx = x
            else:
                newx = np.zeros([1+int(28/res),1+int(28/res)])
                xreshaped = x.reshape(28,28)
                for newi,i in enumerate(np.arange(0,28,res)):
                    for newj,j in enumerate(np.arange(0,28,res)):
                        newx[newi,newj] = np.mean(xreshaped[i:(i+res),j:(j+res)])
            newX.append(newx.reshape(newx.size))
        return np.array(newX)

    X = X[0:Ntraining,:]
    Y = Y[0:Ntraining]
    X = scale(X,scalingfactor)
    newres = int(X.shape[1]**.5)
#    for i in range(len(X)):
#        for j in range(i%4):
#            x = X[i,:].reshape(newres,newres)
#            X[i,:] = np.array(list(zip(*x[::-1]))).reshape(newres**2)


    return X[0:Ntraining,:],Y[0:Ntraining][:,None]

def getsynthexample(N=10000,D=8,noisescale=0.1):
    """
    Produce a non-linearly separable dataset along a path from the origin to [1,1,1...1,1]
    with N training points and D dimensions.
    """
    np.random.seed(0)
    fullX = np.random.rand(N,D)
    fullX[:,1:]=fullX[:,0:1].repeat(D-1,1)+noisescale*np.random.randn(fullX.shape[0],fullX.shape[1]-1)
    distsqrs = np.sum((fullX[:,0:7]-0.5)**2,1)
    distsqrs+=0.0*np.random.randn(distsqrs.shape[0])
    sorteddists = np.sort(distsqrs)
    lowthresh = sorteddists[int(len(distsqrs)*0.25)]
    highthresh = sorteddists[int(len(distsqrs)*0.75)]
    keep = (distsqrs<lowthresh) | (distsqrs>highthresh)
    fullX = fullX[keep,:]
    distsqrs = distsqrs[keep]
    Y = (distsqrs>np.median(distsqrs))
    Y=Y[:,None]
    p = np.random.permutation(len(Y))
    X=fullX[p,:]
    Y=Y[p,:]
    return X,Y
   
   
#TODO AUTOMATICALLY DOWNLOAD THESE DATA!
def getspamexample():
    data = pd.read_csv('spambase.data',header=None).as_matrix()
    np.random.shuffle(data)
    Y = data[:,-1:]
    Y[Y==0]=-1
    X = data[:,0:-1]
    return X,Y
    
def getbankexample():
    data = pd.read_csv('bank.txt').as_matrix()
    np.random.shuffle(data)
    Y = data[:,-1:]
    Y[Y==0]=-1
    X = data[:,0:4]
    return X,Y
    
def getcreditexample():
    data = pd.read_csv('credit.txt',sep=' ').as_matrix()
    np.random.shuffle(data)
    Y = data[:,-1:]
    Y[Y==0]=-1
    X = data[:,0:-1]    
    return X,Y

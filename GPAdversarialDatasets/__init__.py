import numpy as np
import GPy
import mnist

def getMNISTexample(scalingfactor=7,Ntraining=500,splitfiveormore=True):
    X = mnist.train_images()
    Y = mnist.train_labels()
    
    if splitfiveormore:
        Y = Y<5
    else:
        #X = X[Y<2,:]
        #Y = Y[Y<2]
        keep = (Y==3) | (Y==5)
        X = X[keep,:]
        Y = Y[keep]
        Y=(Y==3)

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
    newres = int(X[0,:].size**.5)
    return X[0:Ntraining,:],Y[0:Ntraining][:,None]

    

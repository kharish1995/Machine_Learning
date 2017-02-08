import numpy as np
import matplotlib.pyplot as plt

def sig(z):  
    g= np.zeros(z.shape)
    g=1./(1+np.exp(-z))
    return (g)

def pre(th,X):
    m=X.shape[0]
    p=np.zeros((m,1))
    p=(sig(X.dot(th))>0.5).astype(int)
    return (p)

def cf(th, X, y, Lambda):
    m=len(y)
    h=sig(X.dot(th))
    J=-(1/float(m))*(y*np.log(h)+(1-y)*np.log((1-h))).sum()+(Lambda/(2*float(m)))*(th[1:,:]*th[1:,:]).sum()
    l=(1/float(m))*(h-y)
    grad=(((l.dot(np.ones((1,X.shape[1]))))*X).sum(axis=0))+((Lambda/float(m))*np.insert(th[1:,:], [0], np.zeros((1,th.shape[1])), axis=0)).transpose()
    grad= grad.transpose()
    return (J, grad)

def gr(th,grad,alpha):
    th=th-alpha*grad
    return(th)

def main():
    F=np.zeros((1,30))
    d=np.genfromtxt('data.txt',delimiter=',')
    d1=d[:200,:]
    d2=d[251:371,:]
    data=np.vstack((d1,d2))
    nd1=d[201:251,:]
    nd2=d[371:400,:]
    tdata=np.vstack((nd1,nd2))
    (m,n)=data.shape
    X = data[:, 0:n-1]
    X=(X-np.average(X,axis=0))/np.std(X,axis=0)
    y = (data[:, n-1:n])
    Lambda=0
    alpha=0.001
    (m,n)=X.shape
    X=np.insert(X, [0], np.ones((m,1)), axis=1)
    th=np.ones((n+1,1))
    k=160
    l=0
    Lambdas=np.array([range(-20,40,2)])
    for Lambda in range(-20,40,2): 
        for j in range(10000):
            for i in range(2):
                (cost, grad) = cf(th, X[k*i:k*i+k,:], y[k*i:k*i+k],Lambda/10.)
                J=cost
                th=gr(th,grad,alpha)
        print ('For Lambda = ', Lambda/10.)
        p=pre(th,X)
        print ('Training Accuracy: ', (((p==y).astype(int)).sum()/float(m))*100)
        TP=(np.logical_and((p==1),(y==1)).astype(int)).sum()
        TN=(np.logical_and((p==0),(y==0)).astype(int)).sum()
        FP=(np.logical_and((p==1),(y==0)).astype(int)).sum()
        FN=(np.logical_and((p==0),(y==1)).astype(int)).sum()
        PR=TP/float(TP+FP)
        RE=TP/float(TP+FN)
        f=(2*PR*RE)/(PR+RE)
        F[0,l]=f
        l=l+1
        (mt,nt)=data.shape
        Xt = tdata[:, 0:nt-1]
        yt = (tdata[:, nt-1:nt])
        Xt=(Xt-np.average(Xt,axis=0))/np.std(Xt,axis=0)
        (mt,nt)=Xt.shape
        Xt=np.insert(Xt, [0], np.ones((mt,1)), axis=1)
        pt=pre(th,Xt)
        print ('Testing Accuracy', (((pt==yt).astype(int)).sum()/float(mt))*100)
    F=F.transpose()
    Lambdas=(Lambdas/10.).transpose()
    plt.plot(Lambdas,F)
    plt.show()    
        
    
if __name__=='__main__':
    main()

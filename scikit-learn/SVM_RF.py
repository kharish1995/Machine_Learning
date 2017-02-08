import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

d=np.genfromtxt('data.txt',delimiter=',')
d1=d[:200,:]
d2=d[251:371,:]
d=np.vstack((d1,d2))
nd1=d[201:251,:]
nd2=d[371:400,:]
nd=np.vstack((nd1,nd2))
(m,n)=d.shape
X1 = nd[:, 0:n-1]
X = d[:, 0:n-1]
y = (d[:, n-1])
(mt,nt)=X1.shape
yt = (nd[:, n-1])

svc = svm.SVC(kernel='linear', C=1.0).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1.0).fit(X, y)
RFC=RandomForestClassifier(n_estimators=25).fit(X,y)

p= svc.predict(X)
print ('fmeasure of linear kernel: ',f1_score(p,y))

p= rbf_svc.predict(X)
print ('fmeasure of rbf kernel: ',f1_score(p,y))

p= RFC.predict(X)
print ('fmeasure of Random Forest: ',f1_score(p,y))

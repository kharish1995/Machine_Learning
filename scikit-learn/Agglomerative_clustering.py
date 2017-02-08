from time import time
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.preprocessing import scale
digits = datasets.load_digits(n_class=10)
X = scale(digits.data)
y = digits.target
n_samples, n_features = X.shape
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
clusnew = np.zeros(len(X))
clustering = AgglomerativeClustering(linkage='ward', n_clusters=10)
t0 = time()
clustering.fit(X_red)
print("%s : %.2fs" % ('ward', time() - t0))
c1 = np.zeros((10,10))
d = np.zeros(10)
c = confusion_matrix(clustering.labels_,y)
for j in range (0,10):
	c1[j,:] = c[:,(np.argmax(c[j,:]))]
	clusnew[clustering.labels_ == (np.argmax(c[j,:]))] = j
	d[j] = sum(c1[:,j])
c1[:,(np.argmin(d))] = -1	
print ('Confusion Matrix: ' ,c1)
print ('Fowlkes Mallows Score: ' ,fowlkes_mallows_score(y,clusnew))	


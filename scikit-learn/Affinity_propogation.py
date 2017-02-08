import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn import manifold, datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.preprocessing import scale
digits = datasets.load_digits()
X = scale(digits.data)
y = digits.target
af = AffinityPropagation(preference=-3000, damping = 0.7).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(cluster_centers_indices)
c1 = np.zeros((len(cluster_centers_indices),len(cluster_centers_indices)))
d = np.zeros(len(cluster_centers_indices))
clusnew = np.zeros(len(X))
c = confusion_matrix(labels,y)
print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels))
print("Completeness: %0.3f" % metrics.completeness_score(y, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(y, labels))

print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(y, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(y, labels))

for j in range (0,len(cluster_centers_indices)):
	c1[j,:] = c[:,(np.argmax(c[j,:]))]
	clusnew[labels == (np.argmax(c[j,:]))] = j
	d[j] = sum(c1[:,j])
#c1[:,(np.argmin(d))] = -1	
print ('Confusion Matrix: ' , c1)
print ('Fowlkes Mallows Score: ', fowlkes_mallows_score(y,labels))	


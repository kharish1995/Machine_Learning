from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import fowlkes_mallows_score
digits = load_digits()
data = scale(digits.data)
k = 10;
np.set_printoptions(threshold = 2000)	

def initialize_centroids(data,k):
	
	centroids = data[np.random.randint(0, len(data), size=k)]
	return centroids

def closest_centroid(data, centroids, k):
	
	distances = np.zeros((len(data),k))
	cluster = np.zeros(len(data))
	for i in range(0, k):
		for j in range(0, len(data)):
			distances[j,i] = np.linalg.norm(data[j,:]-centroids[i,:])	
	for j in range(0, len(data)):								
		cluster[j] = np.argmin(distances[j,:])	
	return cluster

def kmeans(data, k):
	
	centroid = initialize_centroids(data,k)
	a = np.zeros((k,k))
	b = np.zeros(k)
	c1 = np.zeros((k,k))
	d = np.zeros(k)
	clusnew = np.zeros(len(data))
	i = 1
	while ( i < 100):
		clusters = closest_centroid(data, centroid, k)
		for l in range(0,k):
			centroid[l,:] = np.mean(data[(np.where(clusters == l))], axis = 0)
		i = i + 1
		print i	
	c = confusion_matrix(clusters,digits.target)
	for j in range (0,k):
		c1[j,:] = c[:,(np.argmax(c[j,:]))]
		clusnew[clusters == (np.argmax(c[j,:]))] = j
		d[j] = sum(c1[:,j])
	c1[:,(np.argmin(d))] = -1	
	print ('Confusion Matrix: ' , c1)
	print ('Fowlkes Mallows Score: ', fowlkes_mallows_score(digits.target,clusnew))	
		
def main():
	kmeans(data,k)

if __name__ == "__main__":
    main()
	 

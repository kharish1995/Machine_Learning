import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
data  = np.loadtxt(open("realdata.txt"))
k = 8 

def initialize_centroids(data,k):
	
	centroids = data[np.random.randint(0, len(data), size=k)]
	return centroids

def closest_centroid(data, centroids, k):
	
	distances = np.zeros((len(data),k))
	cluster = np.zeros(len(data))
	for i in range(0, k):
		for j in range(0, len(data)):
			distances[j,i] = np.linalg.norm(data[j,1:3]-centroids[i,1:3])	
	for j in range(0, len(data)):								
		cluster[j] = np.argmin(distances[j,:])
	return cluster

def kmeans(data, k):
	
	centroid = initialize_centroids(data,k)
	i = 1
	while ( i < 50):
		clusters = closest_centroid(data, centroid, k)
		for l in range(0,k):
			centroid[l,:] = np.mean(data[(np.where(clusters == l))], axis = 0)
		i = i + 1
	d1 = data[:,1:2]
	d2 = data[:,2:3]
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	colours = cm.rainbow(np.linspace(0,1,k))
	for j in range (0,k):
		ax1.scatter(d1[(clusters==j)], d2[(clusters==j)], s= 20, c = colours[j], label = 'cluster' + str(j+1))
	plt.legend(loc = 'lower right', prop = {'size': 9}) 
	plt.ylabel('Width')
	plt.xlabel('Length')
	plt.show()	
		
def main():
	kmeans(data,k)
         
if __name__ == "__main__":
    main()
	 

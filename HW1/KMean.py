from copy import deepcopy
import numpy as np
import pandas as pd
import random
#from matplotlib import pyplot as plt


def random_start(k=3, amount=9):
	a = list(range(0,amount))
	random.shuffle(a)
	return a[:k]

x_val = [1,3,2,8,6,7,-3,-2,-7]
y_val = [2,3,2,8,6,7,-3,-4,-7]
inp1 = np.array(list(zip(x_val, y_val)))
def k_mean(dt,k=3,loop=50):
	min_MSE = np.inf
	for q in range(50):
		mask_num = random_start(k)
		C = np.array(dt[mask_num])
		C_prior = np.zeros(C.shape)
		clusters = np.zeros(len(dt))
		error = np.linalg.norm(C - C_prior, axis=None)
		while error != 0:
			for i in range(len(dt)):
				distances = np.linalg.norm(dt[i] - C, axis=1)
				cluster = np.argmin(distances)
				clusters[i] = cluster
				#print("Considering point ",dt[i], ", it is closest to the cluster ",cluster," which has center located at ", C[cluster])
			C_prior = deepcopy(C)
			for i in range(k):
				points = [dt[j] for j in range(len(dt)) if clusters[j] == i]
				C[i] = np.mean(points, axis=0)
				#print("New assign center for cluster ",i," is ",C[i])
			error = np.linalg.norm(C - C_prior, axis=None)
			#print("Recalculated error function is ", error)
		MSE = 0.0
		for i in range(len(dt)):
				MSE += np.linalg.norm(dt[i] - C[int(clusters[i])], axis=0)**2
		MSE /= len(dt)
		if min_MSE > MSE:
			min_MSE = MSE
			min_C = C
			min_clusters = clusters
	print("min MSE:", min_MSE)
	data_mean = dt.sum(axis=0) / dt.shape[0]
	print("data mean ", data_mean)
	between_clus_var = 0
	for u in range(k):
		in_clus = [dt[v] for v in range(dt.shape[0]) if min_clusters[v] == u]
		print("print len ", len(in_clus))
		print("print min C for ", u, " that is ", min_C[u])
		between_clus_var += len(in_clus) * ((min_C[u] - data_mean)**2).sum() / (dt.shape[0] - 1)
	all_data_var = ((dt - data_mean)**2).sum(axis=1).sum(axis=0) / (dt.shape[0] - 1)
	print("using k=", k)
	print('between clus var:', between_clus_var)
	print('all data var', all_data_var)
	print("fractional: ", between_clus_var / all_data_var)
	return between_clus_var / all_data_var

for i in range(1,3):
	k_mean(inp1,i,50)
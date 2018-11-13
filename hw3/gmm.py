from copy import deepcopy
import numpy as np
import scipy.stats
import pandas as pd
import random
from matplotlib import pyplot as plt


def random_start(k=3, amount=9):
	a = list(range(0,amount))
	random.shuffle(a)
	return a[:k]

x_val = [1,3,2,8,6,7,-3,-2,-7]
y_val = [2,3,2,8,6,7,-3,-4,-7]
input_arr = np.array(list(zip(x_val, y_val)))



def gmm(input, init_mean, k, ittr):
	llh = []
	init_mean = np.array(init_mean)
	cov_mt = np.identity(init_mean.shape[1])
	cov_arr = []
	gm = []
	mean = np.zeros((k, input.shape[1]), dtype=np.float64)
	phi = np.zeros(k, dtype=np.float64)
	num_data = input.shape[0]
	#initialize
	for j in range(k):
		gm.append(scipy.stats.multivariate_normal(mean=init_mean[j], cov=cov_mt))
		mean[j] = init_mean[j]
		cov_arr.append(cov_mt)
		phi[j] = 1/k
	print("initialize mean:", mean)
	print("initialize cov:", cov_arr)
	print("initialize phi:", phi)
	for r in range(ittr):
		print("Itteration: ", r)
		print("Using mean: ", mean)
		print("Using covariant: ", cov_arr)
		print("Using phi: ", phi)
		w = np.zeros((input.shape[0], k), dtype=np.float64)
		w_prob = np.zeros((input.shape[0], k), dtype=np.float64)
		#Compute W matrix
		for i in range(num_data):
			n_sum = 0
			for j in range(k):
				g_val = gm[j].pdf(input[i]) * phi[j]
				w[i][j] = g_val
				w_prob[i][j] = g_val
				n_sum += g_val
			#divide by the sum
			w[i] = w[i]/n_sum
		#finished computing W matrix
		print("W matrix: ", w)
		#compute log likelyhood
		llh.append(np.log(np.prod(np.sum(w_prob, axis=1))))
		old_mean = np.copy(mean)
		col_sum = np.sum(w, axis=0)
		#compute the phi matrix
		phi = np.sum(w, axis=0) / num_data
		print("New phi: ", phi)
		#compute the mu
		for j in range(k):
			mean[j] = np.array((0,0), dtype=np.float64)
			for i in range(num_data):
				mean[j] += w[i][j] * input[i]
			mean[j] = mean[j] / col_sum[j]
		#finish computing mu
		print("New mu: ", mean)
		#computing sigma
		for j in range(k):
			temp_cov = np.zeros((input.shape[1], input.shape[1]), dtype=np.float64)
			for i in range(num_data):
				vv_dif = input[i] - old_mean[j]
				temp_cov = temp_cov + w[i][j] * (vv_dif * vv_dif[np.newaxis].T)
			temp_cov = temp_cov / col_sum[j]
			temp_cov[0][1] = 0.0
			temp_cov[1][0] = 0.0
			cov_arr[j] = temp_cov
		#finish compute sigma
		print("New sigma: ", cov_arr)
		#update model
		for j in range(k):
			gm[j] = scipy.stats.multivariate_normal(mean=mean[j], cov=cov_arr[j])
	return llh, list(range(ittr))

plt.ion()
q1_mean = [[3,3],[2,2],[-3,-3]]
y,x = gmm(input_arr, q1_mean, 3, 4)
plt.plot(x[1:],y[1:])
plt.savefig("T1 LLH.png")
plt.show()
plt.pause(0.1)
plt.close()
q2_mean = [[3,3],[-3,-3]]
y,x = gmm(input_arr, q2_mean, 2, 4)
plt.plot(x[1:],y[1:])
plt.savefig("T3 LLH.png")
plt.show()
plt.pause(0.1)
plt.close()
from tqdm import tqdm
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_float
import numpy as np
np.set_printoptions(threshold=np.nan)
def distance(x,y):
	return np.sqrt(np.sum((x-y)**2))

plt.ion()
data = scipy.io.loadmat("C:\\Users\\admin\\Dropbox\\hw3\\facedata")
# face data is a 2-dimensional array with size 40x10
x = data['facedata']
xf = {}
xd = {}
for i in range(40):
	for j in range(10):
		xf[i,j] = img_as_float(x[i,j])
		xd[i,j] = xf[i,j].reshape(56*46)
dis1 = distance(xd[0,0],xd[0,1])
dis2 = distance(xd[0,0],xd[1,0])
print("distance between 0,0 and 0,1: ", dis1)
print("Distance between 0,0 and 1,0: ", dis2)

def sim (data, t,d):
	mat = np.zeros((len(t), len(d)), dtype=np.float64)
	for i in range(len(t)):
		for j in range(len(d)):
			mat[i][j] = distance(data[t[i]], data[d[j]])
	return mat

l_t = []
l_d = []

for i in range(40):
	for j in range(3):
		l_t.append((i,j))
	for j in range(3,10):
		l_d.append((i,j))

mat = sim(xd, l_t, l_d)

def evaluate_sim(sim, p):
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	for i in range(0,120,3):
		idx = i//3
		for j in range(280):
			jdx = j//7
			comp = min(sim[i][j], sim[i+1][j], sim[i+2][j])
			if ((comp < p) and (idx == jdx)):
				tp += 1
			elif ((comp < p) and (idx != jdx)):
				fp += 1
			elif ((comp >= p) and (idx == jdx)):
				fn += 1
			elif ((comp >= p) and (idx != jdx)):
				tn += 1
	tp_fn = tp+fn if tp+fn > 0 else 1
	fp_tn = fp+tn if fp+tn > 0 else 1
	return (tp/tp_fn),(fp/fp_tn), (fn/tp_fn), tp, fp, fn, tn

plt.imsave("T6 Sim mat.png", mat.T, cmap='gray')
print(evaluate_sim(mat, 10))

def simpleModel(data, t, d, start, end, num, queryThreshold, analysis=False):
	mat = sim(data, t, d)
	if analysis==True:
		print("max value=", mat.max(), ", min value=", mat.min())
	spaces = np.linspace(start,end,num)
	t_r = []
	f_r = []
	frr_r = []
	p_v = []
	for r in tqdm(spaces):
		t_rate, f_rate, frr_rate, tp, fp, fn, tn = evaluate_sim(mat, r)
		t_r.append(t_rate)
		f_r.append(f_rate)
		frr_r.append(frr_rate)
		p_v.append(r)
	t_r_arr = np.array(t_r)
	f_r_arr = np.array(f_r)
	frr_r_arr = np.array(frr_r)
	eer_idx = (np.abs(f_r_arr - frr_r_arr)).argmin()
	q10_idx = (np.abs(f_r_arr - queryThreshold)).argmin()
	print("ERR is at when FAR=", f_r_arr[eer_idx], " and FRR=", frr_r_arr[eer_idx], " with threshold=", p_v[eer_idx])
	print("When FAR=", f_r_arr[q10_idx], " the recall=", t_r_arr[q10_idx])
	return t_r, f_r

recall, f_r = simpleModel(xd, l_t, l_d, 2, 17, 4000, 0.001)
plt.plot(f_r, recall)
plt.savefig("T9-ROC.png")
plt.show()
plt.pause(1)
plt.close()
roc_1x = f_r
roc_1y = recall
l = []
for q in l_t:
	l.append(xd[q])
l = np.array(l)
l = l.T
m = np.mean(l,axis=1)
m = m.reshape(-1,1)
m_img = m.reshape(56,46)
plt.imsave("T11 meanface.png", m_img, cmap='gray')
l_hat = l-m
l_cov = np.cov(l_hat)
l_cov_rank = np.linalg.matrix_rank(l_cov, hermitian=True)
print("size of cov matrix=", l_cov.shape, " and rank of cov matrix=", l_cov_rank)
l_val, l_vec = np.linalg.eigh(l_cov)
l_val = np.round(l_val, 12)
l_sort = l_val.argsort()[::-1]
l_val = l_val[l_sort]
l_vec = l_vec[:,l_sort]
l_norm = np.linalg.norm(l_vec, axis=0).reshape((1,-1))
l_vec_normalized = l_vec/l_norm
g_matrix = l_hat.T.dot(l_hat)
g_rank = np.linalg.matrix_rank(g_matrix, hermitian=True)
print("size of gram matrix=", g_matrix.shape, " and rank of gram matrix=", g_rank)
g_val, g_vec = np.linalg.eigh(g_matrix)
g_val = np.round(g_val, 12)
g_sort = g_val.argsort()[::-1]
g_val = g_val[g_sort]
g_vec = g_vec[:, g_sort]
g_cov_eig = l_hat.dot(g_vec)
g_norm = np.linalg.norm(g_cov_eig, axis=0).reshape((1,-1))
g_vec_normalized = g_cov_eig/g_norm
g_project = g_vec_normalized.T.dot(l_hat)
plt.plot(list(range(g_rank)), g_val[:g_rank])
plt.savefig("T16.png")
plt.show()
plt.pause(1)
plt.close()
count = 0
target = 0.95*np.sum(g_val)
cum_sum = 0
for i in range(len(g_val)):
	cum_sum += g_val[i]
	count += 1
	if cum_sum >= target:
		break

print("We should use ", count, " eigenvalues to cover 95% variant")
for i in range(10):
	img = g_vec_normalized[:,i]
	#img = np.interp(imt, (img.min(), img.max()), (0,1))
	img = img.reshape((56,46))
	plt.imsave("T17-"+str(i+1)+".png", img, cmap='gray')

def prePCA(data, t, mean=None, convertOnly=False):
	l = []
	for q in t:
		l.append(data[q])
	l = np.array(l)
	l = l.T
	if convertOnly == True:
		return l
	if mean is None:
		m = np.mean(l,axis=1)
		m = m.reshape(-1,1)
		l_hat = l-m
		return l_hat, m.reshape(-1)
	else:
		return l - mean.reshape(-1,1)

def PCA(data, t):
	l_hat, l_mean = prePCA(data,t)
	g_matrix = l_hat.T.dot(l_hat)
	g_rank = np.linalg.matrix_rank(g_matrix, hermitian=True)
	g_val, g_vec = np.linalg.eigh(g_matrix)
	g_val = np.round(g_val, 12)
	g_sort = g_val.argsort()[::-1]
	g_val = g_val[g_sort]
	g_vec = g_vec[:, g_sort]
	g_cov_eig = l_hat.dot(g_vec)
	g_norm = np.linalg.norm(g_cov_eig, axis=0).reshape((1,-1))
	g_vec_normalized = g_cov_eig/g_norm
	return g_val, 	g_vec_normalized, g_rank, l_mean, t

def reconstruct(data, arr, t):
	for i in range(len(t)):
		data[t[i]] = arr[:,i]

def decompress(mu, v, p, k):
	mu = mu.reshape(-1,1)
	reconstruct = v[:,:k].dot(p)
	return mu+reconstruct

def MSE(t,d):
	return np.sum(((t-d)**2), axis=0)/len(t)
print("PCA with k=10")
t_val, t_vec, t_rank, t_mean, t_idx = PCA(xd, l_t)
t_hat = prePCA(xd, l_t, t_mean)
d_hat = prePCA(xd, l_d, t_mean)
t_project = t_vec.T.dot(t_hat)[:10]
d_project = t_vec.T.dot(d_hat)[:10]
new_xd = {}
reconstruct(new_xd, t_project, l_t)
reconstruct(new_xd, d_project, l_d)
recall, f_r = simpleModel(new_xd, l_t, l_d, 2, 16, 3000, 0.001, False)
plt.plot(f_r, recall)
plt.savefig("T18 ROC.png")
plt.show()
plt.pause(0.1)
plt.close()
roc_2x = f_r
roc_2y = recall

print("Using PCA...")
for k in range(2,15):
	print("Using k=", k)
	t_val, t_vec, t_rank, t_mean, t_idx = PCA(xd, l_t)
	t_hat = prePCA(xd, l_t, t_mean)
	d_hat = prePCA(xd, l_d, t_mean)
	t_project = t_vec.T.dot(t_hat)[:k]
	d_project = t_vec.T.dot(d_hat)[:k]
	new_xd = {}
	reconstruct(new_xd, t_project, l_t)
	reconstruct(new_xd, d_project, l_d)
	recall, f_r = simpleModel(new_xd, l_t, l_d, 2, 16, 3000, 0.001, False)

print("Test image decompression...")
t_val, t_vec, t_rank, t_mean, t_idx = PCA(xd, l_t)
t_original = prePCA(xd, l_t, convertOnly=True)
t_hat = prePCA(xd, l_t, t_mean)
t_project = t_vec.T.dot(t_hat)[:10]
t_reconstruct = decompress(t_mean, t_vec, t_project, 10)
t_mse = MSE(t_original, t_reconstruct)
print("MSE for the first image when k=10:", t_mse[0])
print("Sweeping k value...")
mse_x = []
mse_y = []
for k in list(range(1,11))+[119]:
	mse_x.append(k)
	t_project = t_vec.T.dot(t_hat)[:k]
	t_reconstruct = decompress(t_mean, t_vec, t_project, k)
	t_img = t_reconstruct[:,0].reshape(56,46)
	plt.imsave("OT2-K"+str(k)+".png", t_img, cmap='gray')
	t_mse = MSE(t_original, t_reconstruct)
	mse_y.append(t_mse[0])
	print("MSE for the first image when k=", k, ":", t_mse[0])
plt.plot(mse_x, mse_y)
plt.savefig("OT2 MSE plot.png")
plt.show()
plt.pause(1)
plt.close()

def computeSwSb (data, t, k, mean=None, start=0, end=120, step=3):
	t_val, t_vec, t_rank, t_mean, t_idx = PCA(data, t)
	if mean is None:
		t_hat,g = prePCA(data, t)
	else:
		t_hat = prePCA(data, t, mean)
	t_project = t_vec.T.dot(t_hat)[:k]
	print("TP:", t_project.shape)
	t_mean = np.mean(t_project, axis=1)
	print("LDA global:", t_mean)
	t_sb = np.zeros((t_mean.shape[0], t_mean.shape[0]))
	t_sw = np.zeros((t_mean.shape[0], t_mean.shape[0]))
	for i in range(start,end,step):
		class_mean = np.mean(t_project[:,i:i+step], axis=1)
		class_dif = class_mean - t_mean
		class_dif = class_dif.reshape(-1,1)
		class_temp = class_dif.dot(class_dif.T)
		t_sb = t_sb + class_temp
	for i in range(start,end,step):
		class_mean = np.mean(t_project[:,i:i+step], axis=1)
		for j in range(0,step):
			class_data = t_project[:,i+j]
			data_dif = class_data - class_mean
			data_dif = data_dif.reshape(-1,1)
			data_temp = data_dif.dot(data_dif.T)
			t_sw = t_sw + data_temp
	return t_sw, t_sb, t_project

for k in tqdm(range(1,120)):
	t_sw, t_sb, t_project = computeSwSb(xd, l_t, k)
	print("For k=", k, " the rank of matrix=", np.linalg.matrix_rank(t_sw))
print("try doing it with test set...")
for k in tqdm(range(1,280)):
	t_sw, t_sb, t_project = computeSwSb(xd, l_d, k, mean=t_mean, start=0, end=280, step=7)
	print("For k=", k, " the rank of matrix=", np.linalg.matrix_rank(t_sw))


k=80
t_val, t_vec, t_rank, t_mean, t_idx = PCA(xd, l_t)
d_hat = prePCA(xd, l_d, t_mean)
d_project_first = t_vec.T.dot(d_hat)[:k]
t_sw, t_sb, t_project = computeSwSb(xd, l_t, k)
swinv = np.linalg.inv(t_sw)
s=swinv.dot(t_sb)
print("The rank of SW^-1 * SB of train set=", np.linalg.matrix_rank(s))
s_val, s_vec = np.linalg.eig(s)
s_val = np.real(s_val)
s_val = np.round(s_val, 12)
s_sort = s_val.argsort()[::-1]
s_val = s_val[s_sort]
s_vec = s_vec[:,s_sort]
s_numvec = np.linalg.matrix_rank(s)
s_realvec = np.real(s_vec[:,:s_numvec])
print("LDA projection is ", s_realvec)
s_pre_project = np.concatenate([s_realvec, np.zeros((k, k-s_numvec))], axis=1)
s_pre_project_norm = np.linalg.norm(s_pre_project, axis=0).reshape((1,-1))
s_pre_project = s_pre_project/s_pre_project_norm
s_project = s_pre_project.T.dot(t_project)[:39]
d_project = s_pre_project.T.dot(d_project_first)[:39]

s_subvector = np.real(s[:,:s_numvec])[:,:10]
s_reconstruct = decompress(t_mean, t_vec, s_subvector, 80)
for i in range(10):
	s_img = s_reconstruct[:,i]
	s_img = np.interp(s_img, (s_img.min(), s_img.max()), (0,1))
	s_img = s_img.reshape(56,46)
	plt.imsave("T23-"+str(i+1)+".png", s_img, cmap='gray')

new_xd = {}
reconstruct(new_xd, s_project, l_t)
reconstruct(new_xd, d_project, l_d)
recall, f_r = simpleModel(new_xd, l_t, l_d, 1, 6, 4000, 0.001, True)
plt.plot(f_r, recall)
plt.savefig("T24 ROC.png")
plt.show()
plt.pause(0.1)
plt.close()
roc_3x = f_r
roc_3y = recall
plt.plot(roc_1x, roc_1y, 'g')
plt.plot(roc_2x, roc_2y, 'b')
plt.plot(roc_3x, roc_3y, 'y')
plt.savefig("T25 all ROC.png")
plt.show()
plt.pause(0.1)
plt.close()
color = ['g', 'b', 'y', 'c', 'm', 'r']
for i in range(0,6):
	x = d_project[0, 7*i:7*i+7]
	y = d_project[1, 7*i:7*i+7]
	plt.plot(x,y,color[i])
plt.savefig("OT4 ROC of LDA.png")
plt.show()
plt.pause(0.1)
plt.close()


for i in range(0,6):
	x = d_project_first[0, 7*i:7*i+7]
	y = d_project_first[1, 7*i:7*i+7]
	plt.plot(x,y,color[i])
plt.savefig("OT4 ROC of PCA.png")
plt.show()
plt.pause(0.1)
plt.close()

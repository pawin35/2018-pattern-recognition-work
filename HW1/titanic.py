import numpy as np
import pandas as pd
#T9
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url) #training set
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url) #test set
print("printing training head...")
print(train.head())
print("printing training tail...")
print(train.tail())
print("Describing the training data...")
print(train.describe())
#T7
print("The median age of training set is ", train["Age"].median())
#T8
print("The mode of the embarked is", train["Embarked"].mode()[0])


def preProcess(x, train):
	x["Age"] = x["Age"].fillna(train["Age"].median())
	x.loc[x["Embarked"] == "S", "Embarked"] = 0
	x.loc[x["Embarked"] == "C", "Embarked"] = 1
	x.loc[x["Embarked"] == "Q", "Embarked"] = 2
	x["Embarked"] = x["Embarked"].fillna(train["Embarked"].mode()[0])
	x.loc[x["Sex"] == "male", "Sex"] = 0
	x.loc[x["Sex"] == "female", "Sex"] = 1
	return np.array(x[["Pclass","Sex","Age","Embarked"]].values, dtype = float)

data_train = preProcess(train, train)
data_train=np.concatenate((np.ones((len(data_train),1)), data_train), axis=1)
data_test = preProcess(test, train)
data_test=np.concatenate((np.ones((len(data_test),1)), data_test), axis=1)
lable_train = np.array(train[["Survived"]].values)
#label_test = np.array(test[["Survived"]].values)

def sigmoid(z):
	return 1/(1+np.exp(-z))

def z(x, theta):
	return x.dot(theta)

def h_log(x,theta):
	return sigmoid(z(x,theta))

def h_lin(x, theta):
	return x.dot(theta)

def log_loss(y,predicted,eps=1e-15):
	predicted = np.clip(predicted, eps, 1 - eps)
	return np.sum((-y*np.log(predicted) - (1-y)*np.log(1 - predicted)) / y.shape[0])

def mean_square_loss(y, predicted):
	error = predicted - y
	return np.sum(error*error) / y.shape[0]

def gradient_descent(theta, x, y, r,h):
	error = h(x,theta) - y
	delta = r*np.matmul(x.T, error)
	return theta - delta

def logistic_train(x, y, epoc, r):
	theta = np.random.uniform(-1, 1, (x.shape[1], 1))
	losses = []
	for i in range(epoc):
		predicted = h_log(x,theta)
		theta = gradient_descent(theta, x, y, r, h_log)
		losses.append(log_loss(y, predicted))
	return theta, losses

theta, losses = logistic_train(data_train, lable_train, 3000, 0.00001)
print("last 10 llosses: ", losses[-11:])
train_result = h_log(data_train, theta) > 0.5
predictions = h_log(data_test, theta) > 0.5
predictions = predictions.astype(int)
train_acc = np.count_nonzero(np.equal(lable_train, train_result))/data_train.shape[0]
#test_acc = np.count_nonzero(np.equal(label_test, prediction))/len(data_test.shape[0])
print("The accuracy of the training set is ", train_acc)
#print("The accuracy of the testing set is ", test_acc)

test_result = pd.DataFrame()
test_result['PassengerId'] = test['PassengerId']
test_result['Survived'] = predictions
print("printing result...")
#print(test_result)
test_result.to_csv("titanic_result.csv", index = False)

#OT2
print("begin OT2...")
train = pd.read_csv(train_url) #training set
test = pd.read_csv(test_url) #test set
data_train_OT2 = preProcess(train, train)
data_train_OT2=np.concatenate((np.ones((len(data_train_OT2),1)), data_train_OT2), axis=1)
data_test_OT2 = preProcess(test, train)
data_test_OT2=np.concatenate((np.ones((len(data_test_OT2),1)), data_test_OT2), axis=1)

def linear_train(x, y, epoc, r):
	theta = np.random.uniform(-1, 1, (x.shape[1], 1))
	losses = []
	for i in range(epoc):
		predicted = h_lin(x,theta)
		theta = gradient_descent(theta, x, y, r, h_lin)
		losses.append(mean_square_loss(y, predicted))
	return theta, losses

theta_OT2, losses_OT2 = linear_train(data_train_OT2, lable_train, 150000, 0.000002)
train_result_OT2 = h_lin(data_train_OT2, theta_OT2) > 0.3
predictions_OT2 = h_lin(data_test_OT2, theta_OT2) > 0.3
train_acc_OT2 = np.count_nonzero(np.equal(lable_train, train_result_OT2))/data_train_OT2.shape[0]

print("The accuracy of the training set of OT 2 is ", train_acc_OT2)

#OT3
print("begin OT3...")
train = pd.read_csv(train_url) #training set
test = pd.read_csv(test_url) #test set
data_train_OT3 = preProcess(train, train)
data_train_OT3=np.concatenate((np.ones((len(data_train_OT3),1)), data_train_OT3), axis=1)
data_test_OT3 = preProcess(test, train)
data_test_OT3=np.concatenate((np.ones((len(data_test_OT3),1)), data_test_OT3), axis=1)

def matrix_train(x, y):
	theta = np.linalg.pinv(x).dot(y)
	return theta
	




theta_OT3 = matrix_train(data_train_OT3, lable_train)
train_result_OT3 = h_lin(data_train_OT3, theta_OT3) > 0.3
predictions_OT3 = h_lin(data_test_OT3, theta_OT3) > 0.3
print("The mean square loss of OT 3 is ", mean_square_loss(lable_train, h_lin(data_train_OT3, theta_OT3)))
train_acc_OT3 = np.count_nonzero(np.equal(lable_train, train_result_OT3))/data_train_OT3.shape[0]

print("The accuracy of the training set of OT 3 is ", train_acc_OT3)
print("old theta is ", theta_OT2)
print("New theta is ", theta_OT3)
theta_diff = theta_OT2 - theta_OT3
print("Theta difference is ",theta_diff)
print("MSE of theta difference is ", theta_diff.T.dot(theta_diff) / theta.shape[0])



#OT4A
print("begin OT4A...")
train = pd.read_csv(train_url) #training set
test = pd.read_csv(test_url) #test set

data_train_OT4A = preProcess(train, train)
data_train_OT4A=np.concatenate((data_train_OT4A, np.array([data_train[:, 1]*data_train[:, 2]]).T), axis=1)
data_train_OT4A=np.concatenate((data_train_OT4A, np.array([data_train[:, 1]*data_train[:, 1]]).T), axis=1)
data_train_OT4A=np.concatenate((np.ones((len(data_train_OT4A),1)), data_train_OT4A), axis=1)
data_test_OT4A = preProcess(test, train)
data_test_OT4A=np.concatenate((data_test_OT4A, np.array([data_test[:, 1]*data_test[:, 2]]).T), axis=1)
data_test_OT4A=np.concatenate((data_test_OT4A, np.array([data_test[:, 1]*data_test[:, 1]]).T), axis=1)
data_test_OT4A=np.concatenate((np.ones((len(data_test_OT4A),1)), data_test_OT4A), axis=1)


theta_OT4A, losses_OT4A = logistic_train(data_train_OT4A, lable_train, 3000, 0.00001)
train_result_OT4A = h_log(data_train_OT4A, theta_OT4A) > 0.5
predictions_OT4A = h_log(data_test_OT4A, theta_OT4A) > 0.5
train_acc_OT4A = np.count_nonzero(np.equal(lable_train, train_result_OT4A))/data_train_OT4A.shape[0]

print("The accuracy of the training set of OT 4A (adding age*sex and age^2) is ", train_acc_OT4A)

#OT4B
print("begin OT4B...")
train = pd.read_csv(train_url) #training set
test = pd.read_csv(test_url) #test set

data_train_OT4B = preProcess(train, train)
data_train_OT4B=np.concatenate((data_train_OT4B, np.array([data_train[:, 1]*data_train[:, 2]]).T), axis=1)
data_train_OT4B=np.concatenate((data_train_OT4B, np.array([data_train[:, 2]*data_train[:, 2]]).T), axis=1)
data_train_OT4B=np.concatenate((np.ones((len(data_train_OT4B),1)), data_train_OT4B), axis=1)
data_test_OT4B = preProcess(test, train)
data_test_OT4B=np.concatenate((data_test_OT4B, np.array([data_test[:, 1]*data_test[:, 2]]).T), axis=1)
data_test_OT4B=np.concatenate((data_test_OT4B, np.array([data_test[:, 2]*data_test[:, 2]]).T), axis=1)
data_test_OT4B=np.concatenate((np.ones((len(data_test_OT4B),1)), data_test_OT4B), axis=1)


theta_OT4B, losses_OT4B = logistic_train(data_train_OT4B, lable_train, 3000, 0.00001)
train_result_OT4B = h_log(data_train_OT4B, theta_OT4B) > 0.5
predictions_OT4B = h_log(data_test_OT4B, theta_OT4B) > 0.5
train_acc_OT4B = np.count_nonzero(np.equal(lable_train, train_result_OT4B))/data_train_OT4B.shape[0]

print("The accuracy of the training set of OT 4B (adding age*sex and sex^2) is ", train_acc_OT4B)



#OT4C
print("begin OT4C...")
train = pd.read_csv(train_url) #training set
test = pd.read_csv(test_url) #test set

data_train_OT4C = preProcess(train, train)
data_train_OT4C=np.concatenate((data_train_OT4C, np.array([data_train[:, 1]*data_train[:, 2]]).T), axis=1)
data_train_OT4C=np.concatenate((data_train_OT4C, np.array([data_train[:, 3]*data_train[:, 3]]).T), axis=1)
data_train_OT4C=np.concatenate((np.ones((len(data_train_OT4C),1)), data_train_OT4C), axis=1)
data_test_OT4C = preProcess(test, train)
data_test_OT4C=np.concatenate((data_test_OT4C, np.array([data_test[:, 1]*data_test[:, 2]]).T), axis=1)
data_test_OT4C=np.concatenate((data_test_OT4C, np.array([data_test[:, 3]*data_test[:, 3]]).T), axis=1)
data_test_OT4C=np.concatenate((np.ones((len(data_test_OT4C),1)), data_test_OT4C), axis=1)


theta_OT4C, losses_OT4C = logistic_train(data_train_OT4C, lable_train, 300000, 0.0000001)
train_result_OT4C = h_log(data_train_OT4C, theta_OT4C) > 0.5
predictions_OT4C = h_log(data_test_OT4C, theta_OT4C) > 0.5
train_acc_OT4C = np.count_nonzero(np.equal(lable_train, train_result_OT4C))/data_train_OT4C.shape[0]

print("The accuracy of the training set of OT 4C (adding age*sex and embarked^2) is ", train_acc_OT4C)







#OT5
print("begin OT5...")
train = pd.read_csv(train_url) #training set
test = pd.read_csv(test_url) #test set
def preProcess_OT5(x, train):
	x["Age"] = x["Age"].fillna(train["Age"].median())
	x.loc[x["Embarked"] == "S", "Embarked"] = 0
	x.loc[x["Embarked"] == "C", "Embarked"] = 1
	x.loc[x["Embarked"] == "Q", "Embarked"] = 2
	x["Embarked"] = x["Embarked"].fillna(train["Embarked"].mode()[0])
	x.loc[x["Sex"] == "male", "Sex"] = 0
	x.loc[x["Sex"] == "female", "Sex"] = 1
	return np.array(x[["Sex","Age"]].values, dtype = float)

data_train_OT5 = preProcess_OT5(train, train)
data_train_OT5=np.concatenate((np.ones((len(data_train_OT5),1)), data_train_OT5), axis=1)
data_test_OT5 = preProcess_OT5(test, train)
data_test_OT5=np.concatenate((np.ones((len(data_test_OT5),1)), data_test_OT5), axis=1)

theta_OT5, losses_OT5 = logistic_train(data_train_OT5, lable_train, 3000, 0.00001)
train_result_OT5 = h_log(data_train_OT5, theta_OT5) > 0.5
predictions_OT5 = h_log(data_test_OT5, theta_OT5) > 0.5
train_acc_OT5 = np.count_nonzero(np.equal(lable_train, train_result_OT5))/data_train_OT5.shape[0]

print("The accuracy of the training set of OT 5 is ", train_acc_OT5)

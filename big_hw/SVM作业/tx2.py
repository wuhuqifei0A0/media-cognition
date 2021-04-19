#%%
from sklearn import svm
from sklearn import preprocessing 
import numpy as np
import scipy.io as sio
import cvxpy as cp
#%%
# 装载数据
data = sio.loadmat('Caltech-256_VGG_10classes.mat')
traindata = data['traindata']
testdata = data['testdata']
clf = svm.SVC(kernel='linear')
x_train = traindata[0][0][0].transpose()
y_train = traindata[0][0][1].ravel()
x_test = testdata[0][0][0].transpose()
y_test = testdata[0][0][1].ravel()
#%% [markdown]
#assignment 1
#%%
# use lable 1 compare with lable2-10
y_train[30:] = 0
y_test[69:] = 0
#%%
scalar = preprocessing.StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.fit_transform(x_test)
#%%
svm_modle = clf.fit(x_train, y_train)
train_model = svm_modle.predict(x_train)
test_model = svm_modle.predict(x_test)
matchrate_in_train = sum(train_model == y_train)/len(y_train)
matchrate_in_test = sum(test_model == y_test) / len(y_test) 
print('matchrete in trian set : {:.6%}'.format(matchrate_in_train))
print('matchrete in test set : {:.6%}'.format(matchrate_in_test))

#%% [markdown]
#assignment 2
#%%
#use lable 1 and lable 2 to optimizing
x_train_1 = x_train[0:60]
y_train_1 = y_train[0:60]
x_test_1 = x_test[0:137]
y_test_1 = y_test[0:137]
x_test_1 = scalar.fit_transform(x_test_1)
x_train_1 = scalar.fit_transform(x_train_1)
w = cp.Variable((4096,1))
b = cp.Variable()
y_train_n = y_train_1.copy()
y_train_n = y_train_n.astype('int')
y_train_n[y_train_n == 0] = -1
y_train_n = y_train_n.reshape(60, 1)
y_test_n = y_test_1.copy()
y_test_n = y_test_n.astype('int')
y_test_n[y_test_n == 0] = -1
#%%
#### 此处填写优化问题的目标函数
obj = cp.Minimize(0.5 * cp.norm(w,2)** 2)
####
I = np.ones((60, 1),int)
#%%
### 此处填写优化问题的约束条件，如果有多个，以逗号隔开
constraint = [cp.multiply(y_train_n, x_train_1 @ w + b) >= I]

###

prob = cp.Problem(obj, constraint)
prob.solve()

print(prob.status)
ww = w.value
bb = b.value

#%%
####填写对训练数据和测试数据的测试代码，并测量分类准确率#
cvmodle_train = x_train_1 @ ww + bb
cvmodle_test = x_test_1 @ ww + bb
def mul_of_order(x, y):
    return x * y
f_train = np.array(list(map(mul_of_order, cvmodle_train, y_train_n)))
f_test = np.array(list(map(mul_of_order, cvmodle_test, y_test_n)))
matchrate1_in_train = 1 - sum(f_train < 0) / len(f_train)
matchrate1_in_test = 1 - sum(f_test < 0) / len(f_test)
print('matchrate in train set:{:6%}'.format(matchrate1_in_train[0]))
print('matchrate in test set:{:6%}'.format(matchrate1_in_test[0]))


# %% [markdown]
#assignment 3
#%%
x_train_2 = traindata[0][0][0].transpose().copy()
y_train_2 = traindata[0][0][1].ravel().copy()
x_test_2= testdata[0][0][0].transpose().copy()
y_test_2 = testdata[0][0][1].ravel().copy()
x_train_2 = scalar.fit_transform(x_train_2)
x_test_2 = scalar.fit_transform(x_test_2)
#%%
def rate_solve(array_p, modle_t, array_t,):
    a = np.array(modle_t.predict(array_p))
    b = np.array(array_t)
    r = sum(a == b) / len(b)
    return r
    
#%%
clf1 = svm.SVC(kernel='linear')
clf2 = svm.SVC(kernel='poly')
clf3 = svm.SVC(kernel='rbf')
modle1 = clf1.fit(x_train_2, y_train_2)
modle2 = clf2.fit(x_train_2, y_train_2)
modle3 = clf3.fit(x_train_2, y_train_2)
rate1 = rate_solve(x_test_2, modle1, y_test_2)
rate2 = rate_solve(x_test_2, modle2, y_test_2)
rate3 = rate_solve(x_test_2, modle3, y_test_2)

clf4 = svm.SVC(kernel='linear', C=1000000)
clf5 = svm.SVC(kernel='linear', C=1000)
clf6 = svm.SVC(kernel='linear', C=1)
clf7= svm.SVC(kernel='linear', C=0.0001)
clf8 = svm.SVC(kernel='linear', C=0.0000001)
modle4 = clf4.fit(x_train_2, y_train_2)
modle5 = clf5.fit(x_train_2, y_train_2)
modle6 = clf6.fit(x_train_2, y_train_2)
modle7 = clf7.fit(x_train_2, y_train_2)
modle8 = clf8.fit(x_train_2, y_train_2)
rate4 = rate_solve(x_test_2, modle4, y_test_2)
rate5 = rate_solve(x_test_2, modle5, y_test_2)
rate6 = rate_solve(x_test_2, modle6, y_test_2)
rate7 = rate_solve(x_test_2, modle7, y_test_2)
rate8 = rate_solve(x_test_2, modle8, y_test_2)

clf9 = svm.SVC(kernel='poly',degree=4)
clf10 = svm.SVC(kernel='poly',degree=5)
clf11 = svm.SVC(kernel='poly', degree=6)
modle9 = clf9.fit(x_train_2, y_train_2)
modle10 = clf10.fit(x_train_2, y_train_2)
modle11 = clf11.fit(x_train_2, y_train_2)
rate9 = rate_solve(x_test_2, modle9, y_test_2)
rate10 = rate_solve(x_test_2, modle10, y_test_2)
rate11 = rate_solve(x_test_2, modle11, y_test_2)

clf12 = svm.SVC(kernel='rbf', C=1,gamma=0.1)
clf13 = svm.SVC(kernel='rbf', C=1, gamma=0.001)
clf14 = svm.SVC(kernel='rbf', C=1, gamma=0.00000001)
modle12 = clf12.fit(x_train_2, y_train_2)
modle13 = clf13.fit(x_train_2, y_train_2)
modle14 = clf14.fit(x_train_2, y_train_2)
rate12 = rate_solve(x_test_2, modle12, y_test_2)
rate13 = rate_solve(x_test_2, modle13, y_test_2)
rate14 = rate_solve(x_test_2, modle14, y_test_2)
#%%
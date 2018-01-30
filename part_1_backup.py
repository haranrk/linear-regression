import numpy as np
from matplotlib import pyplot as plt
import random as rd
import pandas as pd

#Paramters
iter=10
alpha=10**-7
number_of_wts=2
data_size = 1000
grad_threshold=10

#Generate hypothesis function
x = np.arange(0,5,float(5)/data_size)
x_training = np.array(rd.sample(x,int(0.8*data_size)))
x_test = np.array([i for i in x if i not in x_training])
y_test = x_test**2
y_training = x_training**2
print(len(x_training),len(x_test))

#Weight Initialisation
w=np.ones(number_of_wts)
grad_w=np.full(number_of_wts,11)
grad_w_dash=np.zeros(number_of_wts)

X_training = np.zeros((len(x_training),number_of_wts))
X_test = np.zeros((len(x_test),number_of_wts))

for j in range(0,number_of_wts):

	for i,a in enumerate(x_training):
		X_training[i][j]=a**j

	for i,a in enumerate(x_test):
		X_test[i][j]=a**j

#Batch gradient descent
count = 1
while sum(abs(grad_w-grad_w_dash)>10)>0:
	print("Iteration %d"%(count))
	count+=1
	# plt.plot(x_training,y_training,'bo',x_training,np.dot(X_training,w.T),'ro')
	# plt.show()
	Y_training=np.dot(X_training,w.T)
	L=(y_training-Y_training)**2
	grad_w_dash=grad_w
	print(X_training.shape)
	grad_w=np.dot(-1*2*(y_training-Y_training).T,X_training)
	print(grad_w)
	print(w)
	w=w-alpha*grad_w

#Testing
Y_test=np.dot(X_test,w.T)
mean_squared_error=sum((y_test-np.dot(X_test,w.T))**2)/len(y_test)
print(mean_squared_error)
plt.plot(x_test,y_test,'bo',x_test,np.dot(X_test,w.T),'ro')
plt.show()
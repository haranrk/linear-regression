import numpy as np
from matplotlib import pyplot as plt
from functions import *

plt.ion()
plt.show()
#Paramters
alpha=10**-6
number_of_features=4
data_size = 10000
grad_threshold=10**-4

x,y= sample_hypothesis1(data_size,0)
X= create_features(x,number_of_features)
X_training,y_training,X_test,y_test= separate_datasets(X,y)

mean_squared_error=w={}
algo_list=[adagrad
# ,batch_grad
,stochastic_grad
,stochastic_grad_with_momentum
,rmsprop
]

for algo in algo_list:
	print(str(algo.__name__))
	w[algo] = initialize_weights(number_of_features)
	w[algo]=algo(X_training,y_training,alpha,w[algo],grad_threshold)
	mean_squared_error[algo]=calc_error(X_test,y_test,w[algo])
	print("Error: %f"%(mean_squared_error[algo]))
	plt.pause(5)	
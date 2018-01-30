import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from functions import *

alpha=10**-6
grad_threshold=10**-3

X,y,data_size,number_of_features=import_hypothesis('assignment1.csv',0)
X[:,1:]=normalise_dataset(X[:,1:])
X_training,y_training,X_test,y_test=separate_datasets(X,y,0.7)

mean_squared_error=w={}
algo_list=[rmsprop
# ,stochastic_grad
# ,stochastic_grad_with_momentum
# ,rmsprop
]
for algo in algo_list:
	print(str(algo.__name__))
	w[algo]=initialize_weights(number_of_features)
	w[algo]=algo(X_training,y_training,alpha,w[algo],grad_threshold,10**-4,0.9)
	print(w[algo])
	mean_squared_error[algo]=calc_error(X_test,y_test,w[algo])
	print("Error: %f"%(mean_squared_error[algo]))
	print(w[algo])


import numpy as np
from matplotlib import pyplot as plt
import random as rd
import pandas as pd
from functions import *
#Paramters
alpha=10**-6
# number_of_features=4
data_size = 1000
grad_threshold=0.0001
x,y=sample_hypothesis1(data_size,0)

for number_of_features in range(3,5):
	X=create_features(x,number_of_features)
	X_training,y_training,X_test,y_test=separate_datasets(X,y)
	w=initialize_weights(number_of_features)
	w=batch_grad(X_training,y_training,alpha,w,grad_threshold)
	plt.pause(3)
	mean_squared_error=calc_error(X_test,y_test,w)
	print("Error: %f"%(mean_squared_error))
	input()

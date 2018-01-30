import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from functions import *

#Code to generate a csv file
# data_size = 1000
# number_of_wts = 3
# x_training,x_test,y_training,y_test,x,y=f.sample_hypothesis(data_size)
# X_training,X_test,w=f.initialize_weights(number_of_wts,x_training,x_test)
# print(X_training[:,2],y_training)
# df=pd.DataFrame({'y': y_training, 'x1': X_training[:,0],'x2': X_training[:,1],'x3': X_training[:,2]})
# df.to_csv('sample_hypothesis.csv', index_label="Slno")

df=pd.read_csv('sample_hypothesis.csv')
print(df.shape)
number_of_wts=df.shape[1]-2
data_size=df.shape[0]

y_training=np.array(list(df.ix[:,4]))
print(y_training.shape)
X_training=np.ones((data_size,number_of_wts))
print(X_training.shape)
for i in range(1,number_of_wts+1):
	X_training[:,i-1]=df.ix[:,i]
print(X_training)
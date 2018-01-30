import numpy as np
from matplotlib import pyplot as plt
import random as rd
import pandas as pd
import inspect
from functions import *

#Weight Initialisation
def initialize_weights(number_of_features,val=10**-2):
	w=np.full(number_of_features,val)
	return w
	
#Extrapolates Polynomial Features from the given 2D function
def create_features(x,number_of_features):
	X = np.zeros((len(x),number_of_features))
	for j in range(0,number_of_features):
		for i,a in enumerate(x):
			X[i][j]=a**j
	return X

#Separate Dataset into training and test datasets
def separate_datasets(X,y,train_percent=0.8):
	train_indices = rd.sample(range(0,len(X)),int(train_percent*len(X)))
	test_indices = [i for i in range(0,len(X)) if i not in train_indices]
	x_training = np.ones((len(train_indices),X.shape[1]))
	x_test = np.ones((len(test_indices),X.shape[1]))
	y_test = np.ones(len(test_indices))
	y_training = np.ones(len(train_indices))
	for idx,i in enumerate(train_indices):
		x_training[idx,:]=X[i,:]
		y_training[idx]=y[i]

	for idx,i in enumerate(test_indices):
		x_test[idx,:]=X[i,:]
		y_test[idx]=y[i]

	return x_training,y_training,x_test,y_test

def batch_grad(X_training,y_training,alpha,w,grad_threshold):
	number_of_features=len(w)
	data_size=len(y_training)
	grad_w=np.full(number_of_features,11)
	L=np.array([0,1])
	count = 1
	while (abs(L[count]-L[count-1])>grad_threshold):
		Y_training=np.dot(X_training,w.T)
		grad_w=np.dot(-1*2*(y_training-Y_training).T,X_training)
		w=w-alpha*grad_w

		count+=1
		val=sum((y_training-np.dot(X_training,w.T))**2)/float(data_size)
		L=np.append(L,val)
		useful_plots(L,count,X_training,y_training,w,inspect.getframeinfo(inspect.currentframe())[2])
	return w

def stochastic_grad(X_training,y_training,alpha,w,grad_threshold):
	number_of_features=len(w)
	data_size=len(y_training)
	grad_w=np.full(number_of_features,11)
	L=np.array([0,1])
	count = 1
	while (abs(L[count]-L[count-1])>grad_threshold):
		for idx in range(0,data_size):
			Y_training=np.dot(X_training[idx,:],w.T)
			grad_w = np.dot(-1*2*(y_training[idx]-Y_training).T,X_training[idx,:])
			w=w-alpha*grad_w
		count+=1
		val=sum((y_training-np.dot(X_training,w.T))**2)/float(data_size)
		L=np.append(L,val)
		useful_plots(L,count,X_training,y_training,w,inspect.getframeinfo(inspect.currentframe())[2])
	write_plot(L,count,X_training,y_training,w,inspect.getframeinfo(inspect.currentframe())[2])	
	return w

def stochastic_grad_with_momentum(X_training,y_training,alpha,w,grad_threshold,eta=1):
	number_of_features=len(w)
	data_size=len(y_training)
	grad_w_dash=0
	grad_w=np.full(number_of_features,11)
	delta_w=0
	L=np.array([0,1])
	count = 1
	# for idx in range(1,10000):
	while (abs(L[count]-L[count-1])>grad_threshold):
		for idx in range(0,data_size):
			Y_training=np.dot(X_training[idx,:],w.T)
			grad_w = np.dot(-1*2*(y_training[idx]-Y_training).T,X_training[idx,:])
			delta_w=delta_w*eta-alpha*grad_w
			# print(delta_w*eta,alpha*grad_w)
			w=w + delta_w*eta
		count+=1
		val=sum((y_training-np.dot(X_training,w.T))**2)/float(data_size)
		L=np.append(L,val)
		useful_plots(L,count,X_training,y_training,w,inspect.getframeinfo(inspect.currentframe())[2])
	write_plot(L,count,X_training,y_training,w,inspect.getframeinfo(inspect.currentframe())[2])
	return w

def adagrad(X_training,y_training,alpha,w,grad_threshold,eta=1):
	number_of_features=len(w)
	data_size=len(y_training)
	ada_g=np.zeros(number_of_features)
	grad_w=np.zeros(number_of_features)
	
	L=np.array([0,1])
	count = 1
	# for idx in range(1,10000):
	while (abs(L[count]-L[count-1])>grad_threshold):
		for idx in range(0,data_size):
			Y_training=np.dot(X_training[idx,:],w.T)
			grad_w = np.dot(-1*2*(y_training[idx]-Y_training).T,X_training[idx,:])
			ada_g+=grad_w**2
			w = w - eta*grad_w/np.sqrt(ada_g)
		
		count+=1
		val=sum((y_training-np.dot(X_training,w.T))**2)/float(data_size)
		L=np.append(L,val)
		useful_plots(L,count,X_training,y_training,w,inspect.getframeinfo(inspect.currentframe())[2])
	write_plot(L,count,X_training,y_training,w,inspect.getframeinfo(inspect.currentframe())[2])
	return w

def rmsprop(X_training,y_training,alpha,w,grad_threshold,eta=10**-3,gamma=0.9):
	number_of_features=len(w)
	data_size=len(y_training)
	v=np.zeros(number_of_features)
	grad_w=np.zeros(number_of_features)
	
	L=np.array([0,1])
	count = 1
	# for idx in range(1,10000):
	while (abs(L[count]-L[count-1])>grad_threshold):
		for idx in range(0,data_size):
			Y_training=np.dot(X_training[idx,:],w.T)
			grad_w = np.dot(-1*2*(y_training[idx]-Y_training).T,X_training[idx,:])
			v=gamma*v+(1-gamma)*(grad_w**2)
			w = w - eta*grad_w/np.sqrt(v)
		count+=1
		val=sum((y_training-np.dot(X_training,w.T))**2)/float(data_size)
		L=np.append(L,val)
		# useful_plots(L,count,X_training,y_training,w,inspect.getframeinfo(inspect.currentframe())[2],1)
	write_plot(L,count,X_training,y_training,w,inspect.getframeinfo(inspect.currentframe())[2],1)
	return w

def useful_plots(L,count,X_training,y_training,w,algo_name,method=0):
	# print("Iteration %d"%(count))
	# print(L[count])
	plt.figure(1)
	plt.title(algo_name)
	plt.plot(range(0,len(L)),L)
	plt.xlabel("number of iterations")
	plt.ylabel("Error")
	if method==0:
		plt.figure(2)
		plt.clf()
		plt.plot(X_training[:,1],y_training,'ro',markersize=1,label="Actual")
		plt.plot(X_training[:,1],np.dot(X_training,w.T),'b<',markersize=1,label="Predicted")
		plt.legend(loc='upper right')
		plt.xlabel("x")
		plt.ylabel("f(x)")
		plt.title(algo_name)
		plt.draw()
		plt.pause(0.0001)

	elif method==1:
		for idx in [3,4]:
			plt.figure(2+idx)
			plt.clf()
			plt.plot(X_training[:,idx],y_training,'ro',markersize=1,label="Actual")
			plt.plot(X_training[:,idx],np.dot(X_training,w.T),'b<',markersize=1,label="Predicted")
			plt.legend(loc='upper right')
			plt.xlabel("x")
			plt.ylabel("f(x)")
			plt.title(algo_name)
			plt.draw()
		plt.pause(0.0001)

def write_plot(L,count,X_training,y_training,w,algo_name,method=0):
	plt.figure(1)
	plt.clf()
	plt.title(algo_name)
	plt.plot(range(0,len(L)),L)
	plt.xlabel("number of iterations | Total iterations = %f"%(len(L)))
	plt.ylabel("Loss | Final loss %f"%(L[-1]))
	plt.savefig('graphs/%s_convergence.jpg'%(algo_name))
	if method==0:	
		plt.figure(2)
		plt.clf()
		plt.plot(X_training[:,1],y_training,'ro',markersize=1,label="Actual")
		plt.plot(X_training[:,1],np.dot(X_training,w.T),'b<',markersize=1,label="Predicted")
		plt.legend(loc='upper right')
		plt.xlabel("x")
		plt.ylabel("f(x)")
		plt.title(algo_name)
		plt.savefig('graphs/%s_hypothesis.jpg'%(algo_name))
	elif method==1:
		for idx in range(1,X_training.shape[1]):
			plt.figure(1+idx)
			plt.clf()
			plt.plot(X_training[:,idx],y_training,'ro',markersize=1,label="Actual")
			plt.plot(X_training[:,idx],np.dot(X_training,w.T),'b<',markersize=1,label="Predicted")
			plt.legend(loc='upper right')
			plt.xlabel("x")
			plt.ylabel("f(x)")
			plt.title(algo_name)
			plt.savefig('graphs/%s_hypothesis_%d.jpg'%(algo_name,idx))

def sample_hypothesis1(data_size,noise=0):
	x = np.arange(0,2*np.pi,2*np.pi/data_size)
	y = np.sin(x) + 1
	if noise!=0:
		print("noise added")
		mean=np.std(y)
		for i in rd.sample(range(0,len(x)),int(len(x)*noise)):
			# y[i]+=mean*3*(i%2-0.5)
			y[i]=np.cos(x[i])
	return x,y

def sample_hypothesis2(data_size,noise=0):
	x = np.arange(0,5,float(5)/data_size)
	y = (x)**2
	if noise!=0:
		print("noise added")
		mean=np.std(y)
		for i in rd.sample(range(0,len(x)),int(len(x)*noise)):
			y[i]+=mean*3*(i%2-0.5)
	return x,y

def import_hypothesis(file_location,method=0):
	df=pd.read_csv(file_location)
	if method==0:
		print("Method Drop")
		df=df.dropna()
	elif method==1:
		print("Method interpolate")
		df=df.interpolate()
	elif method==2:
		print("Method fill mean")
		df=df.fillna(df.mean())
	data_size=df.shape[0]
	number_of_features = df.shape[1]-1
	data_size = df.shape[0]
	y=np.array(list(df.ix[:,1]))
	x = np.ones((data_size,number_of_features))

	for i in range(3,number_of_features+2):
		x[:,i-2]=df[df.columns[i-1]]
	return x,y,data_size,number_of_features
	
def normalise_dataset(X):
	for idx in range(0,X.shape[1]):
		X[:,idx]=(X[:,idx]-np.mean(X[:,idx]))/np.std(X[:,idx])
		# X[:,idx]=X[:,idx]/np.std(X[:,idx])
	return X	

def plot1(a,b,c,d):
	plt.plot(a,b,'ro',c,d,'bo')
	plt.show()

def plot0(a,b):
	plt.plot(a,b,'ro')
	plt.show()

def calc_error(X_test,y_test,w):
	return ((y_test-np.dot(X_test,w.T))**2).mean()
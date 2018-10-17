# -*- coding: utf-8 -*-
"""
This module defines a class implementing a simple autoencoder (currentplan is only 1 hidden layer). 
All the computation is designed to be run using only python 
"""
from activations import *
import random
class autoEncoder():
	'''this class implement a simplest autoencoder under '''
	def __init__(self,inputdf,num_hidden_layers=1,hidden_size = 10,varnames=None,options =None,batch_size = None,random_seed =0,mseweights = np.zeros(2)):
		'''imput data, format data, normalize data, define options
		@inputdf: pandas df include all columns of data
		@hidden_size: integer, number of hidden units
		@varnames: list of strings, indicating the variables that will be fit in the nn
		@options:
		@num_hidden_layers: model 
		'''
		np.random.seed(random_seed)
		self.input = inputdf if varnames == None else inputdf[varnames]
		self.varnames = varnames
		self.normalize()
		self.sample_size,self.input_size  = self.input.shape
		self.num_hidden_layers = num_hidden_layers
		self.hidden_size = hidden_size
		self.output_size = self.input_size
		self.num_layers = 1 + self.num_hidden_layers + 1 #input, output
		self.batch_size = batch_size        
		self.mseweights = mseweights if mseweights.sum() > 1 else np.ones(self.input_size)
		print("Initializing the class:")
		print("input data dimension is: %5d rows, %5d colums") % (self.sample_size,self.input_size)
		print("hidden layer number is %5d, hidden layer width is %5d, output size is %5d") %(self.num_hidden_layers,self.hidden_size,self.output_size)          
		self.output_mat = np.zeros(self.input_mat.shape)
		self.weights1  = np.random.rand(self.input_size,self.hidden_size) /np.sqrt(self.input_size)
		self.weights2  = np.random.rand(self.hidden_size,self.output_size) /np.sqrt(self.hidden_size)
		self.bias1 = np.zeros(self.hidden_size)    
		self.bias2 = np.zeros(self.output_size)     
	def normalize(self):
		from sklearn.preprocessing import StandardScaler,MinMaxScaler
# 		self.scaler = StandardScaler()
		self.input_mat = self.input.values
		for i in range(self.input.shape[1]):
			varname = self.input.columns[i]        
			if "value" in varname:
				self.input_mat[:,i] = np.log(np.abs(self.input_mat[:,i]) + 1) /(np.log(5)*10)  
			if "volume" in varname:
				self.input_mat[:,i] = np.log(np.abs(self.input_mat[:,i]) + 1)/(np.log(2)*10)  
# 			print(varname,self.input_mat[:,i].sum(),self.input.values[:,i].sum())                
# 		self.input_mat = self.scaler.fit_transform(self.input_mat)
	def forward_prop(self,batch_start =None,batch_end= None):#forward
		'''MLP part of the auto encoder calculation'''
		### forward propagation
		self.input_layer = self.input_mat[batch_start:batch_end] if (batch_start < batch_end) else self.input_mat #original input layer
		self.batch_size = batch_end - batch_start if (batch_start < batch_end) else self.sample_size
# 		print(sample_size)
# 		for layeri in range(1,self.num_layers):
# 			self.caches["z"+str(layeri)] = np.dot(self.caches["a"+str(layeri-1)],self.parameters["w"+str(layeri)]) + self.parameters["b"+str(layeri)]
		self.Z1 = np.dot(self.input_layer,self.weights1) + self.bias1
# 			self.caches["a"+str(layeri)] = ReLU(self.caches["z"+str(layeri)])
		self.hidden_layer = ReLU(self.Z1)
		self.Z2 = np.dot(self.hidden_layer,self.weights2) + self.bias2
		self.output_layer = ReLU(self.Z2)
		if (batch_start < batch_end):        
# 			print(batch_start,batch_end)
			self.output_mat[batch_start:batch_end] = self.output_layer
		else:
			self.output_mat = self.output_layer
		### backward propagation
	def backward_prop(self):#forward
# 		self.caches["dJda"+str(self.num_layers-1)] = self.output_layer - self.input_layer #m*p #multiplied -1 for gd algo
		self.dJdOut =  self.output_layer - self.input_layer #m*p #multiplied -1 for gd algo
# 		for layeri in reversed(range(1,self.num_layers)):
# 			self.caches["da"+str(layeri)+"dz"+str(layeri)] = ReLU_derivative(self.caches["z"+str(layeri)])
		self.dOutdZ2 = ReLU_derivative(self.Z2) #m*p da2dz2, out is a2
		#dZ2dw2 = self.hidden_layer
		self.dhiddendZ1 = ReLU_derivative(self.Z1)
		self.dJdb2 = np.multiply(self.dJdOut,self.dOutdZ2)
		self.dJdw2 = self.hidden_layer.T.dot(self.dJdb2)
		#djdA1
		self.dJdb1 = self.dJdb2.dot(self.weights2.T)
		self.dJdw1 = self.input_layer.T.dot(np.multiply(self.dJdb1,self.dhiddendZ1))
		self.dJdb2 = self.dJdb2.sum(axis =0)/self.batch_size
		self.dJdb1 = self.dJdb1.sum(axis =0)/self.batch_size
		self.dJdw2 /= self.batch_size
		self.dJdw1 /= self.batch_size
# 		print(self.output_mat.max())
		return None
	def lossCalculator(self):
		'''calculate current total loss no matter how many batches were updated'''     
		self.losslist = 0.5*np.square((self.input_mat - self.output_mat)).dot(self.mseweights)
		self.loss = self.losslist.sum()/self.sample_size #directly take total of the matrix
	def test(self,testdf):
		'''calculate current total loss no matter how many batches were updated'''        
		self.test_input = testdf[self.varnames].values
		for i in range(self.input.shape[1]):
			varname = self.input.columns[i]        
			if "value" in varname:
				self.test_input[:,i] = np.log(np.abs(self.test_input[:,i]) + 1) /(np.log(5)*10)  
			if "volume" in varname:
				self.test_input[:,i] = np.log(np.abs(self.test_input[:,i]) + 1)/(np.log(2)*10)  
		self.test_output = ReLU(np.dot((ReLU(np.dot(self.test_input,self.weights1) + self.bias1)),self.weights2) + self.bias2)
		self.test_losslist = np.square((self.test_output - self.test_input)).dot(self.mseweights)   
		self.test_loss = self.test_losslist.sum()/self.test_input.shape[0]  
	def train(self,learning_rate,num_epoch,batch_size,testdf,method = "gd",learning_ratemethod = "nodecay", a = 0.9, b = 0.99,printfreq =1):
		self.batch_size = batch_size        
		for iter_i in xrange(num_epoch):
# 			if iter_i %10 == 0:
			batchstart = 0
			self.lossCalculator()
			self.test(testdf)
			if iter_i % printfreq == 0:        
				print("at epoch "+str(iter_i)+", loss is "+str(self.loss))
				print("at epoch "+str(iter_i)+", the testing data loss is "+str(self.test_loss))
			while batchstart < self.sample_size:
# 				print("this batch start at "+ str(batchstart) + " end at" + str(batch_size+batchstart))
# 				print("at epoch "+ str(iter_i) + " batch " + str(batchstart) +", loss is "+ str(loss))
				self.forward_prop(batchstart,batchstart + batch_size)
				self.backward_prop()
				self.weights1 -= self.dJdw1 * learning_rate
				self.weights2 -= self.dJdw2 * learning_rate
				self.bias1 -= self.dJdb1 * learning_rate
				self.bias2 -= self.dJdb2 * learning_rate
				batchstart += batch_size

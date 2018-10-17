# -*- coding: utf-8 -*-
"""
This module defines a class implementing a simple autoencoder (currentplan is only 1 hidden layer). 
All the computation is designed to be run using only python 
Records of decisions:
	All parameters need to be processed as simple python data types, data will be processed with pyspark RDD or dataframe consider data size.
	Will try to incorporate local implement with pandas dataframe or numpy arrays after
"""
from activations import *
import random
class autoEncoder():
	'''this class implement a simplest autoencoder under '''
	def __init__(self,inputdf,num_hidden_layers=1,hidden_sizes = [10],varnames=None,options =None,batch_size = None,random_seed =0):
		'''imput data, format data, normalize data, define options
		@inputdf: pandas df include all columns of data
		@hidden_sizes: integer, number of hidden units
		@varnames: list of strings, indicating the variables that will be fit in the nn
		@options:
		@num_hidden_layers: model 
		'''
		np.random.seed(random_seed)
		self.input = inputdf if varnames == None else inputdf[varnames]
		self.input_mat = np.zeros((self.input.shape[1],self.input.shape[0]))
		self.varnames = varnames
		self.normalize()
		self.sample_size,self.input_size  = self.input.shape
		self.num_hidden_layers = num_hidden_layers
		if len(hidden_sizes) >= num_hidden_layers:
			hidden_sizes = hidden_sizes[:num_hidden_layers]
		else:
			raise InputError("Hidden layer size has to match the number of hidden layers")
		self.hidden_sizes = hidden_sizes
		self.output_size = self.input_size
		self.layers_sizes = [self.input_size] + hidden_sizes + [self.output_size]
		self.num_layers = 1 + self.num_hidden_layers + 1 #input, output
		self.batch_size = batch_size        
		print("Initializing the class:")
		print("input data dimension is: %5d rows, %5d colums") % (self.sample_size,self.input_size)
		print("hidden layer number is %5d, hidden layer width is %5d, output size is %5d") %(self.num_hidden_layers,self.hidden_sizes,self.output_size)          
		self.output_mat = np.zeros(self.input_mat.shape)
		self.caches = {}
		self.parameters = {}
		self.gradients = {}
		for layeri in range(1,self.num_layers):
			self.parameters["w"+str(layeri)]  = np.random.rand(self.layers_sizes[layeri-1],self.layers_sizes[layeri]) /np.sqrt(self.layers_sizes[layeri-1])
			# self.weights2  = np.random.rand(self.hidden_sizes,self.output_size) /np.sqrt(self.hidden_sizes)
			self.parameters["b"+str(layeri)]  =  np.zeros(self.layers_sizes[layeri]) 
			# self.bias1 = np.zeros(self.hidden_sizes)     
	def normalize(self):
		from sklearn.preprocessing import StandardScaler,MinMaxScaler
# 		self.scaler = StandardScaler()
		self.input_mat = self.input.values
		self.scaler = MinMaxScaler()
		for i in range(test.input.shape[1]):
			varname = test.input.columns[i]        
			if "value" in varname:
				self.input_mat[:,i] = np.log10(np.abs(self.input_mat[:,i]) + 1) /10  
# 			print(varname,self.input_mat[:,i].sum(),self.input.values[:,i].sum())                
		self.input_mat = self.scaler.fit_transform(self.input_mat)
	def autoencoder(self,batch_start =None,batch_end= None):#forward
		'''MLP part of the auto encoder calculation'''
		### forward propagation
		self.caches["a0"] = self.input_mat[batch_start:batch_end] if (batch_start <= batch_end) else self.input_mat #original input layer
		sample_size = batch_end - batch_start if (batch_start < batch_end) else self.sample_size
# 		print(sample_size)
		for layeri in range(1,self.num_layers):
			self.caches["z"+str(layeri)] = np.dot(self.caches["a"+str(layeri-1)],self.parameters["w"+str(layeri)]) + self.parameters["b"+str(layeri)]
			# self.Z1 = np.dot(self.input_layer,self.w1) + self.b1
			self.caches["a"+str(layeri)] = ReLU(self.caches["z"+str(layeri)])
			# self.hidden_layer = ReLU(self.Z1)
			# self.Z2 = np.dot(self.hidden_layer,self.w2) + self.b2
			# self.output_layer = ReLU(self.Z2)
		### backward propagation
		self.caches["dJda"+str(self.num_layers-1)] = self.output_layer - self.input_layer #m*p #multiplied -1 for gd algo, this is dJda2, or dJdOut
		# self.dJdOut =  self.output_layer - self.input_layer #m*p #multiplied -1 for gd algo
		for layeri in reversed(range(1,self.num_layers)):
			self.caches["da"+str(layeri)+"dz"+str(layeri)] = ReLU_derivative(self.caches["z"+str(layeri)])
			# self.dOutdZ2 = ReLU_derivative(self.Z2) #m*p da2dz2, out is a2
			#dZ2dw2 = self.hidden_layer
			# self.dhiddendZ1 = ReLU_derivative(self.Z1)

			#dZ2dw2 = self.hidden_layer
			self.gradients["dJdb"+str(layeri)] = np.multiply(self.caches["dJda"+str(layeri)],self.caches["da"+str(layeri)+"dz"+str(layeri)]) #the elementwise part in backprop
			self.dJdb2 = np.multiply(self.dJdOut,self.dOutdZ2)
			self.dJdw2 = self.hidden_layer.T.dot(self.dJdb2)
			#djdA1
			self.dJdb1 = self.dJdb2.dot(self.weights2.T)
			self.dJdw1 = self.input_layer.T.dot(np.multiply(self.dJdb1,self.dhiddendZ1))

			self.dJdb2 = self.dJdb2.sum(axis =0)/sample_size
			self.dJdb1 = self.dJdb1.sum(axis =0)/sample_size
			self.dJdw2 /= sample_size
			self.dJdw1 /= sample_size
		if (batch_start < batch_end):        
# 			print(batch_start,batch_end)
			self.output_mat[batch_start:batch_end] = self.output_layer
		else:
			self.output_mat = self.output_layer
# 		print(self.output_mat.max())
		return None
	def lossCalculator(self):
		'''calculate current total loss no matter how many batches were updated'''        
		loss = 0.5*np.square((self.input_mat - self.output_mat)).sum()/self.sample_size #directly take total of the matrix
		self.losslist = 0.5*np.square((self.input_mat - self.output_mat)).sum(axis=1)
		return loss
	def train(self,learning_rate,num_epoch,batch_size,method = "gd",learning_ratemethod = "nodecay", a = 0.9, b = 0.99):
		test.batch_size = batch_size        
		for iter_i in xrange(num_epoch):
# 			if iter_i %10 == 0:
			batchstart = 0
			loss = self.lossCalculator()
			if iter_i % 10 == 0:        
				print("at epoch "+str(iter_i)+", loss is "+str(loss))
			while batchstart < self.sample_size:
# 				print("this batch start at "+ str(batchstart) + " end at" + str(batch_size+batchstart))
# 				print("at epoch "+ str(iter_i) + " batch " + str(batchstart) +", loss is "+ str(loss))
				self.autoencoder(batchstart,batchstart + batch_size)
				self.weights1 -= self.dJdw1 * learning_rate
				self.weights2 -= self.dJdw2 * learning_rate
				self.bias1 -= self.dJdb1 * learning_rate
				self.bias2 -= self.dJdb2 * learning_rate
				batchstart += batch_size

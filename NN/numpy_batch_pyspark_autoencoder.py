# -*- coding: utf-8 -*-
"""
This module defines a class implementing a simple autoencoder (currentplan is only 1 hidden layer). 
Specifically to solve the issue that data is too large to fit into pandas/numpy at one time
All the computation is designed to be run using only python 
"""
from activations import *
import random
class autoEncodermixbatch():
	'''this class implement a simplest autoencoder under '''
	def __init__(self,inputdf,indexvar,batch_size=50000,testdf=None,dataprocessed=True,num_hidden_layers=1,hidden_size = 10,testing = True,varnames=None,options =None,random_seed =0,mseweights = np.zeros(2), removecsv = False):
		'''imput data, format data, normalize data, define options
		@inputdf: pandas df include all columns of data
		@hidden_size: integer, number of hidden units
		@varnames: list of strings, indicating the variables that will be fit in the nn
		@options:
		@num_hidden_layers: model 
		'''
		np.random.seed(random_seed)
		self.input = inputdf if varnames == None else inputdf.select(varnames +[indexvar])
		if testdf != None:
			self.test_input = testdf if varnames == None else testdf.select(varnames +[indexvar]) 
		self.indexvar = indexvar
		self.varnames = varnames
		# self.normalize()
		self.sample_size,self.input_size  = self.input.count(),len(self.input.columns)-1 #1 for the index column
		self.num_hidden_layers = num_hidden_layers
		self.hidden_size = hidden_size
		self.output_size = self.input_size
		# self.num_layers = 1 + self.num_hidden_layers + 1 #input, output 
		self.mseweights = mseweights if mseweights.sum() > 1 else np.ones(self.input_size)
		print("Initializing the class:")
		print("Input data dimension is: %5d rows, %5d colums") % (self.sample_size,self.input_size)
		print("Hidden layer number is %5d, hidden layer width is %5d, output size is %5d") %(self.num_hidden_layers,self.hidden_size,self.output_size)          
		# self.output_mat = np.zeros(self.input_mat.shape)
		self.batch_size = batch_size
		self.minibatch(dataset = self.input,alreadybatched=dataprocessed)
		#initialize first minibatch in case some initial check/test is necessary  
		self.input_pd = pd.read_csv(self.tempfile_prefix+str(0)+'.csv')
		self.input_layer = self.normalize(self.input_pd[self.varnames].values)
		self.test_inited = False
		if testing:
			#initialize the first batch test data
			self.test_inited = True
			self.test_inputpd = self.test_input.select(self.varnames +[self.indexvar]).limit(10000).cache().toPandas() # for now just test small portion of batch size, will redo this later
			self.testbatch_input = self.normalize(self.test_inputpd[self.varnames].values) 
		self.weights1  = np.random.rand(self.input_size,self.hidden_size) /np.sqrt(self.input_size)
		self.weights2  = np.random.rand(self.hidden_size,self.output_size) /np.sqrt(self.hidden_size)
		self.bias1 = np.zeros(self.hidden_size)    
		self.bias2 = np.zeros(self.output_size)  
		self.forward_prop();self.backward_prop()
		self.batchlossCalculator()
		print("loaded first random batch and calculated loss, which is ",self.batch_loss)
	def minibatch(self,dataset,batch_size =None,random_seed_batch = 0,tempfile_prefix = "MLP_input_",alreadybatched= False):
		self.tempfile_prefix = tempfile_prefix
		self.batch_size = batch_size if batch_size>0 else self.batch_size
		portion = float(self.batch_size) / self.sample_size
		self.num_batches = int(1.0/portion)
		if not alreadybatched:
			print("Each minibatch contains "+str(self.batch_size)+" observations")
			partition_list = dataset.randomSplit([1.0/self.num_batches] * self.num_batches, random_seed_batch)
			obsnum_check = []
			for par in range(self.num_batches):
				print("partitioning the "+str(par+1) +"th batch")
				obsnum_check.append(partition_list[par].count())
				partition_list[par].toPandas().to_csv(tempfile_prefix+str(par)+'.csv',encoding='utf8',index =False)
			assert(np.sum(obsnum_check) == self.sample_size)
		#load the first batch to calculate loss to get a general idea
	def normalize(self,npArray):
		from sklearn.preprocessing import StandardScaler,MinMaxScaler
# 		self.scaler = StandardScaler()
		temprarray = np.zeros(npArray.shape)
		for i in range(npArray.shape[1]):
			varname = self.varnames[i]
# 			print(varname,npArray[:10,i])
			if "value" in varname:
				temprarray[:,i] = np.log(np.abs(npArray[:,i]) + 1)/(np.log(2)*10)  #cannot use relative normalization since batch proc
			if "volume" in varname:
				temprarray[:,i] = np.abs(npArray[:,i])/100
# 			print(varname,self.input_mat[:,i].sum(),self.input.values[:,i].sum())                
# 		self.input_mat = self.scaler.fit_transform(self.input_mat)
		return temprarray
	def forward_prop(self):#forward
		'''
		forward propagation for the minibatch
		@self.input_layer -- normalized minibatch of data in numpy array format
		@self.output_layer -- minibatch output after forward propagation
		'''
# 		print(sample_size)
# 		for layeri in range(1,self.num_layers):
# 			self.caches["z"+str(layeri)] = np.dot(self.caches["a"+str(layeri-1)],self.parameters["w"+str(layeri)]) + self.parameters["b"+str(layeri)]
		self.Z1 = np.dot(self.input_layer,self.weights1) + self.bias1
# 			self.caches["a"+str(layeri)] = ReLU(self.caches["z"+str(layeri)])
		self.hidden_layer = ReLU(self.Z1)
		self.Z2 = np.dot(self.hidden_layer,self.weights2) + self.bias2
		self.output_layer = ReLU(self.Z2)
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
	def batchlossCalculator(self):
		'''calculate current total loss no matter how many batches were updated'''     
		self.batch_losslist = 0.5*np.square((self.input_layer - self.output_layer)).dot(self.mseweights)
		self.batch_loss = self.batch_losslist.sum()/self.batch_size #directly take total of the matrix
		self.batch_losslist = pd.concat([self.input_pd[self.indexvar],pd.DataFrame(self.batch_losslist,columns = ['reconstruction_error'])],axis = 1)
	def train(self,learning_rate,num_epoch,testdf,method = "gd",learning_ratemethod = "nodecay", printfreq =1, changed_batchsize = 0,testing = True):
		self.tracking_train = []
		self.tracking_test = []        
		for iter_i in xrange(num_epoch):
# 			if iter_i %10 == 0:
			self.loss = 0 
			# self.test_loss =0
			self.loss_list = pd.DataFrame(columns=[self.indexvar,'reconstruction_error'])
			# self.testloss_list = pd.DataFrame(columns=[self.indexvar,'reconstruction_error'])
			for par in range(self.num_batches):
# 				print("this batch start at "+ str(batchstart) + " end at" + str(batch_size+batchstart))
# 				print("at epoch "+ str(iter_i) + " batch " + str(batchstart) +", loss is "+ str(loss))
				self.input_pd = pd.read_csv(self.tempfile_prefix+str(par)+'.csv')
				self.input_layer = self.normalize(self.input_pd[self.varnames].values)
				if iter_i + par ==0:
					print("finish loading the first batch!")
				self.forward_prop()
				self.backward_prop()
				if iter_i + par ==0:
					print("finish forward and backward propagate the first batch!")
				self.batchlossCalculator()
				self.loss += self.batch_loss
				if iter_i == num_epoch -1:
					# print("patching loss for batch" + str(par+1))                    
					self.loss_list = self.loss_list.append(self.batch_losslist)
				self.weights1 -= self.dJdw1 * learning_rate
				self.weights2 -= self.dJdw2 * learning_rate
				self.bias1 -= self.dJdb1 * learning_rate
				self.bias2 -= self.dJdb2 * learning_rate
			if iter_i % printfreq == 0:        
				print("at epoch "+str(iter_i+1)+", total loss is "+str(self.loss))
				self.tracking_train.append(self.loss)
				if testing and self.test_inited:
					self.batchtest()
					# self.test_loss += self.testbatch_loss
					print("test batch loss is "+str(self.testbatch_loss))
					self.tracking_test.append(self.testbatch_loss)
					# if iter_i == num_epoch -1:
						# self.testloss_list = self.testloss_list.append(self.testbatch_losslist)
	def batchtest(self,alldata= False):
		'''calculate current total loss no matter how many batches were updated'''        
		self.testbatch_output = ReLU(np.dot((ReLU(np.dot(self.testbatch_input,self.weights1) + self.bias1)),self.weights2) + self.bias2)
		self.testbatch_losslist = np.square((self.testbatch_output - self.testbatch_input)).dot(self.mseweights)   
		self.testbatch_loss = self.testbatch_losslist.sum()/self.batch_size
		self.testbatch_losslist = pd.concat([self.test_inputpd[self.indexvar],pd.DataFrame(self.testbatch_losslist,columns = ['reconstruction_error'])],axis = 1)
	def fulltest(self,dataprocessed =True,testtempfile_prefix = "test_MLP_input_",random_seed_batch=1):
		'''calculate current total loss no matter how many batches were updated for the testing data''' 
		self.test_loss =0
		self.testloss_list = pd.DataFrame(columns=[self.indexvar,'reconstruction_error'])
		portion = float(self.batch_size) / self.test_input.count()
		self.testnum_batches = int(1.0/portion)
		print("Each minibatch contains "+str(self.batch_size)+" observations")
		testpartition_list = self.test_input.randomSplit([1.0/self.testnum_batches] * self.testnum_batches, random_seed_batch)
		for par in range(self.testnum_batches):
			print("partitioning the "+str(par+1) +"th batch")
			self.test_inputpd = testpartition_list[par].toPandas()
			self.testbatch_input = self.normalize(self.test_inputpd[self.varnames].values) 
			self.batchtest()
			self.test_loss += self.testbatch_loss
			self.testloss_list = self.testloss_list.append(self.testbatch_losslist)

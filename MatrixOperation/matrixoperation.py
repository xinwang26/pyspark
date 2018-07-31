import numpy as np
import pandas as pd
from pyspark.mllib.linalg import DenseMatrix
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry,IndexedRowMatrix, RowMatrix, BlockMatrix, IndexedRow
from pyspark.sql import HiveContext
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import Vectors as MLLibVectors
from pyspark.mllib.linalg import SparseVector as MLLibSparseVector
from pyspark.ml.linalg import SparseVector as MLSparseVector

def MatrixTranspose(mat): #have some issues --1. will cause errors for some data, not sure reasons butreducing number of rows could help.
###2. the transpose sometimes return wrong result which seems due to parition issue -- repartion(1) sometimes fix it, 
#also pypsark change the order of rows after transposed coordinate matrix convert to row matrix
## this bug ref:https://stackoverflow.com/questions/34451253/converting-coordinatematrix-to-rowmatrix-doesnt-preserve-row-order
## use indexed matrix could partially fix this issue by reordering but this is too wierd
	'''
	transpose a row matrix -- to save space/memory use sparse vector when input is sparse vector
	:param mat: the input row matrix
	:return a transposed row matrix
	ref: https://stackoverflow.com/questions/47102378/transpose-a-rowmatrix-in-pyspark
	'''
	if isinstance(mat,IndexedRowMatrix):
		mat = mat.toRowMatrix()
	#this line will turn everythign to some dense matrix entries, try avoid using this function for efficiency
	transposed_mat = CoordinateMatrix(mat.rows.zipWithIndex().flatMap(lambda x: [MatrixEntry(x[1], j, v) for j, v in enumerate(x[0])]))
	transposed_mat = transposed_mat.transpose().toIndexedRowMatrix().rows.toDF().orderBy("index")
	# back to sparse first then convert to indexedrowmatrix
	transposed_mat = transposed_mat.rdd.map(lambda row: IndexedRow(row["index"],MLLibVectors.sparse(row["vector"].size,np.nonzero(row["vector"].values)[0],row["vector"].values[np.nonzero(row["vector"].values)])))
	return IndexedRowMatrix(transposed_mat)
def NpToDense(arr):
	'''
	turn numpy array to Pyspark Dense Matrix so that matrix multiplication could be done 
	:param arr: a numpy array
	'''
	nrows, ncols = arr.shape
	return DenseMatrix(nrows, ncols, arr.flatten(), 1)
def DFtoMatrix(df,quantvars = None):
	'''
	convert a numeric dataframe to a rowmatrix with sparse vector as basic units, won't be applicable to dataframe already having assembled vectors
	'''
	if quantvars == None:
		quantvars = df.columns[1:] #numpy allow string element, should be fine?
	df = VectorAssembler(inputCols=quantvars, outputCol="features").transform(df).select("features") #vector assembler turn it automatically to sparse matrix, so next line should be fine
	df = df.rdd.map(lambda row: MLLibVectors.sparse(row.features.size, row.features.indices, row.features.values))
	return RowMatrix(df)
def DFtoIndexedMatrix(df,quantvars,idcol):
	'''
	convert a numeric dataframe to a rowmatrix with sparse vector as basic units, won't be applicable to dataframe already having assembled vectors
	'''
	df = VectorAssembler(inputCols=quantvars, outputCol="features").transform(df).select([idcol,"features"]) #vector assembler turn it automatically to sparse matrix, so next line should be fine
	df = df.rdd.map(lambda row: IndexedRow(row[idcol],MLLibVectors.sparse(row.features.size, row.features.indices, row.features.values)))
	return IndexedRowMatrix(df)
def vectorDFtoIndexedMatrix(df,vecvar,idcol):
	'''
	applicable to dataframe already having assembled vectors
	'''
	df = df.rdd.map(lambda row: IndexedRow(row[idcol],MLLibVectors.sparse(row[vecvar].size, row[vecvar].indices, row[vecvar].values)))
	return IndexedRowMatrix(df)
def MatrixToDF(mat,varnames = None, postfix = ""):
	'''
	convert RowMatrix with sparse vector as rows to dataframe
	:param mat: the input RowMatrixs
	:varnames: list of strings indicating variable names
	:return a pyspark dataframe w/ or w/o variable names
	'''
	width = mat.numCols()
	#break down to dense vector given specific matrix
	if varnames != None:
		if len(varnames) == width:
			return mat.rows.map(lambda x: (x, )).toDF(["vector"]).rdd.map(lambda row:(row.vector.toArray().tolist())).toDF(varnames)
		else:
			raise ValueError("number of variable name not match number of variable")
	return mat.rows.map(lambda x: (x, )).toDF(["vector"+postfix])
def IndexedMatrixToDF(mat,idcol = None,varnames = None, postfix = ""):
	'''
	convert RowMatrix with sparse vector as rows to dataframe
	:param mat: the input RowMatrixs
	:varnames: list of strings indicating variable names
	:return a pyspark dataframe w/ or w/o variable names
	'''
	width = mat.numCols()
	idcol = idcol if idcol != None else "index" #by default toDF will turn out ['index','vector']
	#break down to dense vector given specific matrix
	if varnames != None:
		if len(varnames) == width:
			return mat.rows.map(lambda x: (x, )).toDF(["vector"]).rdd.map(lambda row:(row.vector.toArray().tolist())).toDF(varnames)
		else:
			raise ValueError("number of variable name not match number of variable")
	return mat.rows.toDF([idcol,"vector"+postfix])

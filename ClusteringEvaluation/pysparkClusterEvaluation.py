"""
This module defines a class calculating clustering or classficiation evaluation metrics. Current metrics include RMS_STD,Rsquared,CalinskiHarabazIndex,Silhouette
"""
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector as MLVector, Vectors as MLVectors
from pyspark.mllib.linalg import Vector as MLLibVector, Vectors as MLLibVectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.feature import PCA
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import HiveContext
sc = SparkContext.getOrCreate()
hive_context = HiveContext(sc)
sc.setLogLevel("ERROR")
from pyspark.sql.functions import countDistinct
class pypsarkCluster():
    print("test ver")
    '''
    -- This class is developed to calculate evaluation metrics of clustering results -- with input of ID, cluster label and quantitative features
    Basic ANOVA statistics of Sum of squared total(SST), Sum of squared of between cluster (SSB), Sum of squared of within cluster error (SSE) can be calculated by corresponding methods
    -- Input should at lease include the pyspark dataframe, quantitative features name list, column labels of (ID column,cluster label column)
    the pyspark dataframe should include the quantitative features as well as ID and cluster label column
    '''
    def __init__(self,df,varNames,collabels,normlized = False,subsample = False):
        self.varnames = varNames
        self.P = len(self.varnames)
        self.df = df.na.fill(0)
        self.N = df.count()
        self.idcol = collabels[0] #string
        self.clusterLabelCol = collabels[1]
    def getNumClusters(self,clusterlab=None):
        from pyspark.sql.functions import countDistinct
        if clusterlab == None:
            clusterlab = self.clusterLabelCol
        temp =self.df.select(clusterlab).agg(countDistinct(clusterlab).alias('NC')).collect()[0][0]
        return temp
    def normalizedDF(self):
        from pyspark.ml.feature import Normalizer
        assembler = VectorAssembler(inputCols=varNames,outputCol="features")
        normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=2) #p is order of norm
        pipeline = Pipeline(stages=[assembler, normalizer])
        self.df_norm = pipeline.fit(vecdf)
        # Normalize each Vector using $L^\infty$ norm.
        lInfNormData = normalizer.transform(dataFrame, {normalizer.p: float("inf")})
        return (0)
    def getDistribution(self,clusterlab=None,varnames=None):
        '''Show cluster distribution in number of observation or aggregated variables'''
        if clusterlab == None:
            clusterlab = self.clusterLabelCol
        freq_table = self.df.groupBy(clusterlab).count().toPandas()
        return freq_table
    def sampledDF(self,sample_rate = 0.02,small_definition = [0.05,25],clusterlab=None):
        '''
        perform stratified sampling, if cluster size smaller than N * fraction, then this cluster will be all selected
        define smallest 10% or clusters smaller than 30 obs as small clusteres 
        '''
        from pyspark.sql.functions import countDistinct
        if clusterlab == None:
            clusterlab = self.clusterLabelCol
        small_rate = small_definition[0]
        small_count = small_definition[1]
        df_freqtable = self.getDistribution().sort_values(['count',clusterlab])
        relative_smallclusters = df_freqtable.head(int(float(self.getNumClusters())*small_rate))
        absolute_smallclusters = df_freqtable[df_freqtable['count']<small_count]
        small_clusters = pd.concat([relative_smallclusters,absolute_smallclusters]).drop_duplicates().sort_values(['count',clusterlab])        
        large_clusters = df_freqtable[~df_freqtable.isin(small_clusters)].dropna()
        small_clusters['sample_rate'] = 1
        large_clusters['sample_rate'] = sample_rate
        fraction_dict = pd.concat([small_clusters,large_clusters]).drop_duplicates()\
                           .sort_values(['count',clusterlab])\
                           .drop('count',axis = 1).set_index(clusterlab).to_dict()['sample_rate']
        return(self.df.sampleBy(clusterlab,fraction_dict))
    def SST(self):
        '''return Sum of Squared Total, mainly for calculation verification'''
        from pyspark.sql.functions import countDistinct
        SST_query = [F.var_pop(varname).alias(varname+'_var_pop') for varname in self.varnames]
        SST_components = self.df.select(self.varnames).agg(*SST_query).toPandas() #vec
        SST = SST_components.as_matrix().sum() *self.N
        return SST,SST_components
    def SSE(self,clusterlab=None):
        '''return Sum of Squared within cluster Error (see cluster as model)'''
        if clusterlab == None:
            clusterlab = self.clusterLabelCol
        SSE_query = [F.var_pop(varname).alias(varname+'_var_cluster') for varname in self.varnames]  + \
                    [F.count(F.lit(1)).alias('count')]
        SSE_components = self.df.groupBy(clusterlab).agg(*SSE_query).toPandas()
        SSE_components1 = SSE_components[[varname+'_var_cluster' for varname in self.varnames]].multiply(SSE_components["count"], axis="index")
        SSE = SSE_components1.as_matrix().sum()
        return SSE,SSE_components
    def SSB(self,clusterlab=None):
        '''return Sum of Squared between clusters'''
        import numpy as np
        import pyspark.sql.functions as F
        if clusterlab == None:
            clusterlab = self.clusterLabelCol
        SSB_query = [F.avg(varname).alias(varname+'_avg_cluster') for varname in self.varnames]  + [F.count(F.lit(1)).alias('count')]
        SSB_components1 = self.df.groupBy(clusterlab).agg(*SSB_query).toPandas() #df
        SSB_components_mat = SSB_components1.drop([clusterlab,'count'],axis=1).as_matrix()
        weights_SSB = SSB_components1['count'].as_matrix() 
        SSB_avg = np.average(SSB_components_mat, weights=weights_SSB,axis = 0)
        # Fast and numerically precise:
        SSB_var = (np.average((SSB_components_mat-SSB_avg)**2, weights=weights_SSB,axis =0)*weights_SSB.sum()).sum()
        return SSB_var
    def CalinskiHarabazIndex(self,clusterlab=None):
        '''return the CalinskiHarabazIndex from SSB & SSE-- Calinski Harabaz Index equals a high dimension anova F statistics'''
        if clusterlab == None:
            clusterlab = self.clusterLabelCol
        SSE, _ = self.SSE(clusterlab)
        SSB = self.SSB(clusterlab )
        self.NC = self.getNumClusters(clusterlab)
        F_ratio = (SSB/(self.NC- 1))/(SSE/(self.N-self.NC))
        return F_ratio
    def Rsquared(self,clusterlab = None):
        '''retrurn R-squared -- R-squared is ratio of SSB and SST'''
        SSE, _ = self.SSE(clusterlab)
        SST, _ = self.SST()
        return ((SST-SSE)/SST)
    def RMS_STD(self,clusterlab = None):
        '''retrurn RMS-STD -- RMS-STD is degree of freedom /dimension adjusted MSE when seeing cluster as a model'''
        from numpy import sqrt
        SSE, _ = self.SSE(clusterlab)
        self.NC = self.getNumClusters(clusterlab)
        return sqrt(SSE/(self.P*(self.N-self.NC)))
    def pairwise_dist(self,df_work,distmethod = "Euclidean",filter_str = None,sampled = False):
        '''return pairwise distance of all pairs of IDs, serve methods calculating distance based evaluation metrics'''
        import numpy as np
        data = df_work
        if filter_str != None:
            data = data.filter(filter_str)
        data1 = (data).rdd.zipWithIndex().map(lambda (val, idx): (idx, val)) #move account id to first, create index and move to first
        idxs = sc.parallelize(range(data1.count()))
        indices = idxs.cartesian(idxs).filter(lambda x: x[0] < x[1])
        joined1 = indices.join(data1).map(lambda (i, (j, val)): (j, (i, val)))
        joined2 = joined1.join(data1).map(lambda (j, ((i, latlong1), latlong2)): ((latlong1[0],latlong2[0]), (latlong1[1:], latlong2[1:])))
        if distmethod == "Euclidean":
            pairwiseRDD = joined2.mapValues(lambda (x, y): np.linalg.norm(np.array(x)-np.array(y)))
        pairwise_mirror = pairwiseRDD.map(lambda ((ID1,ID2),dist): (int(ID2),int(ID1),float(dist))).toDF(["ID1","ID2",distmethod+"_distance"])
        pairwise_selfdf = pairwiseRDD.map(lambda ((ID1,ID2),dist): (int(ID1),int(ID2),float(dist))).toDF(["ID1","ID2",distmethod+"_distance"])
        pairwise_cartisan = pairwise_selfdf.union(pairwise_mirror)
        return pairwise_cartisan #, index_dict
    def Silhouette(self,clusterlab=None,sildistmethod = "Euclidean",silfilter_str = None,dbname ="risk"):
        '''return Silhouette index'''
        from pyspark.sql.functions import col, avg,greatest
        import pyspark.sql.functions as F
        if clusterlab == None:
            clusterlab = self.clusterLabelCol
        prwise_cart = self.pairwise_dist(self.df.select([self.idcol]+self.varnames),distmethod = sildistmethod,filter_str = silfilter_str)
        prwise_cart.createOrReplaceTempView("pairwise_dist")
        hive_context.sql("drop table if exists "+dbname+".pairwise_dist")
        hive_context.sql("create table pairwise_dist as select * from pairwise_dist")
        del prwise_cart
        prwise_cart = hive_context.table("pairwise_dist")
        ID_cluster_link = self.df.select([self.idcol,clusterlab])
        ID_cluster_link.createOrReplaceTempView("id_cluster")
        hive_context.sql("drop table if exists "+dbname+".id_cluster")
        hive_context.sql("create table id_cluster as select * from id_cluster")
        del ID_cluster_link
        ID_cluster_link = hive_context.table("id_cluster")
        #a big cartisan join for pairwise points, coumputation is N^2 mapping
        ID_pairwise_cart = prwise_cart.alias("dist").join(ID_cluster_link.alias("id1"),col("id1."+self.idcol)==col("dist.ID1"))\
                                            .join(ID_cluster_link.alias("id2"),col("id2."+self.idcol)==col("dist.ID2"))\
                                            .selectExpr("id1."+clusterlab+" as ID1_"+clusterlab,\
                                                        "id2."+clusterlab+" as ID2_"+clusterlab,\
                                                        "dist.*")
        #point i to other cluster's average's min        
        ID_pairwise_bi = ID_pairwise_cart.filter("ID1_" + clusterlab +" <> "+"ID2_" + clusterlab)\
                                        .groupBy("ID1_" + clusterlab,"ID1","ID2_" + clusterlab)\
                                        .agg(avg(sildistmethod+"_distance").alias("avg_distance_"+"ID2"+clusterlab))
        ID_pairwise_bi = ID_pairwise_bi.groupBy("ID1_" + clusterlab,"ID1").agg(F.min("avg_distance_"+"ID2"+clusterlab).alias("b_i"))
        #point i to self cluster's average
        ID_pairwise_ai = ID_pairwise_cart.filter("ID1_" + clusterlab +" = "+"ID2_" + clusterlab)\
                                        .groupBy("ID1_" + clusterlab,"ID1")\
                                        .agg(avg(sildistmethod+"_distance").alias("a_i"))
        #calculate the bi-ai / max(ai bi) formula       
        ID_pairwise_aibi = ID_pairwise_ai.alias("a").join(ID_pairwise_bi.alias("b"), ID_pairwise_ai.ID1 == ID_pairwise_bi.ID1)\
                                                    .selectExpr("a.*","b.b_i")
        #calculate silhouette for each data point
        ID_pairwise_aibi =ID_pairwise_aibi.withColumn("silouette", (ID_pairwise_aibi["b_i"]-ID_pairwise_aibi["a_i"])/greatest(ID_pairwise_aibi["a_i"],ID_pairwise_aibi["b_i"]))
        Silhouette = ID_pairwise_aibi.select("silouette").agg(avg("silouette")).collect()[0][0]
        hive_context.sql("drop table if exists "+dbname+".pairwise_dist")
        hive_context.sql("drop table if exists "+dbname+".id_cluster")
        return(Silhouette,ID_pairwise_aibi)

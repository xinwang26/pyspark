import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import sys
import cx_Oracle
import pandas as pd
from pyspark.sql.functions import col, sum, count, min, max, rank, desc, when
import pyspark.sql.functions as F
from pyspark.sql.functions import countDistinct as CD
# from pyspark_modeling import *
from pyspark.sql.types import IntegerType 
from pyspark.sql import HiveContext
from pyspark import SparkConf, SparkContext
now = datetime.datetime.now()


conf = (SparkConf()
         .setMaster("yarn-client")
         .setAppName("Xin_TXN_sup_step1")
         .set("spark.num.executors", "5")
         .set("spark.executor.cores","5")
         .set("spark.driver.cores","2")
         .set("spark.driver.memory", "5g")
         .set("spark.executor.memory", "5g")
         .set("spark.dynamicAllocation.enabled","true"))

sc = SparkContext.getOrCreate()
sc.stop()
sc = SparkContext(conf = conf)
hive_context = HiveContext(sc)
hive_context.sql("use risk") #pass sql sentence, will not really run untill any action taken
sc.setLogLevel("ERROR")

#connect SQL db to load alert data
username = 'SEC_BKT'
pwd ='secbkt93#41'
servname = 'vsam'
hoststr = 't53oraoda921t1a'
portnum = '1521'
dsn_tns = cx_Oracle.makedsn(hoststr, portnum, service_name=servname) #if needed, place an 'r' before any parameter in order to address any special character such as '\'.
conn = cx_Oracle.connect(user=username, password=pwd, dsn=dsn_tns) 
query = 'SELECT * FROM sec_bkt.alert_nov'
supervised_pd = pd.read_sql(query, con=conn)

#correct variables type by name
datevars = [colu for colu in supervised_pd.columns if 'DATE' in colu]
keyvars = [colu for colu in supervised_pd.columns if 'KEY' in colu]
supervised_pd[datevars]= supervised_pd[datevars].apply(pd.to_datetime, errors='coerce')
supervised_pd[keyvars]= supervised_pd[keyvars].astype(str)

#remove alerts generated after case
supervised_pd = supervised_pd.loc[~ (supervised_pd.ALERT_CREATE_DATE > supervised_pd.CASE_CREATE_DATE)]

#use alert data to filter txn data to reduce computation/memory
supervised_pd_filter = supervised_pd[['FINAL_ACCOUNT_KEY','ALERT_MONTH_SK']].drop_duplicates()
supervised_pd_filter.columns = ['account_key','alert_month_sk']
supervised_filter = hive_context.createDataFrame(supervised_pd_filter)

#generate monthly txn summary
sam_txn = hive_context.table('udm_cds_transactions1023').where("month_sk >=216 and month_sk <= 226").withColumn('abs_value',F.abs(col('acct_curr_amount')))
sam_acct = hive_context.table('udm_cds_account0822').where("is_error_account is null").dropDuplicates()
sam_txn_acctsmry = sam_txn.where("acct_curr_amount<>0").groupBy(["account_sk","month_sk"])\
                    .agg(F.sum('abs_value').alias('total_value'),CD('transaction_key').alias('total_volume')).alias('t')\
                    .join(sam_acct.alias('a'),col('t.account_sk')==col('a.entity_sk'),'left').selectExpr('a.account_key',"t.*").alias('t2')\
                    .join(supervised_filter.alias('s'),[col('t2.account_key')==col('s.account_key'),col("t2.month_sk") +1 == col("s.alert_month_sk")], "inner")\
                    .selectExpr("t2.*","s.alert_month_sk").distinct()
sam_txn_acctsmry_pd = sam_txn_acctsmry.toPandas()

#merge alert data and txn summary
supervised_pd_common = supervised_pd.rename(index=str,columns={"ACCOUNT_KEY": "FORT_ACCOUNT_KEY","FINAL_ACCOUNT_KEY":"account_key","ALERT_MONTH_SK":"alert_month_sk"})\
                                    .merge(sam_txn_acctsmry_pd, on = ['account_key','alert_month_sk'], how='inner')
supervised_pd_common.columns = [colu.lower() for colu in supervised_pd_common.columns]
supervised_pd_common.loc[supervised_pd_common.alert_rank.isnull(),'alert_rank'] = '-1'
supervised_pd_common['is_case_raw'] = supervised_pd_common.apply(lambda row: 1 if row['case_id'] else 0,axis =1)
supervised_pd_common['is_case_clean'] = supervised_pd_common\
                    .apply(lambda row: 1 if (row['case_id']!= None) & (row['alert_rank'] not in ['0','1','2','4']) else 0,axis =1)
supervised_pd_common['is_case_strict'] = supervised_pd_common\
                    .apply(lambda row: 1 if (row['case_id']!= None) & (row['alert_rank']  in ['3','5','6','7']) else 0,axis =1)

#EXT alert only keep the top valued account
#Customer SB only keep top 2 valued account based on its logic that second largest (top2) accounts over threshold:
supervised_pd_EXT = supervised_pd_common.loc[supervised_pd_common.alert_id.str[:3]=='EXT']
supervised_pd_EXT = supervised_pd_EXT.sort_values(['alert_id','alert_create_date','total_value','account_key'],ascending = [True,True,False,True])
maxvalue_idx  = (supervised_pd_EXT.groupby(['alert_id','alert_create_date'])['total_value'].transform(lambda x: x.max())== supervised_pd_EXT['total_value'])
supervised_pd_EXT_fixed = supervised_pd_EXT[maxvalue_idx]

supervised_pd_CUSTSB = supervised_pd_common.loc[supervised_pd_common.detail_alert_type=='Customer Security Blanket']
supervised_pd_CUSTSB['top2value'] = supervised_pd_CUSTSB.groupby(['alert_id','alert_create_date'])['total_value'].transform(lambda x: x.nlargest(2).min())
supervised_pd_CUSTSB_fixed = supervised_pd_CUSTSB.loc[supervised_pd_CUSTSB.total_value >= supervised_pd_CUSTSB.top2value]

supervised_pd_other = supervised_pd_common[~(supervised_pd_common.alert_id.isin(supervised_pd_CUSTSB.alert_id) | supervised_pd_common.alert_id.isin(supervised_pd_EXT.alert_id) )]
supervised_pd_eval = supervised_pd_other.append([supervised_pd_CUSTSB_fixed,supervised_pd_EXT_fixed])

supervised_pd_eval.to_csv("ATC_nov_fixed"+now.strftime("%Y%m%d")+".csv",index = False)

#build the data used for unsupervised evaluation
sup_pd_alert = pd.read_csv("ATC_nov_fixed"+now.strftime("%Y%m%d")+".csv")
datevars = [colu for colu in sup_pd_alert.columns if 'date' in colu]
keyvars = [colu for colu in sup_pd_alert.columns if 'key' in colu]
sup_pd_alert[datevars]= sup_pd_alert[datevars].apply(pd.to_datetime, errors='coerce')
sup_pd_alert[keyvars]= sup_pd_alert[keyvars].astype(str)
sup_pd_alert['is_SB'] = sup_pd_alert.detail_alert_type.isin(['Account Security Blanket','Customer Security Blanket']).astype(int)
sup_pd_alert['alert_rank'] = sup_pd_alert['alert_rank'].astype(int)
sup_pd_alert.sort_values(['account_key','alert_month_sk','alert_rank','is_case_strict','is_case_clean','is_case_raw','is_SB']\
                        ,ascending=[True,True,False,False,False,False,False],inplace = True)
sup_eval_pd = sup_pd_alert.groupby(['account_sk','account_key','alert_month_sk'])['is_case_strict','alert_rank','is_case_clean','is_case_raw','is_SB']\
                          .aggregate('max').reset_index()
sup_eval_pd.to_csv("alert_evaluate"+now.strftime("%Y%m%d")+".csv",index = False)

with open("updated_data_name.txt","w") as file:
    file.write("alert_evaluate"+now.strftime("%Y%m%d")+".csv")

def cat_to_vec(data,varlist,nuisance,assemble = False,target = "y"):
    '''
    data -- a spark data frame
    varlist -- nominal variable list
    nuisance -- numeric variables or other variables do not need 
    '''
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.feature import OneHotEncoder
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml import Pipeline
    proced_data = data
    varlist_fix = []
    for var in varlist:
        var_fix = var+"_id"
        indexer = StringIndexer(inputCol=var, outputCol=var+"_id")
        pipeline = Pipeline(stages=[indexer])
        
        if len(data.select(var).distinct().collect())>2:
            var_fix = var+"_vec"
            encoder = OneHotEncoder(inputCol=var+"_id", outputCol=var+"_vec")
            pipeline = Pipeline(stages=[indexer, encoder])
        varlist_fix.append(var_fix)
        proced_data = pipeline.fit(proced_data).transform(proced_data)
        #vector_indexer = VectorIndexer(inputCol="bank_vector", outputCol="bank_vec_indexed")
    model_data = proced_data.select(nuisance+varlist_fix+target)
    if assemble == True:
        assembler = VectorAssembler(inputCols= varlist_fix + nuisance, outputCol="features")
        model_data = assembler.transform(model_data)
    return proced_data,model_data,varlist_fix+nuisance

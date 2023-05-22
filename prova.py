from utility import data_preparation,document_cleaning,getVectorized

import numpy as np

from pyspark.sql import SparkSession

import itertools

import time

def sequential(X, corpus, keys, threshold):
    
    scores = X.dot( X.transpose() )

    def documents_pairs(scores: np.array, threshold:float):
    
        index__doc_id_map=dict(zip(range(0,len(corpus)),keys))
        
        np.fill_diagonal(scores, 0)
        
        pairs = np.argwhere(scores >= threshold)
        
        unique_pairs=set(tuple(sorted(p)) for p in pairs)
        
        return [((index__doc_id_map[index1],index__doc_id_map[index2]),scores[index1,index2]) for index1, index2 in unique_pairs]
    
    return documents_pairs(scores.toarray(), threshold)
    
def spark_(X, corpus, keys, threshold):
    
    # Create SparkSession 
    spark = SparkSession.builder.master("local[8]").appName("all-doc-pairs-similarity.com").config("spark.driver.memory", "10g").getOrCreate()
    sc = spark.sparkContext
    
    vectorized_docs=[]
    for index in range(0,len(corpus)):
        vectorized_docs.append(X.getrow(index))

    keys_rdd=sc.parallelize(keys)
    vectorized_docs_rdd=keys_rdd.zip(sc.parallelize(vectorized_docs)).persist()
    
    def Map(doc_pair):
    
        id, doc = doc_pair
        
        non_zero_terms=doc.nonzero()
        
        term__doc_ids=[]
        
        for term_index in non_zero_terms[1]:
            term__doc_ids.append((term_index,id))
        
        return term__doc_ids

    term_listDocIds_pairs_rdd=vectorized_docs_rdd.flatMap(Map).groupByKey()
    print("GROUP BY KEY DONE")
    
    def bho1(doc):
        
        return set(doc.nonzero()[1])
    
    doc_id_set_term=sc.broadcast(dict(vectorized_docs_rdd.mapValues(bho1).collect()))
    print("TERM SETS DONE")
    
    def bho(pair):
        
        term_id, id_list = pair
        
        pairs=[]
        
        for id1, id2 in itertools.combinations(id_list,2):
            
            if max(doc_id_set_term.value[id1] & doc_id_set_term.value[id2])==term_id:
                pairs.append((id1,id2))
        
        return pairs

    doc_ids_pairs_rdd=term_listDocIds_pairs_rdd.flatMap(bho)
    print("FILTER DONE")
    
    vectorized_docs_broadcast=sc.broadcast(dict(vectorized_docs_rdd.collect()))
    print("VECTORIZATION DONE")
    
    threshold=sc.broadcast(threshold)

    def Reduce(pair):
        
        id1,id2 = pair
        
        similarity=vectorized_docs_broadcast.value[id1].dot(vectorized_docs_broadcast.value[id2].transpose())
        
        return ((id1,id2),similarity[0,0])

    def similar_doc(pair):
        
        _, similarity = pair
        
        return similarity>=threshold.value

    similarity_doc_pairs=doc_ids_pairs_rdd.map(Reduce).filter(similar_doc)
    
    return similarity_doc_pairs.collect()
    

def main(threshold):
    
    corpus, keys = data_preparation("scifact")
    
    #corpus=corpus[0:500]
    #keys=keys[0:500]
    
    cleaned_corpus = document_cleaning(corpus)
    
    X=getVectorized(cleaned_corpus)
    
    start_spark=time.perf_counter()
    spark_list=spark_(X,corpus,keys,threshold)
    end_spark=time.perf_counter()
    
    start_seq=time.perf_counter()
    seq_list=sequential(X,corpus,keys,threshold)
    end_seq=time.perf_counter()
    
    missing_seq=set(spark_list)-set(seq_list)
    print(missing_seq)

    missing_spark=set(seq_list)-set(spark_list)
    print(missing_spark)
    
    print("spark: ",end_spark-start_spark)
    print("seq: ",end_seq-start_seq)
    
if __name__ == '__main__':
    # This code won't run if this file is imported.
    main(0.3)
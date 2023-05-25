from utility import data_preparation

import numpy as np

from pyspark.sql import SparkSession

import itertools

import time

import pandas as pd

def sequential(doc_matrix, keys, threshold):
    
    scores = doc_matrix.dot( doc_matrix.transpose() )

    def documents_pairs(scores: np.array, threshold:float):
    
        index__doc_id_map=dict(zip(range(0, doc_matrix.get_shape()[0]),keys))
        
        np.fill_diagonal(scores, -1)
        
        pairs = np.argwhere(scores >= threshold)
        
        unique_pairs=set(tuple(sorted(p)) for p in pairs)
        
        return [((index__doc_id_map[index1],index__doc_id_map[index2]),scores[index1,index2]) for index1, index2 in unique_pairs]
    
    return documents_pairs(scores.toarray(), threshold)
    
def spark_(vectorized_docs, keys, threshold, n_workers=8, n_slices=5):
    
    # Create SparkSession 
    spark = SparkSession.builder.master("local["+str(n_workers)+"]").appName("all-doc-pairs-similarity.com").config("spark.driver.memory", "10g").getOrCreate()
    sc = spark.sparkContext
    
    threshold=sc.broadcast(threshold)

    keys_rdd=sc.parallelize(keys, n_workers*n_slices)
    vectorized_docs_rdd=keys_rdd.zip(sc.parallelize(vectorized_docs, n_workers*n_slices)).persist()
    
    def compute_d_star(doc1, doc2):
        
        return doc1.maximum(doc2)
    
    d_star=sc.broadcast(vectorized_docs_rdd.values().reduce(compute_d_star))
    
    terms=sc.broadcast(np.array(range(0,vectorized_docs[0].get_shape()[1])))
    
    def Map(doc_pair):
    
        id, doc = doc_pair
        
        sorted_indexes=np.argsort(doc.toarray()[0])[::-1]
        
        sorted_doc=doc[:,sorted_indexes]
        
        mapping=dict(zip(sorted_indexes,terms.value))
        
        dot_prod=0
        
        term_iterator=iter(sorted_indexes)
        current=next(term_iterator,None)
        prev=None
        
        while(dot_prod<threshold.value):
            
            if current is None:
                return []
            
            if (sorted_doc[0,current]==0):
                break
            
            prev=current
            
            dot_prod+=(sorted_doc[0,current])*(d_star.value[:,sorted_indexes][0,current])
            current=next(term_iterator,None)
        
        term__doc_ids=[]
        
        if prev is not None:
            term__doc_ids.append((mapping[prev],id))
        
        while(current is not None):
            term__doc_ids.append((mapping[current],id))
            current=next(term_iterator,None)
        
        return term__doc_ids

    term_listDocIds_pairs_rdd=vectorized_docs_rdd.flatMap(Map).groupByKey()
    
    def term_set(doc):
        
        return set(doc.nonzero()[1])
    
    doc_id_set_term=sc.broadcast(dict(vectorized_docs_rdd.mapValues(term_set).collect()))
    
    def filter_pairs(pair):
        
        term_id, id_list = pair
        
        pairs=[]
        
        for id1, id2 in itertools.combinations(id_list,2):
            
            common_terms=doc_id_set_term.value[id1] & doc_id_set_term.value[id2]
            
            if len(common_terms)!=0 and max(common_terms)==term_id:
                pairs.append((id1,id2))

        return pairs

    doc_ids_pairs_rdd=term_listDocIds_pairs_rdd.flatMap(filter_pairs)
    
    vectorized_docs_broadcast=sc.broadcast(dict(vectorized_docs_rdd.collect()))

    def Reduce(pair):
        
        id1,id2 = pair
        
        similarity=vectorized_docs_broadcast.value[id1].dot(vectorized_docs_broadcast.value[id2].transpose())
        
        return ((id1,id2),similarity[0,0])

    def similar_doc(pair):
        
        _, similarity = pair
        
        return similarity >= threshold.value

    similarity_doc_pairs=doc_ids_pairs_rdd.map(Reduce).filter(similar_doc)
    
    return similarity_doc_pairs.collect()
    
def comparison(keys, doc_matrix, vectorized_docs, threshold:float, n_workers, n_slices):
    
    start_spark=time.perf_counter()
    spark_list=spark_(vectorized_docs, keys, threshold, n_workers, n_slices)
    end_spark=time.perf_counter()
    
    start_seq=time.perf_counter()
    seq_list=sequential(doc_matrix, keys, threshold)
    end_seq=time.perf_counter()

    print("\nspark result len:",len(spark_list),"seq result len:",len(seq_list))
    
    missing_spark=set(seq_list)-set(spark_list)
    print("spark missing pairs: ", missing_spark)
    
    spark_elapsed=end_spark-start_spark
    seq_elapsed=end_seq-start_seq
    
    print("\nspark time: ",spark_elapsed)
    print("seq time: ",seq_elapsed)
    
    return spark_elapsed, seq_elapsed

def test():
    
    dataset = "scifact"
    
    thresholds = [0.2, 0.5]
    
    data = {
            "threshold":[],
            "n_workers":[],
            "n_slices":[],
            "spark_time":[],
            "seq_time":[]
    }
        
    keys, doc_matrix, vectorized_docs = data_preparation(dataset,200)
    
    for threshold in thresholds:
        
        for n_workers in [2, 4, 6, 8, 12, 16]:
            
            for n_slices in [1,4]:
                
                print("\n\nTEST FOR:")
                print("threshold: ",threshold)
                print("n_workers: ",n_workers)
                print("n_slices\n: ",n_slices)
                
                spark_time, seq_time = comparison(keys, doc_matrix, vectorized_docs, threshold, n_workers, n_slices)
                
                data["threshold"].append(threshold)
                data["n_workers"].append(n_workers)
                data["n_slices"].append(n_slices)
                data["spark_time"].append(spark_time)
                data["seq_time"].append(seq_time)
    
    data=pd.DataFrame.from_dict(data)
    data.to_parquet("data.parquet")
        
if __name__ == '__main__':
    test()
    #keys, doc_matrix, vectorized_docs = data_preparation("scifact",200)
    #comparison(keys, doc_matrix, vectorized_docs, 0.3, 8, 1)
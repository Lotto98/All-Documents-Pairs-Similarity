from utility import data_preparation

import numpy as np

import findspark
findspark.init()

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

import itertools

import time
import multiprocessing

import pandas as pd

def sequential(doc_matrix, keys, threshold):
    
    start_seq=time.perf_counter()
    
    scores = doc_matrix.dot( doc_matrix.transpose() )

    def documents_pairs(scores: np.array, threshold:float):
    
        index__doc_id_map=dict(zip(range(0, doc_matrix.get_shape()[0]),keys))
        
        np.fill_diagonal(scores, -1)
        
        pairs = np.argwhere(scores >= threshold)
        
        unique_pairs=set(tuple(sorted(p)) for p in pairs)
        
        return [((index__doc_id_map[index1],index__doc_id_map[index2]),scores[index1,index2]) for index1, index2 in unique_pairs]
    
    to_return=documents_pairs(scores.toarray(), threshold)
    
    stop_seq=time.perf_counter()
    
    return to_return, (stop_seq-start_seq)
    
def spark_(doc_matrix:np.array, keys, threshold_broadcast, n_workers=8, n_slices=5):
    
    # Create SparkSession 
    spark = SparkSession\
    .builder\
    .config(conf = SparkConf().setMaster(f"local[{n_workers}]") \
        .setAppName("all_pairs_docs_similarity.com") \
        .set("spark.executor.memory", "10g") \
        .set("spark.executor.cores", "1") \
        .set("spark.driver.memory", "10g"))\
    .getOrCreate()

    sc = spark.sparkContext # Get sparkContextxt
    
    term_freq = np.sum(doc_matrix>0, axis=0)
    sorted_terms_indexes = np.argsort(term_freq)[::-1]
    doc_matrix=np.array([row[sorted_terms_indexes] for row in doc_matrix])
    
    start_spark=time.perf_counter()
    
    threshold_broadcast=sc.broadcast(threshold_broadcast)
    
    keys_rdd=sc.parallelize(keys, n_workers*n_slices)
    vectorized_docs_rdd=keys_rdd.zip(sc.parallelize(doc_matrix, n_workers*n_slices)).persist()
    
    def compute_d_star(doc1:np.array, doc2:np.array):
        
        return np.maximum(doc1,doc2)
    
    d_star_broadcast=sc.broadcast(vectorized_docs_rdd.values().reduce(compute_d_star))
    
    def Map(doc_pair):
    
        id, doc = doc_pair
        
        dot_prod=0
        
        term=0
        
        while(dot_prod<threshold_broadcast.value):
            
            if term >= doc.shape[0]: 
                return []
            
            dot_prod+=((doc[term])*(d_star_broadcast.value[term]))
            
            term+=1
        
        term__doc_ids=[]
        
        non_zeros=np.nonzero(doc)[0] #se il termine ha score 0 non lo appendo perche' non sara' mai un max dell'intersezione
        non_zeros= non_zeros[non_zeros>=(term-1)]
        
        for term in non_zeros:
            term__doc_ids.append((term,id))
        
        return term__doc_ids

    term_listDocIds_pairs_rdd=vectorized_docs_rdd.flatMap(Map).groupByKey()
    
    def term_set(doc):
        
        return np.nonzero(doc)[0]
    
    doc_id_set_term_broadcast=sc.broadcast(dict(vectorized_docs_rdd.mapValues(term_set).collect()))
    
    def filter_pairs(pair):
        
        term_id, id_list = pair
        
        pairs=[]
        
        for id1, id2 in itertools.combinations(id_list,2):
            
            common_terms=np.intersect1d(doc_id_set_term_broadcast.value[id1],
                                        doc_id_set_term_broadcast.value[id2],
                                        assume_unique=True)
            
            if len(common_terms)!=0 and max(common_terms)==term_id:
                pairs.append((id1,id2))

        return pairs

    doc_ids_pairs_rdd=term_listDocIds_pairs_rdd.flatMap(filter_pairs)
    
    vectorized_docs_broadcast=sc.broadcast(dict(vectorized_docs_rdd.collect()))

    def Reduce(pair):
        
        id1,id2 = pair
            
        similarity=np.dot(vectorized_docs_broadcast.value[id1], vectorized_docs_broadcast.value[id2].transpose())
        
        return ((id1,id2),similarity)

    def similar_doc(pair):
        
        _, similarity = pair
        
        return similarity >= threshold_broadcast.value

    similarity_doc_pairs=doc_ids_pairs_rdd.map(Reduce).filter(similar_doc)
    
    to_return = similarity_doc_pairs.collect()
    
    end_spark=time.perf_counter()
    
    return to_return, (end_spark-start_spark)
    
def comparison(keys, doc_matrix, threshold:float, n_workers, n_slices):
    
    spark_list, spark_elapsed=spark_(doc_matrix.toarray(), keys, threshold, n_workers, n_slices)
    
    seq_list, seq_elapsed=sequential(doc_matrix, keys, threshold)

    print("\nspark result len:",len(spark_list),"seq result len:",len(seq_list))
    
    missing_spark=set(dict(seq_list).keys())-set(dict(spark_list).keys())
    print("spark missing pairs: ", missing_spark, len(missing_spark))
    
    print("\nspark time: ",spark_elapsed)
    print("seq time: ",seq_elapsed)
    
    return spark_elapsed, seq_elapsed

def test():
    
    dataset = "nfcorpus"
    
    data = {
            "threshold":[],
            "n_workers":[],
            "n_slices":[],
            "spark_time":[],
            "seq_time":[]
    }
        
    keys, doc_matrix = data_preparation(dataset,1000)
    
    for threshold in [0.2, 0.4, 0.6]:
        
        for n_workers in [1, 2, 4, 6, 8, 12, 16]:
            
            for n_slices in [1, 3, 9]:
                
                print("\n\nTEST FOR:")
                print("threshold: ",threshold)
                print("n_workers: ",n_workers)
                print("n_slices: ",n_slices,"\n")
                
                with multiprocessing.Pool(1) as p:
                    spark_time, seq_time=p.apply(comparison, (keys, doc_matrix, threshold, n_workers, n_slices,))
                
                data["threshold"].append(threshold)
                data["n_workers"].append(n_workers)
                data["n_slices"].append(n_slices)
                data["spark_time"].append(spark_time)
                data["seq_time"].append(seq_time)
    
    data=pd.DataFrame.from_dict(data)
    data.to_parquet("data.parquet")
        
if __name__ == '__main__':
    test()
    #keys, doc_matrix = data_preparation("nfcorpus",750)
    #comparison(keys, doc_matrix, 0.7, 8, 4)
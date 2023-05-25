from utility import data_preparation

import numpy as np

from pyspark.sql import SparkSession

import itertools

import time

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
    
def spark_(vectorized_docs, keys, threshold, n_workers=8, n_slices=5):
    
    # Create SparkSession 
    spark = SparkSession.builder.master("local["+str(n_workers)+"]").appName("all-doc-pairs-similarity.com").config("spark.driver.memory", "10g").getOrCreate()
    sc = spark.sparkContext
    
    start_spark=time.perf_counter()
    
    threshold=sc.broadcast(threshold)
    
    """
    for doc in vectorized_docs:
        
        sorted_indexes=np.argsort(doc)[::-1]
        term_iterator=iter(sorted_indexes)
        
        current=next(term_iterator, None)
        
        while current is not None:
            print(doc[current])
            current=next(term_iterator, None)
        
        break
    return
    """

    keys_rdd=sc.parallelize(keys, n_workers*n_slices)
    vectorized_docs_rdd=keys_rdd.zip(sc.parallelize(vectorized_docs, n_workers*n_slices)).persist()
    
    def compute_d_star(doc1:np.array, doc2:np.array):
        
        return np.maximum(doc1,doc2)
    
    d_star=sc.broadcast(vectorized_docs_rdd.values().reduce(compute_d_star))
    
    #terms=sc.broadcast(np.array(range(0,vectorized_docs[0].get_shape()[1])))
    
    def Map(doc_pair):
    
        id, doc = doc_pair
        
        sorted_indexes = np.argsort(doc)[::-1]
        
        sorted_doc = doc[sorted_indexes]
        
        dot_prod=0
        
        term=0
        
        while(dot_prod<threshold.value):
            
            #se il termine corrente e' 0 allora tutti i termini che eplorero' dopo saranno 0 
            #e non superero' mai la threshold
            if term >= doc.shape[0] or (sorted_doc[term]==0): 
                return []
            
            dot_prod+=((sorted_doc[term])*(d_star.value[sorted_indexes][term]))
            
            term+=1
        
        term__doc_ids=[]
        
        non_zeros=np.nonzero(sorted_doc)[0] #se il termine ha score 0 non lo appendo perche' non sara' mai un max dell'intersezione
        non_zeros= non_zeros[non_zeros>=(term-1)]
        
        for term in non_zeros:
            term__doc_ids.append((term,id))
        
        return term__doc_ids

    term_listDocIds_pairs_rdd=vectorized_docs_rdd.flatMap(Map).groupByKey()
    
    print(list(map(Map,zip(keys,vectorized_docs))))
    
    def term_set(doc):
        
        return np.nonzero(doc)[0]
    
    doc_id_set_term=sc.broadcast(dict(vectorized_docs_rdd.mapValues(term_set).collect()))
    
    def filter_pairs(pair):
        
        term_id, id_list = pair
        
        pairs=[]
        
        for id1, id2 in itertools.combinations(id_list,2):
            common_terms=np.intersect1d(doc_id_set_term.value[id1],
                                        doc_id_set_term.value[id2],
                                        assume_unique=True)
            
            if (id1=='5836' and id2=='2014909') or (id2=='5836' and id1=='2014909'):
                print(term_id,"max: ",max(common_terms))
            
            #if len(common_terms)!=0 and max(common_terms)==term_id:
            pairs.append((id1,id2))

        return pairs

    doc_ids_pairs_rdd=term_listDocIds_pairs_rdd.flatMap(filter_pairs)
    
    vectorized_docs_broadcast=sc.broadcast(dict(vectorized_docs_rdd.collect()))

    def Reduce(pair):
        
        id1,id2 = pair
            
        
        similarity=np.dot(vectorized_docs_broadcast.value[id1], vectorized_docs_broadcast.value[id2].transpose())
        
        if (id1=='5836' and id2=='2014909') or (id2=='5836' and id1=='2014909'):
            print(similarity)
        
        return ((id1,id2),similarity)

    def similar_doc(pair):
        
        _, similarity = pair
        
        return similarity >= threshold.value

    similarity_doc_pairs=doc_ids_pairs_rdd.map(Reduce).filter(similar_doc)
    
    to_return = similarity_doc_pairs.collect()
    
    end_spark=time.perf_counter()
    
    spark.stop()
    
    return to_return, (end_spark-start_spark)
    
def comparison(keys, doc_matrix, vectorized_docs, threshold:float, n_workers, n_slices):
    
    spark_list, spark_elapsed=spark_(vectorized_docs, keys, threshold, n_workers, n_slices)
    
    seq_list, seq_elapsed=sequential(doc_matrix, keys, threshold)

    print("\nspark result len:",len(spark_list),"seq result len:",len(seq_list))
    
    missing_spark=set(dict(seq_list).keys())-set(dict(spark_list).keys())
    print("spark missing pairs: ", missing_spark, len(missing_spark))
    
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
    #test()
    keys, doc_matrix, vectorized_docs = data_preparation("scifact",500)
    comparison(keys, doc_matrix, vectorized_docs, 0.01, 8, 1)
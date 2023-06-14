from utility import data_preparation

# spark
import findspark

findspark.init()

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

# spark algorithm
import itertools
import numpy as np

# test
import time
import multiprocessing
import pandas as pd

# typing
from typing import List, Tuple, Any
from scipy.sparse import csr_matrix


def sequential(doc_matrix: csr_matrix, keys: list, threshold: float) -> Tuple[List[Tuple[tuple, float]], float]:
    """
    Sequential "Documents All Pairs Similarity" algorithm optimized using numpy.

    Args:
        doc_matrix (csr_matrix): document matrix in csr format.
        keys (list): list of document ids.
        threshold (float): threshold.

    Returns:
        Tuple[ List[Tuple[tuple,float]], float ]:
            -documents id pairs along with their similarity (List[Tuple[tuple,float]])
            -execution time (float)
    """

    start_seq = time.perf_counter()

    # compute cosine sim
    scores = doc_matrix.dot(doc_matrix.transpose()).toarray()

    # exclude the pairs of the same document
    np.fill_diagonal(scores, -1)

    # take the indexes of document pairs which score is >= threshold
    pairs = np.argwhere(scores >= threshold)

    # take only unique pairs
    unique_pairs = set(tuple(sorted(p)) for p in pairs)

    # dict map of row matrix indexes to document keys
    index__doc_id_map = dict(zip(range(0, doc_matrix.get_shape()[0]), keys))

    # map the document indexes to document ids and their similarity
    to_return = [((index__doc_id_map[index1], index__doc_id_map[index2]), scores[index1, index2]) for index1, index2 in
                 unique_pairs]

    stop_seq = time.perf_counter()

    return to_return, (stop_seq - start_seq)


def spark_(doc_matrix: np.array, keys: list, threshold: float, n_workers: int = 8, n_slices: int = 1) -> Tuple[
    List[Tuple[tuple, float]], float]:
    """
    Spark "Documents All Pairs Similarity" algorithm.
    
    Args:
        doc_matrix (np.array): document matrix in numpy format.
        keys (list): list of document ids.
        threshold (float): threshold.
        n_workers (int, optional): number of worker to be feed to spark. Defaults to 8.
        n_slices (int, optional): number of data divisions for each worker. Defaults to 1.

    Returns:
        Tuple[ List[Tuple[tuple,float]], float ]:
            -documents id pairs along with their similarity (List[Tuple[tuple,float]])
            -execution time (float)
    """

    # Create SparkSession 
    spark = SparkSession \
        .builder \
        .config(conf=SparkConf().setMaster(f"local[{n_workers}]") \
                .setAppName("all_pairs_docs_similarity.com") \
                .set("spark.executor.memory", "10g") \
                .set("spark.executor.cores", "1") \
                .set("spark.driver.memory", "10g") \
                .set("spark.driver.maxResultSize", "10g")) \
        .getOrCreate()

    # Get sparkContextxt
    sc = spark.sparkContext

    # sort the document matrix by maximal term frequency in the entire corpus.
    term_freq = np.sum(doc_matrix > 0, axis=0)
    sorted_terms_indexes = np.argsort(term_freq)[::-1]
    doc_matrix = np.array([row[sorted_terms_indexes] for row in doc_matrix])

    start_spark = time.perf_counter()

    # 1) Zip each document id with its vectorial representation
    keys_rdd = sc.parallelize(keys, n_workers * n_slices)
    vectorized_docs_rdd = keys_rdd.zip(sc.parallelize(doc_matrix, n_workers * n_slices)).persist()

    # 2) broadcast variables to be used by spark

    def compute_d_star(doc1: np.array, doc2: np.array) -> np.array:
        """
        Function to compute d_star with spark.

        Args:
            doc1 (np.array)
            doc2 (np.array)

        Returns:
            np.array: array of element-wise maximum values of doc1 and doc2.
        """

        return np.maximum(doc1, doc2)

    def non_zero_terms(doc: np.array) -> np.array:
        """
        Function to find the non-zero element indexes for the given document with spark.
        
        Args:
            doc (np.array)

        Returns:
            np.array: array of indexes.
        """

        return np.nonzero(doc)[0]

    threshold_broadcast = sc.broadcast(threshold)

    # d_star broadcast computed by spark
    d_star_broadcast = sc.broadcast(vectorized_docs_rdd.values().reduce(compute_d_star))

    # vectorized document broadcast
    vectorized_docs_broadcast = sc.broadcast(dict(vectorized_docs_rdd.collect()))

    # non-zero term ids broadcast
    doc_id_set_term_broadcast = sc.broadcast(dict(vectorized_docs_rdd.mapValues(non_zero_terms).collect()))

    def MAP(doc_pair: Tuple[Any, np.array]) -> List[Tuple[int, Any]]:
        """
        MAP function for the "Documents All Pairs Similarity" algorithm.

        Args:
            doc_pair (Tuple[Any,np.array]):
                -document id (Any).
                -vectorized doc (np.array).

        Returns:
            List[Tuple[int,Any]]: list of (term id,doc_id)
        """

        # unpack document pair
        id, doc = doc_pair

        # loop variables
        dot_prod = 0
        term = 0

        # b(d) computation: at the end of this while loop b(d) = term-1.
        while (dot_prod < threshold_broadcast.value):

            # if there is no more terms then a empty list is returned:
            # this document is excluded.
            if term >= doc.shape[0]:
                return []

            dot_prod += ((doc[term]) * (d_star_broadcast.value[term]))

            term += 1

        term__doc_ids = []

        # if the term has TF-IDF=0 then it can not be part of an intersection, thus it is excluded.
        non_zeros = np.nonzero(doc)[0]

        # take only the term which are >= (term-1) (b(d)).
        non_zeros = non_zeros[non_zeros >= (term - 1)]

        # append for each term (term,id).
        for term in non_zeros:
            term__doc_ids.append((term, id))

        return term__doc_ids

    def filter_pairs(pair: Tuple[int, list]) -> List[Tuple[Any, Any]]:
        """
        First part of REDUCE function for the "Documents All Pairs Similarity" algorithm.
        Function to generate (doc_id, doc_id) pairs on which the similarity will be computed.
        
        Args:
            pair (Tuple[int,list]):
                -term id.
                -list of document ids.

        Returns:
            List[Tuple[Any,Any]]: list of (doc_id, doc_id) on which the similarity will be computed.
        """

        # unpack term_id, id_list.
        term_id, id_list = pair

        pairs = []

        # iterate over combinations of document ids: this ensures that the document ids pair is unique.
        for id1, id2 in itertools.combinations(id_list, 2):

            # set intersection for the non-zeros term of the two documents.
            common_terms = np.intersect1d(doc_id_set_term_broadcast.value[id1],
                                          doc_id_set_term_broadcast.value[id2],
                                          assume_unique=True)

            # if the max of the intersection is equal to the tem_id then append.
            if max(common_terms) == term_id:
                pairs.append((id1, id2))

        return pairs

    def compute_similarity(pair: Tuple[Any, Any]) -> Tuple[Tuple[Any, Any], float]:
        """
        Second part of REDUCE function for the "Documents All Pairs Similarity" algorithm. Similarity computation function.
        
        Args:
            pair (Tuple[Any,Any]): doc_ids pair.

        Returns:
            Tuple[Tuple[Any,Any],float]:
                -(doc_id, doc_id) (Tuple[Any,Any])
                -similarity (float)
        """

        id1, id2 = pair

        similarity = np.dot(vectorized_docs_broadcast.value[id1], vectorized_docs_broadcast.value[id2].transpose())

        return ((id1, id2), similarity)

    def similar_doc(pair: Tuple[Tuple[Any, Any], float]) -> bool:
        """
        Third part of REDUCE function for the "Documents All Pairs Similarity" algorithm.
        This function excludes pairs which similarity is < threshold. 
        
        Args:
            pair (Tuple[Tuple[Any,Any],float]):
                    -(doc_id, doc_id) (Tuple[Any,Any])
                    -similarity (float)

        Returns:
            bool:
                -True: take the pair.
                -False: remove the pair.
        """

        _, similarity = pair

        return similarity >= threshold_broadcast.value

    # 3) Compute with spark:
    #   1. MAP function using flatMap(MAP).
    #   2. Group by term id.
    #   3. REDUCE function using:
    #       1. flatMap(filter_pairs)
    #       2. map(compute_similarity)
    #       3. filter(similar_doc)
    similarity_doc_pairs = vectorized_docs_rdd \
        .flatMap(MAP) \
        .groupByKey() \
        .flatMap(filter_pairs) \
        .map(compute_similarity) \
        .filter(similar_doc)

    to_return = similarity_doc_pairs.collect()

    end_spark = time.perf_counter()

    return to_return, (end_spark - start_spark)


def comparison(keys: list, doc_matrix: csr_matrix, threshold: float, n_workers: int, n_slices: int) -> Tuple[
    float, float, set]:
    """_summary_

    Args:
        keys (list): list of document keys.
        doc_matrix (csr_matrix): document matrix in csr format.
        threshold (float): threshold.
        n_workers (int):  number of worker to be feed to spark.
        n_slices (int): number of data divisions for each worker.

    Returns:
        Tuple[float,float,set]:
            -spark execution time (float).
            -seq execution time (float).
            -set of missing spark document id pairs. (set)
    """

    # spark computation
    spark_list, spark_elapsed = spark_(doc_matrix.toarray(), keys, threshold, n_workers, n_slices)

    # seq computation
    seq_list, seq_elapsed = sequential(doc_matrix, keys, threshold)

    print("\nspark result len:", len(spark_list), "seq result len:", len(seq_list))

    # print missing spark doc_id pairs
    missing_spark = set(dict(seq_list).keys()) - set(dict(spark_list).keys())
    print("spark missing pairs: ", missing_spark, len(missing_spark))

    # print time
    print("\nspark time: ", spark_elapsed)
    print("seq time: ", seq_elapsed)

    return spark_elapsed, seq_elapsed, missing_spark


def test() -> None:
    dataset = "nfcorpus"

    data = {
        "threshold": [],
        "n_workers": [],
        "n_slices": [],
        "spark_time": [],
        "seq_time": [],
        "missing": []
    }

    keys, doc_matrix = data_preparation(dataset, 1000)

    for threshold in [0.2, 0.4, 0.6]:

        for n_workers in [1, 2, 4, 6, 8, 12, 16]:

            for n_slices in [1, 3, 9]:
                print("\n\nTEST FOR:")
                print("threshold: ", threshold)
                print("n_workers: ", n_workers)
                print("n_slices: ", n_slices, "\n")

                with multiprocessing.Pool(1) as p:
                    spark_time, seq_time, missing = p.apply(comparison,
                                                            (keys, doc_matrix, threshold, n_workers, n_slices,))

                data["threshold"].append(threshold)
                data["n_workers"].append(n_workers)
                data["n_slices"].append(n_slices)
                data["spark_time"].append(spark_time)
                data["seq_time"].append(seq_time)
                data["missing"].append(missing)

    data = pd.DataFrame.from_dict(data)
    data.to_parquet("data.parquet")


if __name__ == '__main__':
    # test()

    # example of spark computation
    keys, doc_matrix = data_preparation("scifact", None)
    _, time_ = spark_(doc_matrix.toarray(), keys, 0.3, 20, 1)

    print(time_)

from beir.datasets.data_loader import GenericDataLoader
from beir import util
import os
import pathlib

from typing import List, Tuple
import numpy as np
from scipy.sparse import csr_matrix, random

from multiprocessing import Pool
from tqdm.autonotebook import tqdm


from sklearn.feature_extraction.text import TfidfVectorizer

import spacy

def _data_download(dataset: str) -> Tuple[List[str],List]:
    
    #Download dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(os.path.abspath('')), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    
    #Retrieve documents
    documents, _, _ = GenericDataLoader(data_path).load(split="test")
    
    return [document["title"]+" "+document["text"] for document in documents.values()], list(documents.keys())

#load the spacy model for lemmatization 
_nlp = spacy.load("en_core_web_lg",disable=['parser','ner'])
    
def _cleaner(text: str) -> str:
        return " ".join([token.lemma_.lower() for token in _nlp(text) if not token.is_stop and not token.is_punct])

def _document_cleaning(corpus: List[str]) -> List[str]:
    with Pool() as p:
        cleaned_corpus=list(tqdm( p.imap(_cleaner, corpus), 
                                                total=len(corpus),
                                                desc="documents cleaning"))
    return cleaned_corpus

def _getVectorized(cleaned_corpus: List[str]) -> csr_matrix:
    
    return TfidfVectorizer().fit_transform(cleaned_corpus)

def data_preparation(dataset: str, limit:int=None):
    
    #corpus, keys = _data_download(dataset)
    
    #if limit is not None:
    #    corpus = corpus[0:limit]
    #    keys = keys[0:limit]
    
    #cleaned_corpus = _document_cleaning(corpus)
    
    #doc_matrix = _getVectorized(cleaned_corpus)
    
    doc_matrix=random(10,20,0.4,"csr",float,random_state=1)
    keys=[0,1,2,3,4,5,6,7,8,9]
    
    vectorized_docs=[]
    for index in range(0,10):
        vectorized_docs.append(doc_matrix.getrow(index).toarray()[0])
    
    return keys, doc_matrix, vectorized_docs

def preprocessing_spark(corpus, keys, sc):
    corpus_rdd = sc.parallelize(c=corpus)
    keys_rdd = sc.parallelize(keys)
    id_corpus_rdd = keys_rdd.zip(corpus_rdd)

    #load the spacy model for lemmatization 
    _nlp_shared = sc.broadcast(_nlp)

    #text cleaning and tokenization
    def text_cleaning_tokenization( text ):
        return " ".join(token.lemma_.lower() for token in _nlp_shared.value(text) if not token.is_stop and not token.is_punct)

    cleaned_docs_rdd = id_corpus_rdd.mapValues(text_cleaning_tokenization).persist()
    
    def create_vocabulary( v1, v2 ):
        return set(v1) | set(v2)

    def tokenizer(text:str):
        return text.split(" ")
    
    vocabulary = cleaned_docs_rdd.mapValues(tokenizer).values().reduce(create_vocabulary)
    
    vectorirer=sc.broadcast(TfidfVectorizer(vocabulary=vocabulary))

    def vectorize(doc):
        return vectorirer.value.fit_transform([doc])

    vectorized_docs_rdd=cleaned_docs_rdd.mapValues(vectorize).persist()
    
    return vectorized_docs_rdd
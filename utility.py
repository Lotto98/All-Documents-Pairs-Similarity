from beir.datasets.data_loader import GenericDataLoader
from beir import util

import os
import pathlib

from typing import List

from multiprocessing import Pool

from tqdm.notebook import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer

import spacy

def data_preparation(dataset: str) -> List[str]:
    """
    Download the given dataset from beir and transform it in a list of concatenated titles and texts.

    Args:
        dataset (str): dataset name.

    Returns:
        List[str]: corpus: each string represent is a document title concatenated with document text.
    """
    
    #Download dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(os.path.abspath('')), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    
    #Retrieve documents
    documents, _, _ = GenericDataLoader(data_path).load(split="test")
    
    return [document["title"]+" "+document["text"] for document in documents.values()], list(documents.keys())

#load the spacy model for lemmatization 
_nlp = spacy.load("en_core_web_lg",disable=['parser','ner'])
    
def cleaner(text: str ):
        return " ".join([token.lemma_.lower() for token in _nlp(text) if not token.is_stop and not token.is_punct])

def document_cleaning(corpus):
    with Pool() as p:
        cleaned_corpus=list(tqdm( p.imap(cleaner, corpus), 
                                                total=len(corpus),
                                                desc="documents cleaning"))
    return cleaned_corpus

def preprocessing_normal(corpus,keys,sc):
    
    cleaned_corpus=document_cleaning(corpus)
    
    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(cleaned_corpus)

    vectorized_docs=[]
    for index in range(0,len(corpus)):
        vectorized_docs.append(X.getrow(index))

    keys_rdd=sc.parallelize(keys)
    vectorized_docs_rdd=keys_rdd.zip(sc.parallelize(vectorized_docs))

    return vectorized_docs_rdd

def preprocessing_spark(corpus,keys,sc):
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
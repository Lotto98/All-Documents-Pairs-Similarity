# data download
from beir.datasets.data_loader import GenericDataLoader
from beir import util
import os
import pathlib

# typing
from typing import List, Tuple
from scipy.sparse import csr_matrix

# text cleaning
import spacy
from multiprocessing import Pool, cpu_count
from tqdm.autonotebook import tqdm

# documents vectorization
from sklearn.feature_extraction.text import TfidfVectorizer


def _data_download(dataset: str) -> Tuple[List[str], List]:
    """
    Given the dataset's name, download it from beir.
    The document title is prepended to the document text 

    Args:
        dataset (str): dataset's name.

    Returns:
        Tuple[List[str],List]:
            -corpus (List[str]): list of document texts. Each document is a string.
            -keys (List): list of document ids.
    """

    # Download dataset and unzip the dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(os.path.abspath('')), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    # Retrieve documents
    documents, _, _ = GenericDataLoader(data_path).load(split="test")

    return [document["title"] + " " + document["text"] for document in documents.values()], list(documents.keys())


# load the spacy model for lemmatization
_nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])


def _cleaner(text: str) -> str:
    """
    Auxiliary function for text cleaning. Given a str, it applies cleaning and lemmatization using spacy library.
    
    Args:
        text (str): text to clean.

    Returns:
        str: cleaned text.
    """
    return " ".join([token.lemma_.lower() for token in _nlp(text) if not token.is_stop and not token.is_punct])


def _corpus_cleaning(corpus: List[str]) -> List[str]:
    """
    Corpus cleaning function (parallelized). 
    
    Args:
        corpus (List[str]): corpus to clean.

    Returns:
        List[str]: cleaned corpus.
    """
    # print(cpu_count())
    with Pool() as p:
        cleaned_corpus = list(tqdm(p.imap(_cleaner, corpus),
                                   total=len(corpus),
                                   desc="documents cleaning"))
    return cleaned_corpus


def _getVectorized(cleaned_corpus: List[str]) -> csr_matrix:
    """
    Corpus vectorization function.
    Given a cleaned corpus, it coverts it into TF-IDF format using sklearn library.
    
    Args:
        cleaned_corpus (List[str]): cleaned corpus.

    Returns:
        csr_matrix: vectorized corpus, in Compressed Sparse Row matrix format.
        Each row represent a document and each colum a specif term in the corpus.
    """

    return TfidfVectorizer().fit_transform(cleaned_corpus)


def data_preparation(dataset: str, limit: int = None) -> Tuple[List, csr_matrix]:
    """
    Data preparation function. Given a dataset name, it:
    -downloads the dataset.
    -clean the dataset
    -limits the number of documents.
    -vectorize the documents.

    Args:
        dataset (str): dataset's name
        limit (int, optional): limit for the number of documents in the corpus. Defaults to None (take full corpus).

    Returns:
        Tuple[List,csr_matrix]:
            -keys (List): list of document ids.
            -docs_matrix (csr_matrix): vectorized corpus.
    """

    corpus, keys = _data_download(dataset)

    if limit is not None:
        corpus = corpus[0:limit]
        keys = keys[0:limit]

    cleaned_corpus = _corpus_cleaning(corpus)

    doc_matrix = _getVectorized(cleaned_corpus)

    return keys, doc_matrix

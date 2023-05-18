from beir.datasets.data_loader import GenericDataLoader
from beir import util

import os
import pathlib

from typing import List

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

def identity_tokenizer( text ):
    return text

from pyspark.sql import SparkSession

from utility import data_preparation

import spacy

from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    # Create SparkSession 
    spark = SparkSession.builder.master("local[2]").appName("SparkByExamples.com").getOrCreate()
    
    sc = spark.sparkContext
    
    print(sc)
    
    corpus = data_preparation("scifact")
    
    myrdd = sc.parallelize(c=corpus)

    #load the spacy model for lemmatization 
    _nlp = spacy.load("en_core_web_lg",disable=['parser','ner'])

    def foo(text: str ):
        cleaned = " ".join([token.lemma_ for token in _nlp(text) if not token.is_stop and not token.is_punct])
        return cleaned

    vectorized = myrdd.map(foo)
    
    print(vectorized.collect())
    
    spark.stop()


if __name__ == "__main__":
    main()
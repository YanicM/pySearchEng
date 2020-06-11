#!/usr/bin/env python3
"""
This is a simple search engine to query local files.
"""

import os
import glob
import math
import json
import spacy
from collections import defaultdict
nlp = spacy.load("en_core_web_sm")

__author__ = "Yanic Moeller"
__license__ = "MIT"


class search_engine:

    def __init__(self, docs):
        self.inverted_index = defaultdict(set)
        self.forward_index = defaultdict(lambda: defaultdict(int))
        self.all_documents = dict()
        for doc in docs:
            doc_hash = hash(doc)
            self.all_documents[doc_hash] = doc
            tokens = self.tokenize(doc)
            for token in tokens:
                self.inverted_index[token].add(doc_hash)
            for token in set(tokens):
                self.forward_index[doc_hash][token] += tokens.count(token)
        self.number_of_pages = len(self.forward_index.keys())
        print(self.all_documents)
        print(self.inverted_index)
        print(self.forward_index)

    def tokenize(self, doc_to_tokenize):
        stop_words = {"the", "a", "an", "is", "this", "to", "be", "-PRON-"}
        document = []
        for word in doc_to_tokenize.split():
            word = "".join([c for c in word if c and c.isalpha()])
            document.append("".join(word.lower()))
        print(document)
        document = nlp(" ".join(document))
        document = [token.lemma_ for token in document if token.lemma_ not in stop_words]
        return document

    def get_tf_idf(self, doc, term):
        """Calculates the TF-IDF value for a term.

        Args:
            doc: The document itself
            tf: How often the term appears in a document
            df: Number of documents that contain this term
            number_of_pages: Number of all documents

        Returns:
            TF-IDF as a float.

        """
        tf = self.forward_index[doc][term] / len(self.forward_index[doc].keys())
        idf = math.log10(self.number_of_pages / len(self.inverted_index[term]))
        return tf * idf

    def query(self, new_query):
        new_query = set(self.tokenize(new_query))
        # result = defaultdict(lambda: defaultdict(int))
        result = defaultdict(int)
        for q in new_query:
            print(q)
            for found_doc in self.inverted_index[q]:
                print(self.get_tf_idf(found_doc, q))
                result[found_doc] += self.get_tf_idf(found_doc, q)
        for id in sorted(result, key=result.get, reverse=True):
            print(result[id], self.all_documents[id])
        return result

if __name__ == "__main__":
    """ This is executed when run from the command line """
    """
    new_docs = []
    folder_path = "./test/"
    for letter in glob.glob(os.path.join(folder_path, "*.json")):
        print(letter)
        with open(letter, encoding="utf-8") as json_file:
            data = json.load(json_file)
            print(data["text"])
            new_docs.append(data["text"])
    """
    new_docs = [
        "oh romeo wherefore art thou art thou?",
        "These Violent Delights Have Violent Ends",
        "The fool doth think he is wise, but the wise man knows himself to be a fool.",
        "Love all, trust a few, do wrong to none.",
        "Though this be madness, yet there is method in't.",
        "What is love?",
        "Could this be madness or such blah?"
    ]
    test = search_engine(new_docs)
    while True:
        my_new_query = input("Enter query: ")
        test.query(my_new_query)

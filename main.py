#!/usr/bin/env python3
"""
This is a simple search engine to query local files.
"""

import os
import glob
import math
import hashlib
import json
import pickle
import spacy
from pyArango.connection import *
from collections import defaultdict
nlp = spacy.load("en_core_web_sm")

__author__ = "Yanic Moeller"
__license__ = "MIT"

conn = Connection(username="root", password="1234", arangoURL="http://127.0.0.1:8530/")
db = conn["_system"]

class PySearchEngine:
    #def __init__(self, docs, files=False, save=False):
    def __init__(self):
        # TODO: Get indices!
        """
        if not files and docs:
            check_page_rank = False
            self.inverted_index = defaultdict(set)
            self.forward_index = defaultdict(lambda: defaultdict(int))
            if isinstance(docs[0], tuple):
                check_page_rank = True
                self.links = dict()
                self.location_hash = dict()
                print("PageRank activated")
            self.all_documents = dict()
            for doc in docs:
                if check_page_rank:
                    doc, location, links = doc
                doc_hash = hash(doc)
                self.all_documents[doc_hash] = doc
                tokens = self.tokenize(doc)
                for token in tokens:
                    self.inverted_index[token].add(doc_hash)
                for token in set(tokens):
                    self.forward_index[doc_hash][token] += tokens.count(token)
            self.number_of_pages = len(self.forward_index.keys())
            # Save these indices to save time in the future.
            print("Inverted Index")
            print(self.inverted_index)
            if save:
                with open('all_documents.txt', 'w', encoding="utf8") as json_file:
                    json.dump(self.all_documents, json_file, default=serialize_sets)
                self.inverted_index = dict(self.inverted_index)
                with open('inverted_index.txt', 'w', encoding="utf8") as json_file:
                    json.dump(self.inverted_index, json_file, default=serialize_sets)
                self.forward_index = dict(self.forward_index)
                with open('forward_index.txt', 'w', encoding="utf8") as json_file:
                    json.dump(self.forward_index, json_file, default=serialize_sets)
        elif files and not docs:
            with open('all_documents.txt', 'r', encoding="utf8") as json_file:
                self.all_documents = json.load(json_file)
            with open('inverted_index.txt', 'r', encoding="utf8") as json_file:
                self.inverted_index = json.load(json_file)
            with open('forward_index.txt', 'r', encoding="utf8") as json_file:
                self.forward_index = json.load(json_file)
            print("Files were successfully retrieved!")
        else:
            raise KeyError("The necessary files weren't provided.")
        print(self.all_documents)
        print(self.inverted_index)
        print(self.forward_index)
        """

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
            term: A term that is in the search query
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
            #for found_doc in self.inverted_index[q]:
            for found_doc in self.inverted_index.set(q, None):
                print(self.get_tf_idf(found_doc, q))
                result[found_doc] += self.get_tf_idf(found_doc, q)
        for doc_id in sorted(result, key=result.get, reverse=True):
            print(result[doc_id], self.all_documents[doc_id])
        return result

    def create_hash_value(self, s):
        hash_object = hashlib.sha256(str(s).encode('utf-8'))
        return hash_object.hexdigest()


class Indexer(PySearchEngine):

    def __init__(self, new_docs):
        all_documents = db["all_documents"]
        inverted_index = db["inverted_index"]
        forward_index = db["forward_index"]
        for doc in new_docs:
            # Create a unique hash for this document.
            doc_hash = self.create_hash_value(doc[0])
            # Store this document in the database.
            doc = {"text": doc[0],
                   "url": doc[1],
                   "links": doc[2]}
            self.upload_doc(collection=all_documents, doc=doc, key=doc_hash, update=False)
            # Create the forward index.
            tokens = self.tokenize(doc["text"])
            new_forward_doc = {"tokens": tokens}
            self.upload_doc(collection=forward_index, doc=new_forward_doc, key=doc_hash, update=False)
            # Create/Update the inverted index.
            for t in tokens:
                self.upload_doc(collection=inverted_index, doc={"document": [doc_hash]}, key=t, update=True)


    def upload_doc(self, collection, doc, key, update=False):
        try:
            doc_in_db = collection[key]
            for k, v in doc.items():
                if update:
                    existing_values = doc_in_db[k]
                    existing_values.append(v[0])
                    doc_in_db[k] = list(set(existing_values))
                else:
                    doc_in_db[k] = v
            doc_in_db.save()
        except Exception as e:
            create_doc = collection.createDocument()
            for k, v in doc.items():
                create_doc[k] = v
            create_doc._key = key
            create_doc.save()

class Crawler:

    def __init__(self):
        pass

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
        ("oh romeo wherefore art thou art thou?", ["a.com"], ["", ""]),
        ("These Violent Delights Have Violent Ends", ["b.com"], ["", "test.com"]),
        ("The fool doth think he is wise, but the wise man knows himself to be a fool.", ["c.com"], ["", ""]),
        ("Love all, trust a few, do wrong to none.", ["d.com"], ["", ""]),
        ("Though this be madness, yet there is method in't.", ["e.com"], ["", ""]),
        ("What is love?", ["f.com"], ["", ""]),
        ("Could this be madness or such blah?", ["g.com"], ["", ""])
    ]
    engine = PySearchEngine()
    index = Indexer(new_docs=new_docs)
    """
    while True:
        new_query = input("Enter query: ")
        engine.query(new_query)
    """

#!/usr/bin/env python3
"""
This is a simple search engine to query local files.
"""

import math
import hashlib
import spacy
from pyArango.connection import *
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

__author__ = "Yanic Moeller"
__license__ = "MIT"

conn = Connection(username="root", password="1234", arangoURL="http://127.0.0.1:8530/")
db = conn["_system"]


class PySearchEngine:

    @staticmethod
    def tokenize(doc_to_tokenize):
        stop_words = {"the", "a", "an", "is", "this", "to", "be", "-PRON-"}
        document = []
        for word in doc_to_tokenize.split():
            word = "".join([c for c in word if c and c.isalpha()])
            document.append("".join(word.lower()))
        print(document)
        document = nlp(" ".join(document))
        document = [token.lemma_ for token in document if token.lemma_ not in stop_words]
        return document

    @staticmethod
    def create_hash_value(s):
        hash_object = hashlib.sha256(str(s).encode('utf-8'))
        return hash_object.hexdigest()

    @staticmethod
    def get_tf_idf(doc, term):
        forward_index = db["forward_index"]
        inverted_index = db["inverted_index"]
        number_of_pages = db["all_documents"].count()
        forward_tokens = forward_index[doc]["tokens"].getStore()
        tf = forward_index[doc]["tokens"][term] / len(forward_tokens.keys())
        inverted_tokens = inverted_index[term]["document"]
        idf = math.log10(number_of_pages / len(inverted_tokens))
        return tf * idf

    def query(self, this_query):
        this_query = set(self.tokenize(this_query))
        result = defaultdict(int)
        all_documents = db["all_documents"]
        inverted_index = db["inverted_index"]
        for q in this_query:
            print(q)
            try:
                found_doc = inverted_index[q]["document"]
                print(found_doc)
                for doc in found_doc:
                    print(doc)
                    result[doc] += self.get_tf_idf(doc, q)
            except:
                print("Word not found")
        for doc_id in sorted(result, key=result.get, reverse=True):
            print(result[doc_id], all_documents[doc_id]["text"])
        return result


class Indexer(PySearchEngine):

    def __init__(self, docs_to_index):
        all_documents = db["all_documents"]
        inverted_index = db["inverted_index"]
        forward_index = db["forward_index"]
        for doc in docs_to_index:
            # Create a unique hash for this document.
            doc_hash = self.create_hash_value(doc[0])
            # Store this document in the database.
            doc = {"text": doc[0],
                   "url": doc[1],
                   "links": doc[2],
                   "page_rank": 0}
            self.upload_doc(collection=all_documents, doc=doc, key=doc_hash, update=False)
            # Create the forward index.
            tokens = self.tokenize(doc["text"])
            new_forward_doc = {token: tokens.count(token) for token in tokens}
            self.upload_doc(collection=forward_index,
                            doc={"tokens": new_forward_doc},
                            key=doc_hash, update=False)
            # Create/Update the inverted index.
            for t in tokens:
                self.upload_doc(collection=inverted_index,
                                doc={"document": [doc_hash]},
                                key=t, update=True)
        # Calculate the page rank.
        for doc in all_documents.fetchAll():
            links = doc.getStore()["links"]
            # url = doc.getStore()["url"]
            print("NEW PAGE RANK:")
            for link in set(links):
                print(link)
                page_rank = (links.count(link) * 0.25) / len(links)
                print(page_rank)
                all_documents[]



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
        ("oh romeo wherefore art thou art thou?", ["a.com"], ["b.com"]),
        ("These Violent Delights Have Violent Ends", ["b.com"], ["a.com"]),
        ("The fool doth think he is wise, but the wise man knows himself to be a fool.",
         ["c.com"], ["a.com", "a.com", "e.com"]),
        ("Love all, trust a few, do wrong to none.", ["d.com"], ["a.com"]),
        ("Though this be madness, yet there is method in't.", ["e.com"], ["a.com"]),
        ("What is love?", ["f.com"], ["a.com"]),
        ("Could this be madness or such blah?", ["g.com"], ["b.com", "d.com", "e.com"])
    ]
    engine = PySearchEngine()
    index = Indexer(docs_to_index=new_docs)
    #while True:
    #    new_query = input("Enter query: ")
    #    engine.query(new_query)

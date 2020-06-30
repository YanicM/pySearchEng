#!/usr/bin/env python3
"""
This is a simple search engine to query local files.
"""

import math
import hashlib
from collections import defaultdict
import spacy
from pyArango.connection import *
from pyArango.theExceptions import DocumentNotFoundError
from bs4 import BeautifulSoup
from bs4.element import Comment
import validators
import urllib.request
from urllib.parse import urljoin, urlsplit, urldefrag, urlparse
from urllib3.exceptions import NewConnectionError, MaxRetryError
from time import sleep

nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load("de_core_news_sm")

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
        # TESTING
        document = [token.encode("ascii", errors="ignore").decode().strip() for token in document]
        document = list(filter(None, document))
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

    @staticmethod
    def url_to_key(url):
        if not isinstance(url, str):
            return "KeyError"
        replace_chars = "".join([c if c.isalnum() else "_" for c in url])
        return replace_chars.encode("ascii", errors="ignore").decode()

    def upload_doc(self, collection, doc, key, update=False):
        sleep(0.05)
        try:
            doc_in_db = collection[key]
            for k, v in doc.items():
                if len(k) > 254 or not k:
                    continue
                if update:
                    existing_values = doc_in_db[k]
                    existing_values.append(v[0])
                    doc_in_db[k] = list(set(existing_values))
                else:
                    doc_in_db[k] = v
            doc_in_db.save()
        except DocumentNotFoundError:
            create_doc = collection.createDocument()
            for k, v in doc.items():
                create_doc[k] = v
            print(f"Creating doc with key '{key}'")
            create_doc._key = key
            create_doc.save()
        # TODO: if upload error -> sleep
        except Exception as e:
            print(str(e))
            sleep(0.3)
            test = update
            self.upload_doc(collection=collection, doc=doc, key=key, update=test)

    def reset_page_rank(self):
        all_documents = db["all_documents"]
        for doc in all_documents.fetchAll():
            doc["page_rank"] = 0
            doc.save()

    def create_page_rank(self):
        all_documents = db["all_documents"]
        temp_urls = dict()
        urls = db["urls"]
        temp_page_rank = defaultdict(float)
        for doc in all_documents.fetchAll():
            links = doc.getStore()["links"]
            temp_url = self.url_to_key(doc.getStore()["url"])
            temp_urls[temp_url] = doc._key
            for link in set(links):
                page_rank = (links.count(link) * 0.25) / len(links)
                temp_page_rank[link] += page_rank
        for url, rank in temp_page_rank.items():
            print(url, rank)
            url = self.url_to_key(url)
            print(url)
            try:
                result = all_documents[temp_urls[url]]
                result["page_rank"] = rank
                result.save()
                print(f"updated {url}")
            except:
                continue


    def query(self, this_query):
        this_query = set(self.tokenize(this_query))
        result = defaultdict(int)
        all_documents = db["all_documents"]
        inverted_index = db["inverted_index"]
        for q in this_query:
            try:
                found_doc = inverted_index[q]["document"]
                for doc in found_doc:
                    # Calculate the TF-IDF.
                    result[doc] += self.get_tf_idf(doc, q)
                    # Retrieve the PageRank.
                    page_rank = all_documents[doc]["page_rank"]
                    result[doc] += page_rank
            except:
                print("Word not found")
        # Display the results.
        for doc_id in sorted(result, key=result.get, reverse=True):
            if len(all_documents[doc_id]["text"]) > 50:
                print(result[doc_id], all_documents[doc_id]["url"], all_documents[doc_id]["text"][:50])
            else:
                print(result[doc_id], all_documents[doc_id]["url"], all_documents[doc_id]["text"])
        return result


class Indexer(PySearchEngine):

    def __init__(self, docs_to_index):
        all_documents = db["all_documents"]
        inverted_index = db["inverted_index"]
        forward_index = db["forward_index"]
        urls = db["urls"]
        for doc in docs_to_index:
            # Create a unique hash for this document.
            doc_hash = self.create_hash_value(doc[0])
            # Store this document in the database.
            doc = {"text": doc[0],
                   "url": doc[1],
                   "links": doc[2],
                   "page_rank": 0}
            print(f"trying to upload {doc}")
            self.upload_doc(collection=all_documents, doc=doc, key=doc_hash, update=False)
            # Store URLs with their hashes.
            self.upload_doc(collection=urls, doc={"hash": doc_hash}, key=str(doc["url"][0]),
                            update=False)
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
        temp_page_rank = defaultdict(float)
        for doc in all_documents.fetchAll():
            links = doc.getStore()["links"]
            for link in set(links):
                page_rank = (links.count(link) * 0.25) / len(links)
                temp_page_rank[link] += page_rank
        for url, rank in temp_page_rank.items():
            url = self.url_to_key(url)
            try:
                hash_url = urls[url]["hash"]
                to_update = all_documents[hash_url]
                to_update["page_rank"] = rank
                to_update.save()
            except:
                continue


class Crawler(PySearchEngine):

    def __init__(self, domain):
        print(f"Crawling: {domain}")
        self.domain = domain
        self.links = defaultdict(list)
        self.documents = dict()
        parsed_domain = urlparse(self.domain)
        self.netloc = "{uri.netloc}".format(uri=parsed_domain)
        if self.netloc.startswith("www."):
            self.netloc = self.netloc[4:]

    def collect_links(self, domain=None):
        #urls_to_crawl = set([self.domain])
        domain = domain.strip()
        print(f"Crawling -> {domain}")
        """
        ends = [".gz", ".pdf", ".tgz", ".txt", ".pptx", ".ptx", ".zip", ".jpg", ".gif", ".jpeg", ".ppt", ".doc", ".ps", ".png", ".js", ".css", ".md"]
        for end in ends:
            if end in domain:
                del self.links[domain]
                print("Not an HTML file")
                return
        """
        if domain.endswith(".html") or domain.endswith(".php") or domain.endswith("/"):
            print("Reading...")
        elif not "." in domain[-5:]:
            print("Reading...")
        else:
            return
        base = domain
        parsed_domain = urlparse(domain)
        this_netloc = "{uri.netloc}".format(uri=parsed_domain)
        if this_netloc.startswith("www."):
            this_netloc = this_netloc[4:]
        if str(self.netloc) != str(this_netloc):
            return
        print(f"original netloc {self.netloc}")
        print(f"this netloc {this_netloc}")
        # Filter other formats.
        parser = 'html.parser'
        resp = urllib.request.urlopen(domain)
        soup = BeautifulSoup(resp, parser, from_encoding=resp.info().get_param('charset'))
        if self.has_head(soup):
            for link in soup.find_all('a', href=True):
                #sleep(0.2)
                #print(link['href'])
                full_href = urljoin(base, link["href"])
                #print(full_href)
                no_query = urlsplit(full_href)._replace(query=None).geturl()
                unfragmented = urldefrag(no_query)
                print(unfragmented[0])
                if this_netloc in full_href and validators.url(full_href):
                    self.links[domain].append(unfragmented[0])

            for link in self.links[domain]:
                if not self.links[link]:
                    try:
                        self.collect_links(link)
                    except:
                        # 404
                        continue

            if not self.links[domain]:
                self.links[domain].append(True)

    def store_urls(self):
        crawler = db["urls"]
        for url in self.links.keys():
            self.upload_doc(collection=crawler, doc={"links_to": self.links[url]}, key=self.url_to_key(url), update=False)

    def tag_visible(self, element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    def text_from_html(self, body):
        soup = BeautifulSoup(body, 'html.parser')
        texts = soup.findAll(text=True)
        visible_texts = filter(self.tag_visible, texts)
        return u" ".join(t.strip() for t in visible_texts)

    def has_head(self, test_soup):
        if test_soup.head:
            return True
        else:
            return False

    def retrieve_all_documents(self):
        for link in self.links.keys():
            try:
                html = urllib.request.urlopen(link).read()
                text = self.text_from_html(html)
                self.documents[link] = (text, link, self.links[link])
                print(f"{link} = {(text, [link], self.links[link])}")
            except:
                # e.g. no access
                continue
        return self.documents


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
    # TODO: URL not as list
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
    ## c = Crawler("https://www.gatsbyjs.org/")
    #c = Crawler("https://corrux.io/")
    ## c = Crawler("https://www.cis.uni-muenchen.de/")
    ## c = Crawler("http://www.gpsbasecamp.com/")
    ## c = Crawler("http://www.dogthebountyhunter.com/")
    ## c = Crawler("https://www.fullstackpython.com/")
    ## c = Crawler("http://www.realpython.com/")
    ## c.collect_links("https://cis.uni-muenchen.de/")
    ## c.collect_links("https://www.gatsbyjs.org/")
    #c.collect_links("https://corrux.io/")
    ## c.collect_links("http://www.gpsbasecamp.com/")
    ## c.collect_links("http://www.dogthebountyhunter.com/")
    ## c.collect_links("https://www.fullstackpython.com/")
    ## c.collect_links("http://www.realpython.com/")
    #c.store_urls()
    #my_docs = c.retrieve_all_documents()
    ## print(my_docs.values())
    #index = Indexer(docs_to_index=my_docs.values())
    #engine.reset_page_rank()
    engine.create_page_rank()
    while True:
        new_query = input("Enter query: ")
        engine.query(new_query)
    #print(c.url_to_key("http://cis.uni-muenchen.de/"))

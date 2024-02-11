import logging
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk, bulk
from collections import deque

import json
from typing import List, Dict, Union, Optional, Type, Any

logger = logging.getLogger(__name__)

def overrides(interface_class):
    """
    Function override annotation.
    Corollary to @abc.abstractmethod where the override is not of an
    abstractmethod.
    Modified from answer https://stackoverflow.com/a/8313042/471376
    """

    def confirm_override(method):
        if method.__name__ not in dir(interface_class):
            raise NotImplementedError('function "%s" is an @override but that'
                                      ' function is not implemented in base'
                                      ' class %s'
                                      % (method.__name__,
                                         interface_class)
                                      )

        def func():
            pass

        attr = getattr(interface_class, method.__name__)
        if type(attr) is not type(func):
            raise NotImplementedError('function "%s" is an @override'
                                      ' but that is implemented as type %s'
                                      ' in base class %s, expected implemented'
                                      ' type %s'
                                      % (method.__name__,
                                         type(attr),
                                         interface_class,
                                         type(func))
                                      )
        return method

    return confirm_override


class SearchEngine:
    """
    Basic Search Engine
    """
    def __init__(self, name):
        """
        @param name: The name of the Search Engine
        """
        self.name = name

    def _init_search_engine(self):
        """Initialize Search Engine"""
        raise NotImplementedError

    def search(self):
        """The search module to search in the search engine"""
        raise NotImplementedError

    def upload(self, docs: List[Any]):
        """Upload data on the search engine"""
        raise NotImplementedError

    def __str__(self) -> str:
        return "Search Engine \nName: " + self.name


class ElasticEngine(SearchEngine):
    """A search engine based on elastic search"""
    def __init__(self, name: str, ip: str, port: int,
                 indices: str, mapping: Optional[str] = None):
        """
        @param name: The name of the search engine
        @param ip: The host ip of elastic search (ES)
        @param port: The port of ES
        @param indices: The indices in ES, i.e. the collection where the search will happen
        @param mapping: If a string it needs to be file name which contains the mapping json,
                        otherwise it can be the json itself.
        """
        super(ElasticEngine, self).__init__(name)
        self.ip = ip
        self.port = port
        self.indices = indices
        self._init_search_engine()
        if mapping:
            self._create_indices(mapping)

    @overrides(SearchEngine)
    def _init_search_engine(self):
        self.searchEngine = Elasticsearch([{'host': self.ip, 'port': self.port}])

    def delete_indices(self, indices_name: str) -> bool:
        """Delete an index, if it already exists returns True else False"""
        if self.searchEngine.indices.exists(index=indices_name):
            self.searchEngine.indices.delete(index=indices_name)
            return True
        return False

    def _create_indices(self, mapping) -> None:
        if isinstance(mapping, str):
            with open(mapping, "r") as mapping_file:
                mapping = json.load(mapping_file)
        if self.index_exists():
            logger.info(f"The {self.indices} index already exists.")
        else:
            self.searchEngine.indices.create(index=self.indices,
                                             # ignore=400,
                                             body=mapping)
            
    def index_exists(self):
        return self.searchEngine.indices.exists(index=self.indices)

    @overrides(SearchEngine)
    def search(self, text: str, n_results: int) -> Dict:
        query = {
                    "bool":{
                        "must": [
                            {
                                "match": {
                                    "text": {
                                        "query": text
                                    }
                                }
                            }
                        ]
                    }
                }
        results = self.searchEngine.search(index=self.indices,
                                           body={
                                                 "query": query,
                                                 "size": n_results
                                             },
                                             request_timeout=30)
        return results

    @overrides(SearchEngine)
    def upload(self, docs: List[Dict]) -> None:
        ready_docs = [self.ready_for_upload(doc, self.indices) for doc in docs]
        # bulk()
        deque(parallel_bulk(self.searchEngine, ready_docs), maxlen=0)
        self.searchEngine.indices.refresh(index=self.indices)
        
        
    def set_bm25_params(self, b=0.75, k1=1.2):
        self.searchEngine.indices.close(index=self.indices)

        self.searchEngine.indices.put_settings(
            index=self.indices,
            body={
                "similarity": {
                    "myBM25": {"type": "BM25", "b": b, "k1": k1}
                }
            },
        )

        self.searchEngine.indices.open(index=self.indices)

    def __str__(self) -> str:
        start = "Search Engine based on Elastic Search\n"
        mid = f"name: {self.name}\n"
        end = f"host ip: {self.ip} \nport: {self.port}\n"
        return start + mid + end

    @staticmethod
    def ready_for_upload(doc: Dict, indices_name: str) -> Dict:
        ready_doc = {"_index": indices_name}
        if "_id" in doc.keys():
            ready_doc["_id"] = doc["_id"]
            doc.pop("_id")
            ready_doc["_source"] = doc
        return ready_doc

#!/usr/bin/env python3
""" Log stats """
from pymongo import MongoClient


if __name__ == "__main__":
    """ script that provides some stats about Nginx logs stored in MongoDB

        Database: logs
        Collection: nginx
        Display:
            - first line: x logs where x is the number of documents in this
              collection
            - second line: Methods:
            - 5 lines with the number of documents with the
                - method = ["GET", "POST", "PUT", "PATCH", "DELETE"]
                  in this order
            - one line with the number of documents with:
                - method=GET
                - path=/status
    """
    client = MongoClient('mongodb://127.0.0.1:27017')
    collection_logs = client.logs.nginx
    num_docs = collection_logs.count_documents({})
    print("{} logs".format(num_docs))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        num_method = collection_logs.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, num_method))
    filter_path = {"method": "GET", "path": "/status"}
    num_path = collection_logs.count_documents(filter_path)
    print("{} status check".format(num_path))

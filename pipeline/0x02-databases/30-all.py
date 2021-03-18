#!/usr/bin/env python3
""" lists all documents in a collection """


def list_all(mongo_collection):
    """ Lists all documents in a collection:

        Return: an empty list if no document in the collection
            - mongo_collection will be the pymongo collection object
    """
    docs = []
    collection = mongo_collection.find()
    for doc in collection:
        docs.append(doc)
    return docs

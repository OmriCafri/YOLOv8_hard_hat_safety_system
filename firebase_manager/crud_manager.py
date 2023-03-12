import os
import sys

import yaml
import firebase_admin
from firebase_admin import credentials, firestore

try:
    with open('../configurations.yaml', 'r') as f:
        conf = list(yaml.load_all(f, Loader=yaml.SafeLoader))[0]
except:
    with open('configurations.yaml', 'r') as f:
        conf = list(yaml.load_all(f, Loader=yaml.SafeLoader))[0]

# Initializing connection
cred = credentials.Certificate(conf['firebase_creds'])
firebase_admin.initialize_app(cred)
db = firestore.client()
# Connecting to collection
default_collection = db.collection(conf['collection_name'])


def create(documents, documents_names, collection_name=None):
    """
    Function that writes documents to firebase database

    Parameters
    ----------
    documents: list[dict,...]
        List of dictionaries (documents) for writing to the DB.
    documents_names: list[str,...]
        List of strings for documents names.
    collection_name: str (optional)
        The name of the collection if we don't want to write to the default collection.
    """
    # Getting the right collection
    if collection_name is not None:
        collection = db.collection(collection_name)
    else:
        collection = default_collection

    # Checking the lengths are equals
    assert len(documents) == len(documents_names), 'documents and documents_names should be in the same length'

    # writing docs to the DB
    for name, document in zip(documents_names, documents):
        collection.document(name).set(document)


def read(collection_name=None, document_name=None):
    """
    Function for reading a collection / a document from a collection.

    Parameters
    ----------
    collection_name: str (optional)
        The name of the collection if we don't want to write to the default collection.
    document_name: str (optional)
        A specific document for querying.

    Returns
    -------
    list[dic,...]
        List of dictionaries with all the documents
    """
    # Getting the right collection
    if collection_name is not None:
        collection = db.collection(collection_name)
    else:
        collection = default_collection

    # specific document
    if document_name is not None:
        return [collection.document(document_name).get().to_dict()]
    # the whole collection
    return [document.to_dict() for document in collection.stream()]


if __name__ == '__main__':
    results = read(collection_name='new_collection')
    print(results)
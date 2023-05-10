from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np

df = pd.read_csv('your_text_dataset.csv')

model = hub.load("path/model v4")

with tf.compat.v1.Session() as session:
    session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
    query_array = session.run(model(df["Text"]))

vectors = []
for i in query_array:
    vectors.append(i)

df["Embeddings"] = vectors

http_auth = ("elastic_username", "elastic_password")
es_host = "https://localhost:9200"
context = create_default_context(cafile="http_ca.crt")
es = Elasticsearch(
    es_host,
    basic_auth=http_auth,
    ssl_context=context
)

configurations = {
    "settings": {
        "analysis": {
            "filter": {
                 "ngram_filter": {
                     "type": "edge_ngram",
                     "min_gram": 2,
                     "max_gram": 15,
                 },
                "english_stop": {
                  "type":       "stop",
                  "stopwords":  "_english_" 
                },
                 "english_keywords": {
                   "type":       "keyword_marker",
                   "keywords":   ["example"] 
                 },
                "english_stemmer": {
                  "type":       "stemmer",
                  "language":   "english"
                },
                "english_possessive_stemmer": {
                  "type":       "stemmer",
                  "language":   "possessive_english"
                }
            },
            "analyzer": {
                "en_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", 
                                "ngram_filter", 
                                "english_stemmer",
                               "english_possessive_stemmer",
                                "english_stop"
                                "english_keywords",
                                ]
                }
            } 
        } 
    },
    "mappings": {
        "properties": {
          "Embeddings": {
            "type": "dense_vector",
            "dims": 512,
            "index": True,
            "similarity": "cosine" 
          },
          } 
        } 
    } 
configurations["settings"]


es.indices.create(index='my_new_index',
                    settings=configurations["settings"],
                    mappings=configurations["mappings"]
                    )


actions = []
index_name = 'my_new_index'
for index, row in df.iterrows():
    action = {"index": {"_index": index_name, "_id": index}}
    doc = {
        "id": index,
        "Text": row["Text"],
        "Price": row["Price"],
        "Quantity": row["Quantity"],
        "Embeddings": row["Embeddings"]
    }
    actions.append(action)
    actions.append(doc)

es.bulk(index=index_name, operations=actions)


query = "Which is the latest phone available in your shop"

with tf.compat.v1.Session() as session:
    session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
    query_array = session.run(model([query])).tolist()[0]


query_for_search = {
  "knn": {
    "field": "Embeddings",
    "query_vector": query_array,
    "k": 5,
    "num_candidates": 2414
  },
  "_source": [ "Text"]
}


result = es.search(
    index="my_new_index",
    body=query_for_search)
result0["hits"]
import os
from django.test import TestCase

# Create your tests here.
from ai_llm_competition.settings import milvus_config
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.milvus import Milvus

os.chdir('../')
print(os.getcwd())
milvus_retriever = None
embeddings = {}


# def get_embedding(model_name):
#     if embeddings.get(model_name) is None:
#         embedding = HuggingFaceEmbeddings(
#             model_name=model_name,
#             model_kwargs={'device': 'cpu'},
#             encode_kwargs={'normalize_embeddings': False}
#         )
#         embeddings[model_name] = embedding
#     return embeddings[model_name]
#
#
# def get_retriever(drop_old=False, k=10, collection=None):
#     global milvus_retriever
#     embedding = get_embedding('./bge-base-zh')
#     if milvus_retriever is None:
#         vector_store = Milvus(
#             embedding_function=embedding,
#             collection_name=collection,
#             connection_args=milvus_config,
#             auto_id=True,
#             drop_old=drop_old
#         )
#         # milvus_retriever = vector_store.as_retriever()
#         milvus_retriever = vector_store.as_retriever(search_kwargs={'k': k})
#
#     return milvus_retriever
#
#
# ret = get_retriever(k=20, collection='audit_know_milvus_update').invoke(
#     "被审计期间内，xxxx万高立项自研科技项目24项，其中14项已结项，审查全xxxx验收项目结项资料"
#     "（包括但不限于项目可研报告、工作报告、技术报告、决算报告、验收报告等资料），发现自研科技项目验收管理不规范，审计期间内14项验收项目审计报告缺失。")
# print(ret)



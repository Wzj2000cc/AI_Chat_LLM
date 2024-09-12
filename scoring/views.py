import asyncio
import hashlib
import json
import os
import re
import uuid
import pandas as pd

from datetime import datetime
from collections import defaultdict, OrderedDict
from django.http import JsonResponse, HttpRequest, FileResponse, Http404, StreamingHttpResponse
from django.shortcuts import render
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
from langchain.vectorstores.milvus import Milvus
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from FlagEmbedding import FlagReranker
from concurrent.futures import ThreadPoolExecutor

from ai_llm_competition.settings import es_config, milvus_config, conn, BASE_DIR, get_Qianfan, get_Tongyi, get_Spark, \
    llm_flow, generate_text

os.chdir('./')

# ----------------------------------------------- 知识召回功能 -----------------------------------------------------------
embeddings = {}
_bert_cache = {}
_bert_tokenizer = None
_bert_model = None
milvus_retriever = None

es_client = Elasticsearch(es_config['host'], basic_auth=es_config['basic_auth'])


def clean_text(text):
    # 去除特定字符和连续的星号
    text = re.sub(r'[^\w\s，。、：#【】（）,.-]', '', text, flags=re.IGNORECASE)
    # 替换金额和连续的星号
    pattern = r'\d{1,3}(?:,\d{3})*\.\d+|\d{1,3}(?:,\d{3})*|\d+'
    text = re.sub(pattern, lambda match: '*', text)
    text = re.sub(r'xx+', '*', text)
    text = re.sub(r'\*+', '*', text)
    return text.strip()


def generate_unique_identifier(text):
    # 使用 SHA-256 哈希算法
    sha256 = hashlib.sha256()
    sha256.update(text.encode('utf-8'))
    return sha256.hexdigest()


def get_embedding(model_name):
    if embeddings.get(model_name) is None:
        embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        embeddings[model_name] = embedding
    return embeddings[model_name]


def get_retriever(drop_old=False, k=10, collection=None):
    global milvus_retriever
    milvus_retriever = None
    embedding = get_embedding('./bge-base-zh')
    if milvus_retriever is None:
        vector_store = Milvus(
            embedding_function=embedding,
            collection_name=collection,
            connection_args={
                'host': milvus_config['host'],
                'port': milvus_config['port']
            },
            auto_id=True,
            drop_old=drop_old
        )
        milvus_retriever = vector_store.as_retriever(search_kwargs={'k': k})
    return milvus_retriever


def set_result_search(question, res1, res2):
    merged_list = []
    result_recall_unique = []
    for item in res1:
        merged_list.append(item)
    for item in res2:
        merged_list.append(item)
    # 使用set去重
    unique_set = set(merged_list)
    for item in unique_set:
        result_recall_unique.append({'question_text': question, 'content': item})
        # result_recall_unique.append(item)

    return result_recall_unique


def get_ranker(question, content_list, num):
    # 加载模型和tokenizer  BGE本地模型路径
    model_path = "./bge-reranker"
    reranker = FlagReranker(model_path, use_fp16=True)

    # 存储结果的列表
    result_list = []

    def compute_similarity(content):
        score = reranker.compute_score([question, content], normalize=True)
        return score, content

    with ThreadPoolExecutor() as executor:
        # 并行计算相似度
        similarities = executor.map(compute_similarity, content_list)

    # 按相似度排序，索引截取获得想要召回内容的个数
    top_5_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)[:num]

    for score, content in top_5_similarities:
        content = str(content).strip()
        result_list.append(content)

    return result_list


def bert_encode_string(s, need_cache=False):
    if s in _bert_cache:
        return _bert_cache[s]
    global _bert_tokenizer, _bert_model
    _bert_tokenizer = None
    _bert_model = None
    if _bert_tokenizer is None:
        _bert_tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese', clean_up_tokenization_spaces=True)
    if _bert_model is None:
        _bert_model = BertModel.from_pretrained('./bert-base-chinese')
    inputs = _bert_tokenizer(s, return_tensors='pt')
    outputs = _bert_model(**inputs)
    # 取[CLS]标记的向量表示
    return outputs.last_hidden_state[:, 0, :].detach().numpy()


def bert_ranker_search(query: str, res_data, k: int = None):
    # 向量化编码并存入内存
    list_data = [item['content'] for item in res_data]
    encoded_strings = {s: bert_encode_string(s[:512], True) for s in list_data}
    # 条件字符串
    encoded_query = bert_encode_string(query, False)
    result_fine_recall_milvus = []
    # 计算相似度并排序
    similarities = {s: cosine_similarity(encoded_query, v).flatten()[0] for s, v in encoded_strings.items()}
    sorted_strings = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    for s, similarity in sorted_strings:
        content = None
        for part in res_data:
            if part['content'] in s or s in part['content']:
                content = part['content']
                break
        result_fine_recall_milvus.append({
            "question_text": query,
            "content": content
        })
        if len(result_fine_recall_milvus) >= k:
            return result_fine_recall_milvus


class CoarseRecall:

    def __init__(self, query, coarse_size, es_name, milvus_name):
        self.query = query
        self.coarse_size = coarse_size
        self.es_name = es_name
        self.milvus_name = milvus_name

    def es_recall_search(self):
        if self.query:
            s = Search(using=es_client, index=self.es_name) \
                .query('match', content=self.query) \
                .extra(size=self.coarse_size)
        else:
            s = Search(using=es_client, index=self.es_name) \
                .query(Q('function_score',
                         functions=[{'random_score': {}}],
                         boost_mode='replace')) \
                .extra(size=self.coarse_size)
        ret = s.execute()
        result_recall_es_search = []
        for doc in ret:
            text = doc.content
            # result_recall_es_search.append({
            #     "question_text": self.query,
            #     "content": text
            # })
            result_recall_es_search.append(text)

        return result_recall_es_search, ret

    def milvus_recall_search(self):
        if self.query:
            res = get_retriever(k=int(self.coarse_size), collection=self.milvus_name).invoke(self.query)
        else:
            res = get_retriever(k=int(self.coarse_size), collection=self.milvus_name).invoke('')
        result_recall_milvus_search = []
        for doc in res:
            text = doc.page_content
            result_recall_milvus_search.append(text)
        return result_recall_milvus_search, res


def recall(request):
    context = {}
    if request.method == 'GET':
        if request.user.is_authenticated:
            return render(request, 'get_data.html')
        return render(request, 'login.html', context)

    if request.method == 'POST':
        question = request.POST.get('question')
        es_name = request.POST.get('es')
        milvus_name = request.POST.get('milvus')
        coarse_size = request.POST.get('coarse_topk')
        fine_size = request.POST.get('fine_topk')

        if not question or not coarse_size or not fine_size:
            return JsonResponse({'message': '注意⚠️必填项信息存在遗漏、请完整信息后提交！😤'})
        elif not es_name and not milvus_name:
            return JsonResponse({'message': '请至少输入一个知识库的名称！🙄'})

        coarse_recall = CoarseRecall(
            query=question,
            coarse_size=coarse_size,
            es_name=es_name,
            milvus_name=milvus_name  # 如果不使用Milvus，可以传递None或设置默认值
        )
        es_res = []
        milvus_res = []

        if es_name and not milvus_name:
            es_res, _ = coarse_recall.es_recall_search()
        elif not es_name and milvus_name:
            milvus_res, _ = coarse_recall.milvus_recall_search()
        else:
            es_res, _ = coarse_recall.es_recall_search()
            milvus_res, _ = coarse_recall.milvus_recall_search()

        set_res = set_result_search(question, es_res, milvus_res)
        bert_res = bert_ranker_search(question, set_res, int(fine_size))

        res_list = []

        for part in bert_res:
            item = part['content']
            res_list.append(item)

        return JsonResponse({'message': res_list})


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------- 知识录入功能 -----------------------------------------------------------

class KnowInsert:

    def __init__(self, df, es_index_name, milvus_index_name, es_embedding, milvus_embedding):
        self.df = df
        self.es_index_name = es_index_name
        self.milvus_index_name = milvus_index_name
        self.es_embedding = es_embedding
        self.milvus_embedding = milvus_embedding

    async def es_data_ruku(self):
        try:
            if es_client.indices.exists(index=self.es_index_name):
                es_client.indices.delete(index=self.es_index_name)
            settings = {
                "analysis": {"analyzer": {"default": {"type": "standard"}}},
                "similarity": {
                    "custom_bm25": {
                        "type": self.es_embedding,
                        "k1": 2.,
                        "b": .75,
                    }
                },
            }
            mappings = {
                "properties": {
                    "content": {
                        "type": "text",
                        "similarity": "custom_bm25",  # Use the custom BM25 similarity
                    }
                }
            }
            es_client.indices.create(index=self.es_index_name, mappings=mappings, settings=settings)
        except Exception as e:
            print(f"Error creating Elasticsearch index: {e}")

        try:
            if isinstance(self.df, pd.DataFrame):
                df_renamed = self.df.rename(columns={'chunk': 'content'})
                df_list = df_renamed.to_dict('records')
                num = len(df_list)
                for row in df_list:
                    es_client.index(index=self.es_index_name, body=row)
            else:
                num = len(self.df)
                for item in self.df:
                    es_client.index(index=self.es_index_name,
                                    body={'content': item["content"], 'content_id': item["content_id"]})

            print(f'ES库已更新，索引为{self.es_index_name}')
            index_statute = "已完成"
            es_values = (num, index_statute, self.es_index_name, 'ES')
            mysql_update(es_values)
            return num
        except Exception as eror:
            print(f"Error adding texts to Elasticsearch: {eror}")

    async def milvus_data_ruku(self):

        model_name = './' + self.milvus_embedding
        embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

        vector_store = Milvus(
            embedding_function=embedding,
            collection_name=self.milvus_index_name,
            connection_args={
                'host': milvus_config['host'],
                'port': milvus_config['port']
            },
            auto_id=True,
            drop_old=True
        )

        if isinstance(self.df, pd.DataFrame):
            texts = self.df['chunk'].tolist()
            other_fields = self.df.columns.tolist()
            other_fields.remove('chunk')
            other_df = self.df[other_fields]
            tables_info = other_df.to_dict('records')
            total = len(texts)
            vector_store.add_texts(texts=texts, metadatas=tables_info)

        else:
            total = len(self.df)
            for text in self.df:
                table_info = {"content_id": text['content_id']}
                vector_store.add_texts(texts=[text["content"]], metadatas=[table_info])

        print(f'Milvus库已更新，索引为{self.milvus_index_name}')
        index_statute = "已完成"
        Milvus_values = (total, index_statute, self.milvus_index_name, 'Milvus')
        mysql_update(Milvus_values)

        return total


def mysql_add_userdata(values):
    cursor = conn.cursor()
    insert_stmt = (
        "INSERT INTO KNOW_USER_INFO (username,database_type,embedding,index_name,get_date,project_name,index_statute,"
        "know_content) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
    )
    # 要插入的数据
    data = values

    # 执行SQL语句
    cursor.execute(insert_stmt, data)
    conn.commit()


def mysql_update(values):
    cursor = conn.cursor()
    insert_stmt = (
        "update KNOW_USER_INFO set kg_count = %s,index_statute=%s where index_name =%s and database_type = %s"
    )
    # 要插入的数据
    data = values

    # 执行SQL语句
    cursor.execute(insert_stmt, data)
    conn.commit()


def handle_uploaded_file(file, file_name):
    file_path = f"./scoring/uploaded_files/{file_name}"

    with open(file_path, "wb+") as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    return file_path


async def sync_function_correctly(content_list, result_list, type_name, model_name, index_name):
    if type_name == 'ES':
        know_insert_query = KnowInsert(
            df=result_list,
            es_index_name=index_name + '_query_all',
            milvus_index_name=None,
            es_embedding=model_name,
            milvus_embedding=None
        )
        await asyncio.create_task(know_insert_query.es_data_ruku())

    elif type_name == 'Milvus':
        know_insert_query = KnowInsert(
            df=result_list,
            es_index_name=None,
            milvus_index_name=index_name + '_query_all',
            es_embedding=None,
            milvus_embedding=model_name
        )
        await asyncio.create_task(know_insert_query.milvus_data_ruku())


async def sync_function_querynum(df, index_name, model_name, type_name, project_name):
    if isinstance(df, pd.DataFrame):
        know_insert = KnowInsert(
            df=df,
            es_index_name=str(project_name) + '_es',
            milvus_index_name=str(project_name) + '_milvus',
            es_embedding='BM25',
            milvus_embedding='bge-base-zh'
        )
        await asyncio.create_task(know_insert.es_data_ruku())
        await asyncio.create_task(know_insert.milvus_data_ruku())

    elif type_name == 'ES' and not project_name:
        know_insert = KnowInsert(
            df=df,
            es_index_name=index_name,
            milvus_index_name=None,
            es_embedding=model_name,
            milvus_embedding=None
        )
        await asyncio.create_task(know_insert.es_data_ruku())

    elif type_name == 'Milvus' and not project_name:
        know_insert = KnowInsert(
            df=df,
            es_index_name=None,
            milvus_index_name=index_name,
            es_embedding=None,
            milvus_embedding=model_name
        )
        await asyncio.create_task(know_insert.milvus_data_ruku())


def get_prompt(filename):
    with open(filename, 'r') as f:
        texts = [i.strip() for i in f.read().strip().replace('\r', '').split('\n\n')]
    return [{'content_id': str(uuid.uuid4()).replace("-", ""), 'content': text} for text in texts]


def upload(request):
    context = {}
    if request.method == 'GET':
        if request.user.is_authenticated:
            return render(request, 'knowledge.html', context)
        return render(request, 'login.html', context)

    elif request.method == 'POST':
        user = request.user
        up_file = request.FILES.get('file')
        up_dataframe = request.POST.get('up_dataframe')
        index_name = request.POST.get('name')
        model_name = request.POST.get('embedding')
        project_name = request.POST.get('project_name')
        know_content = request.POST.get('content')

        if isinstance(up_dataframe, pd.DataFrame):
            project_name = project_name.strip()
            asyncio.run(
                sync_function_querynum(up_dataframe, None, None, index_name, project_name))

        else:
            if not up_file or not index_name or not model_name:
                error_message = '注意⚠️必填项信息存在遗漏、请将信息补充完整后提交！😤'
                return JsonResponse({'message': error_message})
            cursor = conn.cursor()
            check_stmt = "select count(1) from KNOW_USER_INFO where index_name = %s"
            cursor.execute(check_stmt, (index_name,))
            if cursor.fetchall()[0][0]:
                return JsonResponse({'message': f'集合名称{index_name}重复了，请修改！'})

            type_name, model_name = model_name.split('@', 1)
            file_path = handle_uploaded_file(up_file, str(up_file))
            index_name = index_name.strip()

            texts = []
            df = pd.read_excel(file_path, engine='openpyxl')
            for index, row in df.iterrows():
                row_string = ', '.join([str(value) for value in row])
                texts.append(row_string)
            content_list = [
                {'content_id': str(uuid.uuid4()).replace("-", ""), 'content': text, 'index_name': index_name}
                for text in texts]

            now = datetime.now()
            if type_name == 'ES':
                es_values = (str(user), type_name, model_name, index_name, now, project_name, '执行中', know_content)
                # es_values_all = (
                #     str(user), type_name, model_name, index_name + '_query_all', now, project_name, '执行中',
                #     know_content)
                mysql_add_userdata(es_values)
                # mysql_add_userdata(es_values_all)
                asyncio.run(
                    sync_function_querynum(content_list, index_name, model_name, type_name, None))
            elif type_name == 'Milvus':
                miv_values = (str(user), type_name, model_name, index_name, now, project_name, '执行中', know_content)
                # miv_values_all = (
                #     str(user), type_name, model_name, index_name + '_query_all', now, project_name, '执行中',
                #     know_content)
                mysql_add_userdata(miv_values)
                # mysql_add_userdata(miv_values_all)
                asyncio.run(
                    sync_function_querynum(content_list, index_name, model_name, type_name, None))

            # pattern = r'问题：\s*(.*?)(?=\n问题：|\n|$)'
            # result_list = []
            # count = 0
            # for index, item in enumerate(content_list, start=0):
            #     count += 1
            #     content = item['content'].replace('\n', '')
            #     prompt = f"""
            #         你是一个问题改写助手，根据以下参考问题，改写提出2个新的问题，你需按照如下流程工作:
            #         1. 思考理解参考问题的含义。
            #         2. 根据以上思考改写生成新问题，问题内容紧密围绕参考问题。问题需包含清晰的主语和提问诉求。
            #
            #         ## 注意事项
            #         1. 对于生成的问题，禁止任何解释，仅使用精炼、清晰的措辞。
            #         2. 根据参考问题，生成4个问题。并将生成的每个问题对、都以换行的格式进行输出。请严格按照以下格式输出:
            #         问题：【生成的问题？】\n
            #         问题：【生成的问题？】
            #
            #         ## 参考资料：{clean_text(content)}
            # """
            #     try:
            #         llm_result = get_Spark(prompt)
            #     except Exception as e:
            #         continue
            #     matches = re.findall(pattern, llm_result, re.DOTALL)
            #     result_list.extend([{'content': match, "content_id": item['content_id']} for match in matches])
            #
            # query = "INSERT INTO questions_answers (q_a, q_a_id, index_name) VALUES (%s, %s, %s)"
            # cursor.executemany(query, [(content["content"], content["content_id"], content["index_name"])
            #                            for content in content_list])
            # conn.commit()
            # # 异步执行
            # asyncio.run(
            #     sync_function_correctly(content_list, result_list, type_name, model_name, index_name))

        return render(request, 'knowledge.html')


# 模拟一个 API POST 请求
def simulate_post_request(user, up_dataframe, project_name):
    request = HttpRequest()
    request.method = 'POST'
    request.POST = {
        'up_dataframe': up_dataframe,
        'project_name': project_name
    }
    # 模拟一个用户
    request.user = user  # 你可以设置为一个有效的用户对象，如果需要
    return request


def select(request):
    if request.method == 'GET':
        user = request.user

        cursor = conn.cursor()
        sql = """SELECT id, database_type, index_name, index_statute FROM KNOW_USER_INFO 
        WHERE username = %s AND index_name NOT LIKE '%_query_all' ORDER BY database_type"""
        cursor.execute(sql, (str(user),))
        results = cursor.fetchall()
        list_of_lists = [[item[2], item[1], item[0], item[-1]] for item in results]

        return JsonResponse({'message': list_of_lists})


def detail(request, id):
    context = {}
    if request.method == 'GET':
        if not request.user.is_authenticated:
            return render(request, 'login.html', context)

        # 创建游标对象
        cursor = conn.cursor()
        # 定义查询 SQL

        sql = """SELECT id, database_type, index_name, kg_count, get_date, project_name, know_content 
        FROM KNOW_USER_INFO WHERE username = %s and id = %s ORDER BY database_type"""

        # 执行查询
        user = request.user
        cursor.execute(sql, (str(user), id))
        results = cursor.fetchall()[0]

        context["ID"] = results[0]
        context["kg_type"] = results[1]
        context["kg_name"] = results[2]
        context["kg_count"] = results[3]
        context["kg_create_time"] = results[4].strftime("%Y-%m-%d %H:%M:%S")
        context["project_name"] = results[-2]
        context["know_content"] = results[-1]

        # 预览10条数据
        if context["kg_type"] == "ES":
            coarse_recall = CoarseRecall(
                query=None,
                coarse_size=10,
                es_name=context["kg_name"],
                milvus_name=None  # 如果不使用Milvus，可以传递None或设置默认值
            )
            Review_Recall, _ = coarse_recall.es_recall_search()
            context["kg_data_view"] = Review_Recall

        if context["kg_type"] == "Milvus":
            coarse_recall = CoarseRecall(
                query=None,
                coarse_size=10,
                es_name=None,
                milvus_name=context["kg_name"]  # 如果不使用Milvus，可以传递None或设置默认值
            )
            Review_result, _ = coarse_recall.milvus_recall_search()
            Review_result = [item for item in Review_result if item != '']
            context["kg_data_view"] = Review_result

        return render(request, "detail.html", context=context)


def get_recall(request, id):
    context = {}
    if not request.user.is_authenticated:
        return render(request, 'login.html', context)

    cursor = conn.cursor()
    user = request.user
    if ',' not in id:
        sql = """SELECT id, database_type, index_name FROM KNOW_USER_INFO  WHERE username = %s and id = %s 
        ORDER BY database_type"""
        # 执行查询
        cursor.execute(sql, (str(user), int(id)))
        results = cursor.fetchall()[0]
        if results[1] == 'ES':
            context["es_name"] = results[2]
        elif results[1] == 'Milvus':
            context["milvus_name"] = results[2]
    else:
        name = id.split(',')
        sql = """SELECT id, database_type, index_name FROM KNOW_USER_INFO  WHERE username = %s and id in (%s, %s) 
        ORDER BY database_type"""
        cursor.execute(sql, (str(user), name[0], name[-1]))
        results = cursor.fetchall()
        context["es_name"] = results[0][-1]
        context["milvus_name"] = results[1][-1]
    return render(request, "get_data.html", context=context)


def to_llm_id(request):
    if not request.user.is_authenticated:
        return render(request, 'login.html')

    cursor = conn.cursor()
    sql = ("SELECT database_type, index_name FROM KNOW_USER_INFO WHERE username = %s "
           "AND index_name NOT LIKE '%_query_all' ORDER BY database_type")
    # 执行查询
    user = request.user
    cursor.execute(sql, (str(user),))
    results = cursor.fetchall()
    database_dict = defaultdict(list)

    # 将查询结果整理到字典中
    for db_type, index_name in results:
        database_dict[db_type].append(index_name)

    # 将 defaultdict 转换为普通字典
    final_dict = dict(database_dict)
    return render(request, "llm.html", context={'final_dict': final_dict})


def delete_know(request, id):
    if not request.user.is_authenticated:
        return render(request, 'login.html')

    cursor = conn.cursor()
    query_stmt = "select index_name,database_type,embedding from KNOW_USER_INFO where id=%s"
    # 执行SQL语句
    cursor.execute(query_stmt, (id,))
    kg_config = cursor.fetchall()[0]
    index_name = kg_config[0]
    database_type = kg_config[1]

    # 检查索引是否已经存在
    if database_type == "ES":
        if es_client.indices.exists(index=index_name):
            es_client.indices.delete(index=index_name)
            # es_client.indices.delete(index=index_name + '_query_all')
    if database_type == "Milvus":
        client = MilvusClient(uri=f"http://{milvus_config['host']}:{milvus_config['port']}")
        # 检查集合是否存在于Milvus中
        try:
            # 删除集合
            client.drop_collection(index_name)
            # client.drop_collection(index_name + '_query_all')
            print(f"Collection '{index_name}' has been deleted successfully.")
        except Exception as e:
            print(f"{e}\nCollection '{index_name}' does not exist.")

    cursor = conn.cursor()
    del_stmt = "delete from KNOW_USER_INFO where id = %s"
    # 要插入的数据 user, database_type, index_name, embedding, now,num
    # 执行SQL语句
    cursor.execute(del_stmt, (id,))
    conn.commit()
    return render(request, 'knowledge.html')


def download(request):
    if not request.user.is_authenticated:
        return render(request, 'login.html')
    # 构建文件的绝对路径
    file_path = os.path.join(BASE_DIR, 'scoring', 'templates', 'upload_template.xlsx')

    if not os.path.exists(file_path):
        raise Http404("File does not exist")

    return FileResponse(open(file_path, 'rb'), as_attachment=True, filename='upload_template.xlsx')


def get_content(ids):
    if not ids:
        return []
    cursor = conn.cursor()
    # 使用参数化查询以防止SQL注入
    query = "SELECT q_a FROM questions_answers WHERE q_a_id IN (%s)" % ','.join(['%s'] * len(ids))
    cursor.execute(query, ids)
    results = cursor.fetchall()
    cursor.close()
    return [row[0] for row in results]


def llm_chat(request):
    if not request.user.is_authenticated:
        return render(request, 'login.html')

    if request.method == 'POST':
        try:
            body_str = request.body.decode('utf-8')
            data = json.loads(body_str)
            scene = data['scene']
            es_index_name = data.get('esKnowledge')
            milvus_index_name = data.get('milvusKnowledge')
            llm_type = data.get('llmModel')
            cu_coarse_size = data.get('coarseRecall')
            jg_coarse_size = data.get('fineRecall')
            query_count = data.get('rewriteCount')
            query = data.get('message')
            temperature = data.get('temperature')

            if not scene or not es_index_name or not milvus_index_name or not llm_type:
                def check():
                    yield f"data: {json.dumps({'message': '请将必填信息补充完整再进行提问！😤'})}\n\n"

                return StreamingHttpResponse(check(), content_type='application/json')

            matches = [query]
            if query_count:
                query_prompt = f"""
                    你是一个问题改写助手，根据以下参考问题，改写提出{query_count}个新的问题，你需按照如下流程工作:
                    1. 思考理解参考问题的含义。
                    2. 根据以上思考改写生成新问题，问题内容紧密围绕参考问题。问题需包含清晰的主语和提问诉求。

                    ## 注意事项
                    1. 对于生成的问题，禁止任何解释，仅使用精炼、清晰的措辞。
                    2. 根据参考问题，生成{query_count}个问题。并将生成的每个问题对、都以换行的格式进行输出。请严格按照以下格式输出:
                    问题：【生成的问题？】\n
                    问题：【生成的问题？】

                    ## 参考问题：{query}
                """
                query_llm = None
                if llm_type == '文心千帆':
                    query_llm = get_Qianfan(query_prompt)
                elif llm_type == '通义千问':
                    query_llm = get_Tongyi(query_prompt)
                elif llm_type == '讯飞星火':
                    query_llm = get_Spark(query_prompt)
                pattern = r'问题：(.+?)\？'
                match_pat = re.findall(pattern, query_llm)
                if match_pat:
                    matches.extend(match_pat)

            ES_Review_result = None
            Milvus_Review_result = None
            ranker_list = []
            for match in matches:
                coarse_recall = CoarseRecall(
                    query=match,
                    coarse_size=cu_coarse_size,
                    es_name=es_index_name,
                    milvus_name=milvus_index_name
                )
                ES_Review_result, _ = coarse_recall.es_recall_search()
                Milvus_Review_result, _ = coarse_recall.milvus_recall_search()
                # 针对通过LLM生成问题的知识库使用的代码
                # coarse_recall_all = CoarseRecall(
                #     query=match,
                #     coarse_size=cu_coarse_size,
                #     es_name=es_index_name + '_query_all',
                #     milvus_name=milvus_index_name + '_query_all'
                # )
                # _, ES_Review_query = coarse_recall_all.es_recall_search()
                # es_ids = [content['content_id'] for content in ES_Review_query]
                # es_list = get_content(es_ids)
                # _, Milvus_Review_query = coarse_recall_all.milvus_recall_search()
                # milvus_ids_q = [doc.metadata["content_id"] for doc in Milvus_Review_query]
                # milvus_list = get_content(milvus_ids_q)
                # set_res = set_result_search(query, ES_Review_result + es_list, Milvus_Review_result +
                #                             milvus_list)
                set_res = set_result_search(query, ES_Review_result, Milvus_Review_result)
                ranker_list += set_res

            if scene == '通用场景':
                ranker_result = bert_ranker_search(query, ranker_list, int(jg_coarse_size))
                res_list = []
                for part in ranker_result:
                    item = part['content']
                    if part['content'] not in res_list:
                        res_list.append(item)

                prompt_text = '\n\n'.join(res_list)
                llm_prompt = f"你是一名智能问答助手，请根据用户提出的问题，结合已知信息进行思考并解答。\n\n" \
                             f"## 知识库已知信息：\n {prompt_text} \n\n## 提出的问题：{query}\n\n### 思考过程：\n【此处请回答你给出该解答的思考过程】" \
                             f"\n### 回答：\n【此处请依据已知内容信息进行解答】"

            elif scene == '营销审计':
                ES_top_data = []
                Milvus_top_data = []
                pattern = r'【(.*?)】'
                for i, es_doc in enumerate(ES_Review_result, start=1):
                    es_recall = es_doc.replace('法规：', '').replace('\n', '').strip()
                    es_statute = es_recall.split('条文：')[0]
                    es_clause = es_recall.split('条文：')[1]
                    if es_statute and es_clause:
                        ES_top_data.append(f'【法规名称:{es_statute}】--相关条款:{es_clause}')

                for i, miv_doc in enumerate(Milvus_Review_result, start=1):
                    miv_recall = miv_doc.replace('法规：', '').replace('\n', '').strip()
                    miv_statute = miv_recall.split('条文：')[0]
                    miv_clause = miv_recall.split('条文：')[1]
                    if miv_statute and miv_clause:
                        Milvus_top_data.append(f'【法规名称:{miv_statute}】--相关条款:{miv_clause}')

                combined_list = ES_top_data + Milvus_top_data
                distinct_list = list(OrderedDict.fromkeys(combined_list))
                content_list = []
                ranker_top_data = []
                ranker_recall = {}
                # 粗召回 TOP 结果丢给 ranker
                for i, result in enumerate(distinct_list):
                    match = re.search(pattern, result)
                    if match:
                        content_in = result.split('--')[1]
                        session_in = match.group(1)
                        content_list.append(content_in)
                        ranker_recall[generate_unique_identifier(content_in.strip())] = session_in
                ranker_result = get_ranker(query, content_list, int(jg_coarse_size))

                for content in ranker_result:
                    session = ranker_recall[generate_unique_identifier(content.strip())].strip('--')
                    new_content = content.replace('法规名称', '### 法规名称').replace('相关条款', '\n#### 相关条款')
                    ranker_top_data.append(f'【{session.strip()}】--{new_content.strip()}')

                result_str = ""
                data_dict = defaultdict(list)
                for qa in ranker_top_data:
                    filtered = qa.split('】--')[0].strip('【')
                    cnt = qa.split('】--')[1]
                    if filtered:
                        data_dict[filtered].append(cnt)
                    else:
                        result_str += f"\n{cnt}\n"

                for filtered, cnt_list in data_dict.items():
                    result_str += f"\n{filtered}\n"
                    for cnt in cnt_list:
                        result_str += f"{cnt}\n"

                llm_prompt = f"你是一个营销审计专家，具备专业的营销审计法律知识，可针对用户提出的问题进行解答、并推荐出相应的法规、条款。" \
                             f"\n\n## 已知营销审计法规、相关条款：\n{result_str}\n请根据已知信息，认真阅读每部法规及相关条款，并思考每个条款适用于哪些场景，" \
                             f"对下面用户提出的问题进行解答并推荐相应的法规、条款。在推荐的过程中，请一步步思考为什么这样推荐，并给出合理的四种不同法规及相关的多个条款全部内容信息" \
                             f"\n\n## 问题：{query}\n### 解答：\n【此处请根据已知营销审计法规、相关条款信息与思考内容对问题进行解答】\n" \
                             f"### 相关法规及对应条款依据：\n【此处请参考已知营销审计法规、相关条款，选择推荐出与问题所依据的四种不同法规名称及对应的多个条款完整内容信息并以换行的方式进行回答，" \
                             f"每个推荐法规中对应的多个条款完整内容信息以无序列表的方式进行回答（需要回答出第几章第几条规定和完整的条目内容）。" \
                             f"\n## 注意：若推荐的法规重名，则将重名法规中的所有条款全部归集到一个法规下输出。" \
                             f"回答的条款必须是已知示例中存在的且内容完整不得丢失，禁止对已知示例中条款进行改写或总结再返回，禁止进行总结或推荐理由等冗余信息。】" \
                             f"\n请严格按照以上标题级别格式进行回答，先解答用户的问题再推荐出合理的四种不同法规名称及每部法规所对应的多个相关条款！"

            elif scene == '底稿推荐':
                pass

            def generate():
                full_response = ""
                if llm_type == 'qwen2:7b':
                    for char in generate_text(llm_prompt):
                        full_response += char
                        yield f"data: {json.dumps({'message': full_response})}\n\n"
                elif llm_type == '讯飞星火':
                    for char in get_Spark(llm_prompt):
                        full_response += char
                        yield f"data: {json.dumps({'message': full_response})}\n\n"
                else:
                    for chunk in llm_flow(llm_prompt, float(temperature), stream_out=True, llm_type=llm_type):
                        full_response += chunk
                        # 每次都发送完整的响应
                        yield f"data: {json.dumps({'message': full_response})}\n\n"

            return StreamingHttpResponse(generate(), content_type='application/json')
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)



def get_llm(request):
    user = request.user
    if not user.is_authenticated:
        return render(request, 'login.html')

    body_str = request.body.decode('utf-8')
    data = json.loads(body_str)
    id = data['id']
    query = data['query']

    cursor = conn.cursor()
    Review_result = None

    if ',' not in id:
        query_stmt = "select index_name,database_type,embedding from KNOW_USER_INFO where username = %s and id=%s"

        # 执行SQL语句
        cursor.execute(query_stmt, (str(user), int(id)))
        kg_config = cursor.fetchall()[0]

        index_name = kg_config[0]
        database_type = kg_config[1]

        if database_type == "ES":
            coarse_recall = CoarseRecall(
                query=query,
                coarse_size=10,
                es_name=index_name,
                milvus_name=None
            )
            Review_result = coarse_recall.es_recall_search()

        if database_type == "Milvus":
            coarse_recall = CoarseRecall(
                query=query,
                coarse_size=10,
                es_name=None,
                milvus_name=index_name
            )
            Review_result = coarse_recall.milvus_recall_search()
        prompt_text = '\n\n'.join(Review_result)

    else:
        name = id.split(',')
        query_stmt = ("select index_name,database_type,embedding from KNOW_USER_INFO where username = %s and "
                      "id in (%s, %s) AND index_name NOT LIKE '%_query_all' ORDER BY database_type")

        cursor.execute(query_stmt, (str(user), name[0], name[-1]))
        results = cursor.fetchall()
        es_index_name = results[0][0]
        milvus_index_name = results[-1][0]
        coarse_recall = CoarseRecall(
            query=query,
            coarse_size=10,
            es_name=es_index_name,
            milvus_name=milvus_index_name
        )
        ES_Review_result, _ = coarse_recall.es_recall_search()
        Milvus_Review_result, _ = coarse_recall.milvus_recall_search()
        prompt_text = '\n\n'.join(ES_Review_result + Milvus_Review_result)

    # 构建最终的 prompt
    prompt = f"您是一名智能问答助手，请根据用户提出的问题，结合并根据已知信息进行思考与解答。\n\n" \
             f"## 已知信息：\n {prompt_text} \n\n## 提出的问题：{query}\n\n### 思考过程：\n【此处请回答你给出该解答的思考过程】" \
             f"\n### 解答：【此处请依据已知内容信息进行解答】"
    llm_one_word = get_Qianfan(prompt)
    success_message = f"针对您的提问 为您回答：\n\n + {llm_one_word}"
    return JsonResponse({'message': success_message})
# ----------------------------------------------------------------------------------------------------------------------

import re
import hashlib
import pandas as pd

from django.http import JsonResponse
from ai_llm_competition.settings import conn
from django.shortcuts import render
from django.utils import timezone
from scoring.views import simulate_post_request, upload as sco_upload, CoarseRecall
from nl2sql import db_config
from ai_llm_competition import settings
from datetime import datetime


def calculate_md5(input_string):
    """计算输入字符串的MD5哈希值，并返回十六进制表示的结果"""
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    md5_result = md5_hash.hexdigest()
    return md5_result


def initialize_name_Data(file_name, name, user):
    df_table_info = pd.read_excel(file_name, sheet_name='表结构', na_filter=False)
    df_column_enum = pd.read_excel(file_name, sheet_name='枚举值', na_filter=False)
    df_wendadui = pd.read_excel(file_name, sheet_name='问答对', na_filter=False)

    # 创建表（如果表不存在）
    create_table_query = f"""
CREATE TABLE `column_info_{user}_{name}` (
  `table_en_name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL COMMENT '表英文名',
  `column_en_name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL COMMENT '字段英文名',
  `table_cn_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '表中文名',
  `column_cn_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '字段中文名',
  `description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci COMMENT '字段详细描述',
  `is_import` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '是否常用',
  `column_type` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '字段类型',
  `is_private` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL COMMENT '是否主键',
  `code_desc` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci COMMENT '标准代码',
  `sort` int DEFAULT '0' COMMENT '表内排序',
  PRIMARY KEY (`table_en_name`,`column_en_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
    """
    db_config.execute(create_table_query, db_uri=settings.db_sysuri)
    create_table_query = f"""
CREATE TABLE `测试_{user}_{name}` (
  `序号` int DEFAULT NULL,
  `业务问题` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
  `业务问题-日常措辞版本` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
  `模板` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
  `类型` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `涉及表及字段中文名` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
  `涉及表及字段英文名` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
  `涉及表数量` int DEFAULT NULL,
  `涉及输出字段数量` int DEFAULT NULL,
  `涉及条件字段数量` int DEFAULT NULL,
  `SQL` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
  `运行结果` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
  `备注` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `tables` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `human_question` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
  `md5` char(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
        """
    db_config.execute(create_table_query, db_uri=settings.db_sysuri)
    create_table_query = f"""
CREATE TABLE `枚举表_{user}_{name}` (
  `表名` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `字段名` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `标准代码（枚举值/码值）` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `含义` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
"""
    db_config.execute(create_table_query, db_uri=settings.db_sysuri)

    # 将 DataFrame 写入 MySQL 数据库
    for index, row in df_column_enum.iterrows():
        insert_query = f"""
        INSERT INTO 枚举表_{user}_{name} (表名, 字段名, `标准代码（枚举值/码值）`,含义)
        VALUES (%s,%s,%s,%s)
        """
        data_tuplr_enum = (row['表名'], row['字段名'], row['标准代码（枚举值/码值）'], row['含义'])
        db_config.execute(insert_query, data=data_tuplr_enum, db_uri=settings.db_sysuri)

    for index, row in df_table_info.iterrows():
        insert_query = f"""
        INSERT INTO column_info_{user}_{name} (table_en_name, column_en_name, table_cn_name,column_cn_name,description,is_import,column_type,is_private,code_desc,sort)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        data_tuple_column_info = (
            row['表英文名'], row['字段英文名'], row['表中文名'], row['字段中文名'], row['字段详细描述'],
            row['是否常用'], row['字段类型'], row['是否主键'],
            row['标准代码'], row['表内排序'])
        db_config.execute(insert_query, data=data_tuple_column_info, db_uri=settings.db_sysuri)

    for index, row in df_wendadui.iterrows():
        insert_query = f"""
        INSERT INTO 测试_{user}_{name} (序号, 业务问题, `业务问题-日常措辞版本`,模板,类型,涉及表数量,`SQL`,`tables`,human_question,md5)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)

       """
        data_tuple_cs_info = (
            index, row['业务问题'], row['业务问题-日常措辞版本'], row['模板'], row['类型'], row['涉及表数量'],
            row['SQL'], row['tables'],
            row['业务问题-日常措辞版本'], calculate_md5(row['业务问题-日常措辞版本']))
        db_config.execute(insert_query, data=data_tuple_cs_info, db_uri=settings.db_sysuri)

    query = f"""select human_question as chunk , md5 ,tables from 测试_{user}_{name} """
    rows = db_config.execute(query)
    df = pd.DataFrame(rows)
    return df


def init_test_data(file_name, name, user):
    # 检查数据库是否存在
    # result = db_config.execute("SHOW DATABASES LIKE %s", (f"{user}_{name}",), settings.db_uri)
    # if not result:
    #     print(f"Database {user}_{name} does not exist. Creating it...")
    #     db_config.execute(f"CREATE DATABASE {user}_{name}", data=None, db_uri=settings.db_uri)
    #     print(f"Database {user}_{name} created successfully.")
    # else:
    #     print(f"Database {user}_{name} already exists.")
    with open(file_name, 'r', encoding='utf-8') as file:
        sql_file = file.read()

    # 使用正则表达式分割 SQL 语句
    sql_statements = re.split(r';\s*$', sql_file, flags=re.MULTILINE)
    for statement in sql_statements:
        statement = statement.strip()
        if statement:
            try:
                db_config.execute(statement, data=None, db_uri=settings.db_uri)
            except Exception as e:
                print(f"Error executing statement: {statement}")
                print(e)


def mysql_add_userdata(values):
    cursor = conn.cursor()
    insert_stmt = (
        "INSERT INTO KNOW_USER_INFO (username,database_type,embedding,index_name,get_date,project_name,index_statute,"
        "know_content) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
    )
    cursor.execute(insert_stmt, values)
    conn.commit()


def upload(request):
    context = {}
    if request.method == 'GET':
        if request.user.is_authenticated:
            return render(request, 'nl2_upload.html', context)
        return render(request, 'login.html', context)

    if request.method == 'POST':
        user = request.user
        # 获取上传的文件
        file1 = request.FILES.get('file1')
        file2 = request.FILES.get('file2')
        name = request.POST.get('name')
        content = request.POST.get('content')
        # 检查是否填写了项目名
        if not name:
            error_message = '请填写项目名'
            return JsonResponse({'message': error_message})

        now = datetime.now()
        es_values = (
            str(user), 'ES', 'BM25', str(name) + '_es', now, name, '执行中', f'智能问数-{content}项目创建ES知识库')
        mysql_add_userdata(es_values)
        milvus_values = (
            str(user), 'Milvus', 'bge-base-zh', str(name) + '_milvus', now, name, '执行中',
            f'智能问数-{content}项目创建Milvus知识库')
        mysql_add_userdata(milvus_values)

        # # 检查是否已存在相同的项目名
        # if user_kg.objects.filter(name=name).exists():
        #     # context["error_message"] = '项目名已存在，请填写一个不同的项目名'
        #     error_message = '项目名已存在，请填写一个不同的项目名'
        #     return JsonResponse({'message': error_message})
        timestamp = timezone.now().strftime('%Y%m%d%H%M%S')
        file1_name = f"{timestamp}_{str(file1)}"
        file2_name = f"{timestamp}_{str(file2)}"
        handle_uploaded_file(file1, file1_name, name, user)
        handle_uploaded_file(file2, file2_name, name, user)

        # 添加上传成功的消息
        return JsonResponse({'message': '知识库录入成功'})


def handle_uploaded_file(file, file_name, name, user):
    file_path = f"./nl2sql/upload_files/{file_name}"
    try:
        with open(file_path, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)
    except Exception as e:
        print('文件上传为空', e)
    if file_name.endswith('.xlsx'):
        df = initialize_name_Data(file_path, name, user)
        request = simulate_post_request(user, df, name)
        sco_upload(request)

    if file_name.endswith('.sql'):
        init_test_data(file_path, name, user)
    return "success"


def select(request):
    user = request.user

    if request.method == 'GET':
        if not user.is_authenticated:
            return render(request, 'login.html')

        cursor = conn.cursor()
        sql = """SELECT DISTINCT project_name, index_statute, get_date FROM KNOW_USER_INFO 
        WHERE project_name is not null and username = %s"""
        cursor.execute(sql, (str(user),))
        results = cursor.fetchall()

        result_dict = {}
        for project_name, index_statute, get_date in results:
            if project_name not in result_dict:
                # 如果项目名称不在字典中，直接添加记录
                result_dict[project_name] = (index_statute, get_date)

                if index_statute == '执行中':
                    # 如果当前记录的状态是 '执行中'，优先选择
                    result_dict[project_name] = (index_statute, get_date)

        list_of_lists = [[project_name, index_statute, get_date.strftime("%Y-%m-%d %H:%M:%S")] for
                         project_name, (index_statute, get_date) in result_dict.items()]

        return JsonResponse({'message': list_of_lists})


def detail(request, name):
    if request.method == 'GET':
        context = {}
        if not request.user.is_authenticated:
            return render(request, 'login.html', context)
        # 创建游标对象
        cursor = conn.cursor()
        # 定义查询 SQL
        user = request.user
        sql = """SELECT id, project_name, database_type, index_name, kg_count, get_date, know_content 
        FROM KNOW_USER_INFO WHERE username = %s and project_name = %s ORDER BY database_type"""

        # 执行查询
        cursor.execute(sql, (str(user), name))
        results = cursor.fetchall()

        context["ID"] = f'{results[0][0]},{results[1][0]}'
        context["ID1"] = results[0][0]
        context["ID2"] = results[1][0]
        context["project_name"] = results[0][1]
        context["kg_name1"] = results[0][3]
        context["kg_name2"] = results[1][3]
        context["kg_type"] = f'{results[0][2]}, {results[1][2]}'
        context["kg_count"] = f'{results[0][4]}'
        context["kg_create_time"] = results[0][-2].strftime("%Y-%m-%d %H:%M:%S")
        context["know_content"] = f'{results[0][-1]}'

        # 预览10条数据
        coarse_recall = CoarseRecall(
            query=None,
            coarse_size=5,
            es_name=results[0][3],
            milvus_name=results[1][3]
        )
        Review_Recall, _ = coarse_recall.es_recall_search()
        Review_result, _ = coarse_recall.milvus_recall_search()
        context["kg_data_view"] = Review_Recall + Review_result

        return render(request, "nl2_detail.html", context=context)

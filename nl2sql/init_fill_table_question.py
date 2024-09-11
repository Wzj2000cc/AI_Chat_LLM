import re
import hashlib
import pandas as pd

from ai_llm_competition import settings
from nl2sql import db_config


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
    print('column_info_建表成功')
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
    print('测试_建表成功')

    create_table_query = f"""
CREATE TABLE `枚举表_{user}_{name}` (
  `表名` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `字段名` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `标准代码（枚举值/码值）` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `含义` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
"""
    db_config.execute(create_table_query, db_uri=settings.db_sysuri)
    print('枚举表_建表成功')

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
            row['表英文名'], row['字段英文名'], row['表中文名'], row['字段中文名'], row['字段详细描述'], row['是否常用'], row['字段类型'], row['是否主键'],
            row['标准代码'], row['表内排序'])
        db_config.execute(insert_query, data=data_tuple_column_info, db_uri=settings.db_sysuri)

    for index, row in df_wendadui.iterrows():
        insert_query = f"""
        INSERT INTO 测试_{user}_{name} (序号, 业务问题, `业务问题-日常措辞版本`,模板,类型,涉及表数量,`SQL`,`tables`,human_question,md5)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)

       """
        data_tuple_cs_info = (
            index, row['业务问题'], row['业务问题-日常措辞版本'], row['模板'], row['类型'], row['涉及表数量'], row['SQL'], row['tables'],
            row['业务问题-日常措辞版本'], calculate_md5(row['业务问题-日常措辞版本']))
        db_config.execute(insert_query, data=data_tuple_cs_info, db_uri=settings.db_sysuri)

    query = f"""select human_question as chunck , md5 ,tables from 测试_{user}_{name} """
    rows = db_config.execute(query)
    df = pd.DataFrame(rows)
    return df


def init_test_data(file_name, name, user):
    # 检查数据库是否存在
    result = db_config.execute("SHOW DATABASES LIKE %s", (f"{user}_{name}",), settings.db_uri)
    if not result:
        print(f"Database {user}_{name} does not exist. Creating it...")
        db_config.execute(f"CREATE DATABASE {user}_{name}", data=None, db_uri=settings.db_uri)
        print(f"Database {user}_{name} created successfully.")
    else:
        print(f"Database {user}_{name} already exists.")
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

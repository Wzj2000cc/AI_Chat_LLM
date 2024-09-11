import pymysql
import time

import pymysql.cursors
from ai_llm_competition.settings import db_sysuri

# 数据库连接缓存
databases = {}


def get_db(db_uri: str = None, force_new: bool = False) -> pymysql.connections.Connection:
    if db_uri is None:
        db_uri = db_sysuri
    conn: pymysql.connections.Connection = None
    if not force_new and db_uri in databases:
        conn = databases[db_uri]
    else:
        # 从URI解析连接参数
        db_params = parse_db_uri(db_uri)
        conn = pymysql.connect(
            host=db_params['host'],
            user=db_params['user'],
            password=db_params['password'],
            database=db_params['database'],
            port=db_params['port'],
            cursorclass=pymysql.cursors.DictCursor  # 使用 DictCursor 返回字典格式
        )
        databases[db_uri] = conn
    return conn


def parse_db_uri(db_uri: str) -> dict:
    # 假设URI格式为 mysql://user:password@host:port/database
    import urllib.parse
    result = urllib.parse.urlparse(db_uri)
    return {
        'host': result.hostname,
        'user': result.username,
        'password': result.password,
        'database': result.path[1:],
        'port': result.port
    }


def execute(sql: str, data: tuple = None, db_uri: str = None, retries: int = 3, delay: int = 2):
    conn = get_db(db_uri)
    attempt = 0
    err = None
    while attempt < retries:
        try:
            with conn.cursor() as cursor:
                # print(sql)
                if data:
                    cursor.execute(sql, data)
                else:
                    cursor.execute(sql)
                res = cursor.fetchall()
            conn.commit()
            return list(res)
            # return res
        except Exception as e:
            err = f"{e},{data}"
            attempt += 1
            time.sleep(delay)
            conn = get_db(db_uri, True)
    raise Exception(err)


# def insert(table_name: list, data: list[dict], db_uri: str = None, retries: int = 3, delay: int = 2):
#     if len(data) < 1:
#         return False
#     conn = get_db(db_uri)
#     err = None
#     columns = ', '.join(data[0].keys())
#     placeholders = ', '.join(['%s'] * len(data[0]))
#     insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
#     cursor = conn.cursor()
#     try:
#         for da in data:
#             cursor.execute(insert_query, tuple(da.values()))
#         return True
#     except Exception as e:
#         err = f"{e}"
#     raise Exception(err)

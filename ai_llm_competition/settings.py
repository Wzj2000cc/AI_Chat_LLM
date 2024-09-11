"""
Django settings for ai_llm_competition project.

Generated by 'django-admin startproject' using Django 4.1.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.1/ref/settings/
"""

from pathlib import Path
from os.path import join

import os
import mysql.connector
import time
import qianfan
import pymysql

from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
from langchain_community.llms.tongyi import Tongyi
from langchain_core.language_models.llms import BaseLLM
from langchain_community.llms import QianfanLLMEndpoint

pymysql.install_as_MySQLdb()

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-%dw8$rkzchyfv(f8u($_qc=+x)a%v*r3%(c&n=16p*h4_q0fg6'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition

INSTALLED_APPS = [
    'scoring',
    'authing',
    'nl2sql',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'ai_llm_competition.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'static/../templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'ai_llm_competition.wsgi.application'

# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases

# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.sqlite3',
#         'NAME': join(BASE_DIR, 'db.sqlite3'),
#     }
# }
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'my_database',
        'USER': 'root',
        'PASSWORD': '20001126',
        'HOST': 'localhost',
        'PORT': '3306',
        'OPTIONS': {
            'charset': 'utf8mb4',  # 设置字符集
        },
    }
}

# Password validation
# https://docs.djangoproject.com/en/4.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.authing.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.authing.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.authing.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.authing.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/4.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.1/howto/static-files/

STATIC_URL = '/static/'

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'

# Default primary key field type
# https://docs.djangoproject.com/en/4.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Elasticsearch
es_config = {
    "host": 'http://127.0.0.1:9200',
    "basic_auth": ('elastic', 'sgcc1234')
}

# Milvus
milvus_config = {
    "host": '127.0.0.1',
    "port": 19530,
    "user": "root",
    "password": ""
}

# db
conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="20001126",
    database="my_database"
)

db_uri = f'mysql+pymysql://root:20001126@127.0.0.1:3306/yangben'
db_sysuri = f'mysql+pymysql://root:20001126@127.0.0.1:3306/yangben_tableinfo'
nl2_sql_conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="20001126",
    database="yangben_tableinfo"
)

# -------------------------------------- 文心千帆 ------------------------------------------------
os.environ["QIANFAN_AK"] = "vZleY7SF2x4S3euJvXgYK1ta"
os.environ["QIANFAN_SK"] = "XnkygwUQC8OmshvkGIwkv6omG1TVyvcJ"


def get_Qianfan(question):
    resp = qianfan.ChatCompletion().do(endpoint="ernie-speed-128k", messages=[
        {"role": "user", "content": question}
    ], temperature=0.4, retry_count=5, request_timeout=120)
    return resp.body['result']


# -------------------------------------- 通义千问 ------------------------------------------------
def get_Tongyi(question):
    llm = Tongyi(
        # model_name='qwen1.5-72b-chat',
        model_name='qwen2-72b-instruct',
        # dashscope_api_key='sk-12120561a8554f319a08ad4469eb3f9a',  # 国电通1
        # dashscope_api_key='sk-f6aa7562af8a4e5287b8c2f7b04bfb5b',  # 国电通2
        dashscope_api_key='sk-79c68526c53c4e959d09b2fbd509ee99',  # 泽军key：9月18到期
        temperature=0
    )
    result = llm.invoke(question)
    return result


# -------------------------------------- 讯飞星火 ------------------------------------------------
SPARKAI_APP_ID = 'e0bab50a'
SPARKAI_API_SECRET = 'OWE4NDljOWZkODRlMDY4NTE2ODNkNmE1'
SPARKAI_API_KEY = '891a551ae4e57a2d325a211dbcf4f3eb'

SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
SPARKAI_DOMAIN = 'generalv3.5'


def get_Spark(question):
    spark = ChatSparkLLM(
        spark_api_url=SPARKAI_URL,
        spark_app_id=SPARKAI_APP_ID,
        spark_api_key=SPARKAI_API_KEY,
        spark_api_secret=SPARKAI_API_SECRET,
        spark_llm_domain=SPARKAI_DOMAIN,
        streaming=False,
    )
    messages = [ChatMessage(
        role="user",
        content=question
    )]
    handler = ChunkPrintHandler()
    result = spark.generate([messages], callbacks=[handler])
    return result.generations[0][0].text


def llm_flow(msg, tempr=0.01, stream_out=False, llm_type=None):
    err_num = 0
    err_msg = ''
    while err_num < 2:
        err_num += 1
        try:
            main_llm: BaseLLM = None
            if main_llm is None:
                if llm_type == '文心千帆':
                    main_llm = QianfanLLMEndpoint(
                        model='ERNIE-Speed-128K',
                        qianfan_ak='vZleY7SF2x4S3euJvXgYK1ta',
                        qianfan_sk='XnkygwUQC8OmshvkGIwkv6omG1TVyvcJ',
                        temperature=tempr,
                        model_kwargs={'temperature': tempr, "top_k": 2},
                    )
                elif llm_type == '通义千问':
                    main_llm = Tongyi(
                        # model_name='qwen1.5-72b-chat',
                        model_name='qwen2-72b-instruct',
                        # dashscope_api_key='sk-12120561a8554f319a08ad4469eb3f9a',  # 国电通1
                        # dashscope_api_key='sk-f6aa7562af8a4e5287b8c2f7b04bfb5b',  # 国电通2
                        dashscope_api_key='sk-79c68526c53c4e959d09b2fbd509ee99',  # 泽军key：9月18到期
                        model_kwargs={'temperature': tempr, "top_k": 2},
                    )
                if not stream_out:
                    result = main_llm(msg)
                    if result is None or result == '' or 'api' in result:
                        continue
                    return result
                else:
                    for chunk in main_llm.stream(msg):
                        yield chunk
            return  # 成功完成后退出函数
        except Exception as e:
            print('LLM error', type(e), e.args)
            time.sleep(1)
            err_msg = str(e.args)

    # 如果到达这里，说明所有尝试都失败了
    yield '模型返回异常：' + err_msg
# --------------------------------------------------------------------------------------------

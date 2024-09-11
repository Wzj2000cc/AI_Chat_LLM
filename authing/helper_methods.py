import re

from django.contrib.auth.models import User

from authing.models import *
import sys
from django.core.files.storage import FileSystemStorage
import base64
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from .apriori import operate
import plotly.express as px
import plotly.graph_objects as go
from pdb import set_trace as st

# ==================================================== #
#                  Helper Functions                    #
# ==================================================== #


def username_exist(username):
    """
    :param username: The username to be tested
    :return: A bool indicating if the username is taken
    """
    return User.objects.filter(username=username).exists()


def cosVector(a, b):#计算相似度
    a = np.array(a)
    b = np.array(b)
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def get_car_vector(car):
    brand = car.brand
    price = car.price
    distance = car.distance
    brand_enc = BRAND_ENCODE[brand]
    price_enc, distance_enc = None, None
    for enc, pair in PRICE_MAP.items():
        if pair[0] <= price < pair[1]:
            price_enc = enc
    for enc, pair in DIST_MAP.items():
        if pair[0] <= distance < pair[1]:
            distance_enc = enc
    vector = [int(brand_enc), int(price_enc), int(distance_enc)]
    return vector


# ==================================================== #
#                Recommendation Systems                #
# ==================================================== #


def recommend_for(user, method="cf"):
    if method == "cf":
        return recommend_by_collaborative_filtering(user)
    elif method == "ar":
        return recommend_by_association_rule(user)


def recommend_by_association_rule(myself):
    #判断有过浏览记录
    if myself.is_authenticated and len(myself.history_set.all()) > 0:
        my_history = [car.car_id for car in myself.history_set.all() if Car.objects.filter(id=int(car.car_id))]

        data = list()
        for user in User.objects.all():#对于每一个用户的浏览记录书写第几个用户看了什么商品
            if user.id is not None:
                for history in user.history_set.all():
                    data_line = [str(user.id), history.car_id]
                    data.append(data_line)
        data = pd.DataFrame(data=data, columns=["tid", "items"])
        #tid items
        #1(用户id)   51205(车id)
        result = operate(data)#apriori.py里的代码
        #X|Y|SUPPORT|confidence

        #进行排序
        result["score"] = result["support"] * result["confidence"]
        result = result.sort_values(by="score", ascending=False)

        recommends = set()
        #把score前八取出，不够8的自动补上
        for index, row in result.iterrows():
            if len(recommends) >= 8:
                break
            items = row["X"].split(",")
            items = [item.strip for item in items]
            if all([True if item in my_history else False for item in items]):
                recommends.add(row["Y"])

        if len(recommends) < 8:#推荐数不足
            pool = [car for car in Car.objects.all() if car.id not in recommends]
            recommends = [Car.objects.get(car_id) for car_id in recommends]
            #随机挑
            recommends.extend(list(np.random.choice(pool, 8 - len(recommends))))
        return recommends

    else:
        return np.random.choice(Car.objects.all(), 8)  # 一个浏览记录都没，随便挑


def recommend_by_collaborative_filtering(user):
    if user.is_authenticated and len(user.history_set.all()) > 0:
        user_history = user.history_set.all()
        user_history = sorted(user_history, key=lambda x: x.time, reverse=True)

        if len(user_history) > 8:
            for record in user_history[8:]:
                history_id = record.id
                History.objects.get(id=history_id).delete()
        elif len(user_history) < 8:
            return np.random.choice(Car.objects.all(), 8)

        user_history = user.history_set.all()
        history_cars = [Car.objects.get(id=int(record.car_id)) for record in user_history if Car.objects.filter(id=int(record.car_id)).exists()]

        test_vectors = list()
        for car in history_cars:
            test_vectors.append(get_car_vector(car))

        matches = dict()
        for car in Car.objects.all():
            if car not in history_cars:
                vector = get_car_vector(car)
                scores = list()
                for test_vector in test_vectors:
                    scores.append(cosVector(test_vector, vector))
                score = np.mean(scores)
                matches[car.id] = score

        matches = [k for k, v in sorted(matches.items(), key=lambda item: item[1], reverse=True)]
        recommend_ids = matches[:8]

        recommends = list()
        for car_id in recommend_ids:
            car = Car.objects.get(id=int(car_id))
            recommends.append(car)

        return recommends  # cars
    else:
        return np.random.choice(Car.objects.all(), 8)  # cars


# ==================================================== #
#                       Plotting                       #
# ==================================================== #


def plot_gender(side="sell"):#判断卖家和买家
    if side == "sell":
        brand_data = get_brand_data(side)
        x = list(brand_data.keys())

        y_male = [sum([1 if car.user.userinfo_set.all()[0].gender == "male" else 0 for car in cars])
                  for name, cars in brand_data.items()]
        y_female = [sum([1 if car.user.userinfo_set.all()[0].gender == "female" else 0 for car in cars])
                    for name, cars in brand_data.items()]

        fig = go.Figure(go.Bar(x=x, y=y_male, name='男性'))
        fig.add_trace(go.Bar(x=x, y=y_female, name='女性'))

        fig.update_layout(barmode='stack', title=dict(text='男女卖家上架不同品牌的人数数量', font=dict(size=30, color='green')),
                          xaxis=dict(title=dict(text='X-品牌名', font=dict(size=15, color='green'))),
                          yaxis=dict(title=dict(text='Y-数量', font=dict(size=15, color='green'))))
    else:
        brand_data = get_brand_data(side)
        x = list(brand_data.keys())

        y_male = [sum([1 if bought.user.userinfo_set.all()[0].gender == "male" else 0 for bought in boughts])
                  for _, boughts in brand_data.items()]
        y_female = [sum([1 if bought.user.userinfo_set.all()[0].gender == "female" else 0 for bought in boughts])
                  for _, boughts in brand_data.items()]

        fig = go.Figure(go.Bar(x=x, y=y_male, name='男性'))
        fig.add_trace(go.Bar(x=x, y=y_female, name='女性'))

        fig.update_layout(barmode='stack', title=dict(text='男女买家购买不同品牌商品数量', font=dict(size=30, color='green')),
                          xaxis=dict(title=dict(text='X-品牌名', font=dict(size=15, color='green'))),
                          yaxis=dict(title=dict(text='Y-数量', font=dict(size=15, color='green'))))
    return fig


def get_brand_data(side="sell"):
    if side == "sell":
        cars = Car.objects.all()#把所有车况导出
        brands = get_brands(cars)
        brand_data = {brand: list() for brand in brands}
        for car in cars:
            brand_data[car.brand].append(car)
        brand_data = {k: v for k, v in sorted(brand_data.items(), key=lambda item: len(item[1]), reverse=True)}
    elif side == "buy":#被买过的车统计
        ####bought表里拿
        cars = [Car.objects.get(id=int(car.car_id)) for car in Bought.objects.all() if Car.objects.filter(id=int(car.car_id)).exists()]
        brands = get_brands(cars)
        brand_data = {brand: list() for brand in brands}
        for car in cars:
            brand_data[car.brand].extend(list(Bought.objects.filter(car_id=str(car.id))))
        brand_data = {k: v for k, v in sorted(brand_data.items(), key=lambda item: len(item[1]), reverse=True)}
    return brand_data


def get_brands(cars):
    brands = list(set([car.brand for car in cars]))
    return brands


PRICE_TEXT_MAP = {
    '1': "$0~$10,000",
    '2': "$10,000~$20,000",
    '3': "$20,000~$30,000",
    '4': "$30,000~$40,000",
    '5': "$40,000~$50,000",
    '6': "$50,000~$60,000",
    '7': "$60,000~$70,000",
    '8': "$70,000~$80,000",
    '9': "$80,000+",
}


DISTANCE_TEXT_MAP = {
    '1': "0KM~55000KM",
    '2': "55000KM~70000KM",
    '3': "70000KM~80000KM",
    '4': "80000KM~90000KM",
    '5': "90000KM~100000KM",
    '6': "100000KM+",
}


PRICE_MAP = {
    '1': [0, 10000],
    '2': [10000, 20000],
    '3': [20000, 30000],
    '4': [30000, 40000],
    '5': [40000, 50000],
    '6': [50000, 60000],
    '7': [60000, 70000],
    '8': [70000, 80000],
    '9': [80000, sys.maxsize]
}


DIST_MAP = {
    '1': [0, 55000],
    '2': [55000, 70000],
    '3': [70000, 80000],
    '4': [80000, 90000],
    '5': [90000, 100000],
    '6': [100000, sys.maxsize]
}

#判断男女性别的API
GENDER_API_KEY = "YyMeQjXbAWrYmyxbYn"
GENDER_API_URL = "https://gender-api.com/get?name={}&key=YyMeQjXbAWrYmyxbYn"


BRAND_ENCODE = {
    'Subaru': 0,
    'MINI': 1,
    'Volkswagen': 2,
    'Skoda': 3,
    'Mazda': 4,
    'Renault': 5,
    'Nissan': 6,
    'LandRover': 7,
    'Maxus': 8,
    'Hyundai': 9,
    'Ford': 10,
    'Mitsubishi': 11,
    'Isuzu': 12,
    'Lamborghini': 13,
    'Fiat': 14,
    'Peugeot': 15,
    'Mercedes-Benz': 16,
    'Ssangyong': 17,
    'Haval': 18,
    'Chery': 19,
    'Citroen': 20,
    'Chevrolet': 21,
    'AlfaRomeo': 22,
    'Porsche': 23,
    'Toyota': 24,
    'Proton': 25,
    'AstonMartin': 26,
    'Lexus': 27,
    'Honda': 28,
    'Kia': 29,
    'BMW': 30,
    'Perodua': 31,
    'Smart': 32,
    'Jaguar': 33,
    'Bentley': 34,
    'Suzuki': 35,
    'Volvo': 36,
    'Jeep': 37,
    'Audi': 38,
    'Naza': 39
}
from django.contrib.auth.models import User
from django.shortcuts import render
from django.http import JsonResponse
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib import auth
from authing.helper_methods import *
from datetime import timedelta


def register(request):
    context = {}

    if request.method == 'GET':
        return render(request, 'registration.html', context)

    if request.method == 'POST':

        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        # User object here is default by Django
        if username_exist(username):  # 判断user是否存在
            context["user_already_exist"] = True
            return render(request, "registration.html", context)

        # 密码是否一致
        if password == confirm_password and password != "":
            User.objects.create_user(username=username, email=email, password=password)

            return render(request, "login.html", context)
        else:
            context["psw_not_match"] = True
            return render(request, "registration.html", context)


def login(request):
    """
    :param request: request received
    :return: http response about logging in
    """
    context = {}

    if request.method == 'GET':
        if request.user.is_authenticated:
            return HttpResponseRedirect('/scoring/upload/')
        return render(request, 'login.html', context)

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if not username_exist(username):  # 用户不存在
            context["user_not_exist"] = True
            return render(request, 'login.html', context)

        # 判断用户名和密码, if valid, return a User object
        user = auth.authenticate(username=username, password=password)
        if not user:
            user = User.objects.get(username=username)
            if user.password != password:
                user = None
        if user:
            auth.login(request, user)
            return HttpResponseRedirect('/scoring/upload/')
        else:
            context["psw_not_match"] = True
            return render(request, 'login.html', context)


def profile(request):
    context = {}
    user = request.user

    if not user.is_authenticated:
        return HttpResponseRedirect('/authing/login/')

    user_info = User.objects.get(username=user)

    context["id"] = user_info.id
    context["username"] = user
    context["password"] = user_info.password
    context['email'] = user_info.email
    date_joined = user_info.date_joined + timedelta(hours=8)
    context["date_joined"] = date_joined.strftime("%Y-%m-%d %H:%M:%S")
    last_login = user_info.last_login + timedelta(hours=8)
    context["last_login"] = last_login.strftime("%Y-%m-%d %H:%M:%S")

    if request.method == 'GET':
        return render(request, 'profile.html', context)

    if request.method == 'POST':
        name = request.POST.get('username')
        email = request.POST.get('email')
        if user.username != name:
            if User.objects.filter(username=name).__len__() >= 2:  # 搜索出USER list用户已经存在
                context["user_already_exist"] = True
                return render(request, 'profile.html', context)
            else:
                user.username = name
        user_info.email = email
        user_info.username = name
        user_info.save()
        auth.logout(request)
        return HttpResponseRedirect('/authing/login/')
        # return HttpResponseRedirect('/scoring/upload/')


def logout(request):
    auth.logout(request)
    return HttpResponseRedirect('/authing/login/')

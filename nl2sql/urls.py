from django.urls import path

from nl2sql import views

urlpatterns = [
    path('nl2_detail/<str:name>/', views.detail, name='detail'),
    path('nl2_select/', views.select, name='select'),
    path('nl2_upload/', views.upload, name='upload')
]

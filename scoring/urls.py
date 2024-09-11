from django.urls import path

from . import views

urlpatterns = [
    path('upload/', views.upload, name='upload'),
    path('recall/', views.recall, name='recall'),
    path('select/', views.select, name='select'),
    path('detail/<int:id>/', views.detail, name='detail'),
    path('get_recall/<str:id>/', views.get_recall, name='get_recall'),
    path('to_llm_id/', views.to_llm_id, name='to_llm_id'),
    path('get_llm/', views.get_llm, name='get_llm'),
    path('delete_know/<int:id>/', views.delete_know, name='delete_know'),
    path('download/', views.download, name='download'),
    path('llm_chat/', views.llm_chat, name='llm_chat'),
]

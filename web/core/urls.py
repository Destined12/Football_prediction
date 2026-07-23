from django.contrib import admin
from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('model-info/', views.model_info, name='model_info'),
    path('dataset-info/', views.dataset_info, name='dataset_info'),
    path('methodology/', views.methodology, name='methodology'),
]

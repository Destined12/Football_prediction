from django.contrib import admin
from django.urls import path
from . import views

app_name = 'dataset'

urlpatterns = [
    path('', views.dataset_detail, name='dataset_detail'),
]

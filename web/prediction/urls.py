from django.contrib import admin
from django.urls import path
from . import views

app_name = 'prediction'

urlpatterns = [
    path('', views.predict, name='predict'),
    path('history/', views.history, name='history'),
    path('api/predict/', views.api_predict, name='api_predict'),
]

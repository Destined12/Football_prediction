from django.contrib import admin
from django.urls import path
from . import views

app_name = 'ml_engine'

urlpatterns = [
    path('features/', views.features, name='features'),
]

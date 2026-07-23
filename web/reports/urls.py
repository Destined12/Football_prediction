from django.contrib import admin
from django.urls import path
from . import views

app_name = 'reports'

urlpatterns = [
    path('', views.index, name='index'),
    path('download/<str:report_type>/<str:fmt>/', views.download_report, name='download_report'),
]

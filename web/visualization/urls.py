from django.contrib import admin
from django.urls import path
from . import views

app_name = 'visualization'

urlpatterns = [
    path('figures/', views.figures_gallery, name='figures_gallery'),
    path('figures/<str:category>/<str:filename>/', views.serve_figure, name='serve_figure'),
]

from django.contrib import admin
from .models import DatasetInfo


@admin.register(DatasetInfo)
class DatasetInfoAdmin(admin.ModelAdmin):
    list_display = ('filename', 'season', 'rows', 'columns', 'uploaded_at')
    list_filter = ('season',)

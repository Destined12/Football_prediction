from django.contrib import admin
from .models import PredictionHistory


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ('home_team', 'away_team', 'predicted_label', 'confidence', 'created_at')
    list_filter = ('predicted_result', 'created_at')
    search_fields = ('home_team', 'away_team')
    readonly_fields = ('created_at',)

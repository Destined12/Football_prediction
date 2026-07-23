from django.contrib import admin
from .models import ModelMetrics, FeatureInfo


@admin.register(ModelMetrics)
class ModelMetricsAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc', 'evaluated_at')
    list_filter = ('model_name',)


@admin.register(FeatureInfo)
class FeatureInfoAdmin(admin.ModelAdmin):
    list_display = ('name', 'feature_group', 'importance_rank', 'shap_importance')
    list_filter = ('feature_group',)
    search_fields = ('name',)

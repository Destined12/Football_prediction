from django.db import models


class ModelMetrics(models.Model):
    model_name = models.CharField(max_length=100)
    accuracy = models.FloatField()
    precision_macro = models.FloatField()
    recall_macro = models.FloatField()
    f1_macro = models.FloatField()
    roc_auc = models.FloatField()
    log_loss_value = models.FloatField()
    balanced_accuracy = models.FloatField()
    evaluated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name_plural = 'Model Metrics'

    def __str__(self):
        return f"{self.model_name} - Acc: {self.accuracy:.4f}"


class FeatureInfo(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    feature_group = models.CharField(max_length=100)
    importance_rank = models.IntegerField(null=True, blank=True)
    shap_importance = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['importance_rank']

    def __str__(self):
        return self.name

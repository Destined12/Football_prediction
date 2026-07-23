from django.db import models


class PredictionHistory(models.Model):
    RESULT_CHOICES = [
        ('H', 'Home Win'),
        ('D', 'Draw'),
        ('A', 'Away Win'),
    ]

    home_team = models.CharField(max_length=100)
    away_team = models.CharField(max_length=100)
    predicted_result = models.CharField(max_length=1, choices=RESULT_CHOICES)
    predicted_label = models.CharField(max_length=20)
    confidence = models.FloatField()
    prob_home = models.FloatField()
    prob_draw = models.FloatField()
    prob_away = models.FloatField()
    shap_features = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Prediction History'

    def __str__(self):
        return f"{self.home_team} vs {self.away_team} -> {self.predicted_label}"

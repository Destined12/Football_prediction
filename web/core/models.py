from django.db import models


class SiteConfiguration(models.Model):
    site_name = models.CharField(max_length=200, default='Anointing Prediction System')
    site_description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Site Configuration'
        verbose_name_plural = 'Site Configuration'

    def __str__(self):
        return self.site_name

from django.db import models


class DatasetInfo(models.Model):
    filename = models.CharField(max_length=200)
    season = models.CharField(max_length=20)
    rows = models.IntegerField()
    columns = models.IntegerField()
    date_range_start = models.DateField(null=True, blank=True)
    date_range_end = models.DateField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name_plural = 'Dataset Info'

    def __str__(self):
        return f"{self.season} ({self.filename})"

from django.db import models

# Create your models here.

class Prediction(models.Model):
    symptom = models.CharField(max_length=200)
    nature_of_disease = models.CharField(max_length=200)
    sex = models.CharField(max_length=200)

from rest_framework import serializers

from predictions.models import Prediction


class PredictionSerializer(serializers.Serializer):
    symptoms = serializers.CharField(max_length=100)
    nature_of_disease = serializers.CharField(max_length=100)
    sex = serializers.CharField(max_length=100)


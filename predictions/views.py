from django.shortcuts import render
from rest_framework.decorators import api_view

from predictions.disease_prediction_notebook import predicted_disease
from predictions.serializer import PredictionSerializer
from rest_framework.response import Response
from rest_framework import status


# Create your views here.
@api_view(['POST'])
def get_disease(request):
    if request.method == 'POST':
        serializer = PredictionSerializer(data=request.data)
        if serializer.is_valid():
            symptoms = request.data['symptoms']
            nature_of_disease = request.data['nature_of_disease']
            sex = request.data['sex']
            disease = predicted_disease(symptoms, nature_of_disease, sex)
            return Response({"disease": disease}, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

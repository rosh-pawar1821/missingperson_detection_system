from rest_framework import serializers
from .models import Missingperson, Detectionmodelresult

class MissingPersonSerializer(serializers.ModelSerializer):
    class Meta:
        model = Missingperson
        fields = ["id", "name", "date_repo", "description", "image"] 

class DetectionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model=Detectionmodelresult
        fields="__all__"
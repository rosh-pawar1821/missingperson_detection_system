from django.urls import path
from . import views
from .models import Missingperson, Detectionmodelresult

urlpatterns = [
    path('', views.upload_person_form, name='upload_person_form'),
    path('upload_person/', views.upload_person_form, name='upload_person'),
    path('detect_video/', views.detect_video_form, name='detect_video_form'),
    path('results/', views.results_view, name='results_view'),
    path('upload-missing-person/', views.upload_missing_person, name='upload_missing_person'),
    path('detect_in_video/', views.detect_in_video, name='detect_in_video'),
    path('detection_result/<int:detection_id>/', views.get_detection_result, name='detection_result_by_id'),
    path('detection_result/', views.get_detection_result, name='detection_result'),
]
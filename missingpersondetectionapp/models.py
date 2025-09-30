from django.db import models
import json
from django.utils import timezone

class Missingperson(models.Model):
    name=models.CharField(max_length=50)
    image=models.ImageField(upload_to='missing_person/')
    description=models.TextField(blank=True)
    date_repo=models.DateField(auto_now_add=True)
    embedding = models.BinaryField(null=True, blank=True) 

    def __str__(self):
        return self.name if self.name else f"Missing Person #{self.id}"



class Detectionmodelresult(models.Model):
    video_filename = models.CharField(max_length=255,default="unknown_video.mp4")
    missing_person_name = models.CharField(max_length=255,default="Unknown")
    found = models.BooleanField(default=False)
    frames_detected = models.TextField(default="[]")  
    total_frames = models.IntegerField(default=0)
    created_at = models.DateTimeField(default=timezone.now)

    def set_frames(self, frames):
        self.frames_detected = json.dumps(frames)

    def get_frames(self):
        return json.loads(self.frames_detected)

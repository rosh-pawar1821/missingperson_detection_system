from django.contrib import admin
from .models import Missingperson, Detectionmodelresult
from django.utils.html import format_html

@admin.register(Missingperson)
class MissingpersonAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "date_repo", "description", "image_preview")
    search_fields = ("name",)
    list_filter = ("date_repo",)
    ordering = ("-date_repo",)

    def image_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" width="60" height="60" style="object-fit:cover;border-radius:5px;" />',
                obj.image.url,
            )
        return "No Image"

    image_preview.short_description = "Image"



@admin.register(Detectionmodelresult)
class DetectionmodelresultAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "video_filename",
        "missing_person_name",
        "found",
        "total_frames",
        "created_at",
    )
    ordering = ("-created_at",)  
    list_filter = ("found", "created_at")
    search_fields = ("video_filename", "missing_person_name")


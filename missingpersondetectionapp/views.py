import os
import cv2
import tempfile
import pickle
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from django.shortcuts import render, redirect
from django.contrib import messages
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from .models import Missingperson, Detectionmodelresult
from .serializers import MissingPersonSerializer, DetectionResultSerializer


# ----------------------------------------------------------------
# GLOBAL LOADERS (Loaded Once)
# ----------------------------------------------------------------
detector = None
embedder = None
model = None
label_encoder = None


def get_face_detector_and_embedder():
    """
    Lazy-load MTCNN, FaceNet, and trained classifier models.
    """
    global detector, embedder, model, label_encoder

    if detector is None:
        detector = MTCNN()
        print("✅ MTCNN face detector loaded.")

    if embedder is None:
        embedder = FaceNet()
        print("✅ FaceNet embedder loaded.")

    if model is None or label_encoder is None:
        model_path = os.path.join("model", "face_detection_model.pkl")
        label_path = os.path.join("model", "label_encoder.pkl")

        if not os.path.exists(model_path) or not os.path.exists(label_path):
            raise FileNotFoundError("Model or label encoder file missing in /model directory.")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(label_path, "rb") as f:
            label_encoder = pickle.load(f)

        print("✅ Face recognition model and label encoder loaded.")

    return detector, embedder, model, label_encoder


# ----------------------------------------------------------------
# HELPER FUNCTION
# ----------------------------------------------------------------
def extract_and_process_faces(video_path, missing_person_name, frame_skip=10):
    """
    Process video frames and detect if missing person appears.
    """
    detector, embedder, model, label_encoder = get_face_detector_and_embedder()
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    found_frames = []

    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            results = detector.detect_faces(frame)
            for r in results:
                x, y, w, h = r["box"]
                x, y = abs(x), abs(y)
                face = frame[y:y + h, x:x + w]
                if face.size == 0:
                    continue

                face_resized = cv2.resize(face, (160, 160))
                embedding = embedder.embeddings([face_resized])[0].reshape(1, -1)

                pred = model.predict(embedding)
                detected_name = label_encoder.inverse_transform(pred)[0]

                if detected_name.lower() == missing_person_name.lower():
                    found_frames.append(frame_number)
                    print(f"🎯 FOUND {missing_person_name} at frame {frame_number}")

        frame_number += 1

    cap.release()
    return {
        "found": len(found_frames) > 0,
        "frames": found_frames,
        "total_frames_processed": frame_number
    }


# ----------------------------------------------------------------
# API: Upload Missing Person
# ----------------------------------------------------------------
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_missing_person(request):
    try:
        serializer = MissingPersonSerializer(data=request.data)
        if serializer.is_valid():
            # Validate image
            for _, file_obj in request.FILES.items():
                if not file_obj.name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    return Response(
                        {"error": "Invalid image format. Use JPG, JPEG, PNG, or BMP."},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                if file_obj.size > 5 * 1024 * 1024:
                    return Response(
                        {"error": "File too large. Max 5 MB."},
                        status=status.HTTP_400_BAD_REQUEST
                    )

            person = serializer.save()
            return Response(MissingPersonSerializer(person).data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ----------------------------------------------------------------
# API: Detect Missing Person in Video
# ----------------------------------------------------------------
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def detect_in_video(request):
    try:
        if "video" not in request.FILES:
            return Response({"error": "No video file provided."}, status=status.HTTP_400_BAD_REQUEST)

        video_file = request.FILES["video"]
        missing_person_name = request.data.get("missing_person_name")
        if not missing_person_name:
            return Response({"error": "Missing person name is required."}, status=status.HTTP_400_BAD_REQUEST)

        if not video_file.name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            return Response({"error": "Invalid video format."}, status=status.HTTP_400_BAD_REQUEST)

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as temp_file:
            for chunk in video_file.chunks():
                temp_file.write(chunk)
            temp_video_path = temp_file.name

        try:
            detection_results = extract_and_process_faces(temp_video_path, missing_person_name)

            record = Detectionmodelresult.objects.create(
                video_filename=video_file.name,
                missing_person_name=missing_person_name,
                found=detection_results["found"],
                frames_detected=detection_results["frames"],
                total_frames=detection_results["total_frames_processed"]
            )

            return Response({
                "message": "Video processed successfully",
                "detection_id": record.id,
                "found": detection_results["found"],
                "frames_found": detection_results["frames"],
                "total_frames_processed": detection_results["total_frames_processed"],
                "missing_person_name": missing_person_name
            }, status=status.HTTP_200_OK)

        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

    except Exception as e:
        return Response({"error": f"Error processing video: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ----------------------------------------------------------------
# API: Get Detection Results
# ----------------------------------------------------------------
@api_view(["GET"])
def get_detection_result(request, detection_id=None):
    try:
        if detection_id:
            detection = Detectionmodelresult.objects.get(id=detection_id)
            serializer = DetectionResultSerializer(detection)
            return Response(serializer.data, status=status.HTTP_200_OK)

        name = request.query_params.get("name")
        if name:
            detections = Detectionmodelresult.objects.filter(missing_person_name__iexact=name)
            if not detections.exists():
                return Response({"error": "No results found for this name"}, status=status.HTTP_404_NOT_FOUND)
            serializer = DetectionResultSerializer(detections, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)

        detections = Detectionmodelresult.objects.all()
        serializer = DetectionResultSerializer(detections, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    except Detectionmodelresult.DoesNotExist:
        return Response({"error": "Detection not found"}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ----------------------------------------------------------------
# DJANGO FORM VIEWS (for templates)
# ----------------------------------------------------------------
def upload_person_form(request):
    if request.method == "POST":
        name = request.POST.get("name")
        image = request.FILES.get("image")
        if name and image:
            Missingperson.objects.create(name=name, image=image)
            messages.success(request, f"✅ Missing person {name} uploaded successfully.")
            return redirect("upload_person_form")
    return render(request, "upload_person.html")


def detect_video_form(request):
    if request.method == "POST":
        response = detect_in_video(request)
        if isinstance(response, Response) and response.status_code == 200:
            messages.success(request, "🎥 Video processed successfully!")
            return redirect("results_view")
        else:
            messages.error(request, f"Error: {getattr(response, 'data', 'Unknown error')}")
    return render(request, "detect_video.html")


def results_view(request):
    results = Detectionmodelresult.objects.all().order_by("-created_at")
    return render(request, "results.html", {"results": results})


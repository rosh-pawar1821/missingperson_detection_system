import os
import cv2
import tempfile
import pickle
import joblib
import numpy as np
import logging

from django.shortcuts import render, redirect
from django.contrib import messages
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response

from .models import Missingperson, Detectionmodelresult
from .serializers import MissingPersonSerializer, DetectionResultSerializer

logger = logging.getLogger(__name__)


detector = None
embedder = None
model = None
label_encoder = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_FILE = os.path.join(MODEL_DIR, "face_detection_model.pkl")
VOTING_CLASSIFIER_FILE = os.path.join(MODEL_DIR, "voting_classifier.pkl")
LABEL_ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")


def get_face_detector_and_embedder():
    """
    Lazy-loads heavy ML components only when needed.
    Attempts joblib.load for classifier first (common for sklearn),
    falls back to pickle if joblib fails.
    """
    global detector, embedder, model, label_encoder
    
    if detector is None or embedder is None:
        try:
            # import inside function to avoid importing TensorFlow at module import time
            from mtcnn import MTCNN
            from keras_facenet import FaceNet
        except Exception as e:
            logger.exception("Failed to import MTCNN/FaceNet: %s", e)
            raise RuntimeError(f"Failed to import face detection/embedding libraries: {e}")

        detector = MTCNN()
        embedder = FaceNet()
        logger.info("Loaded MTCNN detector and FaceNet embedder.")

    # load classifier and label encoder lazily
    if model is None or label_encoder is None:
        # prefer an explicit voting_classifier if present
        # try joblib (common) then pickle fallback
        tried = []
        if os.path.exists(VOTING_CLASSIFIER_FILE):
            try:
                model = joblib.load(VOTING_CLASSIFIER_FILE)
                logger.info("Loaded model from %s (joblib).", VOTING_CLASSIFIER_FILE)
            except Exception as e:
                tried.append(("joblib_voting", str(e)))
                model = None

        if model is None and os.path.exists(MODEL_FILE):
            try:
                # try joblib first
                model = joblib.load(MODEL_FILE)
                logger.info("Loaded model from %s (joblib).", MODEL_FILE)
            except Exception as e_joblib:
                tried.append(("joblib_model", str(e_joblib)))
                # fallback to pickle
                try:
                    with open(MODEL_FILE, "rb") as f:
                        model = pickle.load(f)
                    logger.info("Loaded model from %s (pickle fallback).", MODEL_FILE)
                except Exception as e_pickle:
                    tried.append(("pickle_model", str(e_pickle)))
                    model = None

        if model is None:
            logger.error("Failed to load model. attempts: %s", tried)
            # Do not raise here if you want to allow endpoints that don't need model,
            # but in your flow detection requires model so raise to surface the problem.
            raise FileNotFoundError(f"Failed to load model from {VOTING_CLASSIFIER_FILE} or {MODEL_FILE}. Details: {tried}")

        # load label encoder (pickle)
        if os.path.exists(LABEL_ENCODER_FILE):
            try:
                with open(LABEL_ENCODER_FILE, "rb") as f:
                    label_encoder = pickle.load(f)
                logger.info("Loaded label encoder from %s.", LABEL_ENCODER_FILE)
            except Exception as e:
                logger.exception("Failed to load label encoder: %s", e)
                raise RuntimeError(f"Failed to load label encoder: {e}")
        else:
            raise FileNotFoundError(f"Label encoder not found at {LABEL_ENCODER_FILE}")

    return detector, embedder, model, label_encoder


def get_embedding(face_img, embedder):
    """
    Resizes to 160x160, converts to float32 and runs FaceNet embedder.
    """
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype("float32")
    embedding = embedder.embeddings([face_img])[0]
    return embedding

def extract_and_process_faces(video_path, missing_person_name, frame_skip=10):
    """
    Opens video, iterates frames (skipping by frame_skip), detects faces,
    computes embedding and predicts with model. Returns structured dict.
    """
    # Ensure models are loaded
    detector_local, embedder_local, model_local, label_encoder_local = get_face_detector_and_embedder()
    if model_local is None or label_encoder_local is None:
        raise RuntimeError("Classification model or label encoder not loaded.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video file for processing.")

    frame_number = 0
    found_frames = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_skip != 0:
                frame_number += 1
                continue

            # detect faces
            try:
                results = detector_local.detect_faces(frame)
            except Exception as e:
                logger.exception("Error running detector.detect_faces: %s", e)
                results = []

            for r in results:
                # MTCNN returns 'box' possibly with negative coords
                try:
                    x, y, w, h = r.get("box", (0, 0, 0, 0))
                except Exception:
                    # unexpected format -> skip
                    continue

                x = max(0, int(x))
                y = max(0, int(y))
                w = max(0, int(w))
                h = max(0, int(h))

                # clamp to frame bounds
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)

                # avoid empty crops
                if x2 <= x or y2 <= y:
                    continue

                face = frame[y:y2, x:x2]
                if face.size == 0:
                    continue

                # embedding + predict
                try:
                    embedding = get_embedding(face, embedder_local).reshape(1, -1)
                    pred = model_local.predict(embedding)
                    detected_name = label_encoder_local.inverse_transform(pred)[0]
                except Exception as e:
                    logger.exception("Error embedding/predicting face: %s", e)
                    continue

                if detected_name == missing_person_name:
                    found_frames.append(frame_number)
                    logger.info("FOUND %s at frame %s", missing_person_name, frame_number)

            frame_number += 1

    finally:
        # always release capture
        cap.release()

    return {
        "found": len(found_frames) > 0,
        "frames": found_frames,
        "total_frames_processed": frame_number,
    }

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_missing_person(request):
    """
    Save missing person (image + name) using serializer validation.
    """
    try:
        serializer = MissingPersonSerializer(data=request.data)
        if serializer.is_valid():
            # validate file(s)
            if request.FILES:
                for field_name, file_obj in request.FILES.items():
                    allowed_formats = [".jpg", ".jpeg", ".png", ".bmp"]
                    if not any(file_obj.name.lower().endswith(fmt) for fmt in allowed_formats):
                        return Response(
                            {"error": f"Invalid image format for {field_name}. Supported: jpeg, jpg, png, bmp"},
                            status=status.HTTP_400_BAD_REQUEST,
                        )
                    if file_obj.size > 5 * 1024 * 1024:
                        return Response(
                            {"error": f"Image too large for {field_name}. Max size: 5MB"},
                            status=status.HTTP_400_BAD_REQUEST,
                        )
            person = serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        logger.exception("Error in upload_missing_person: %s", e)
        return Response({"error": f"Error uploading missing person: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(["POST"])
def detect_in_video(request):
    """
    Accepts video file and missing_person_name, processes video and saves Detectionmodelresult.
    """
    try:
        if "video" not in request.FILES:
            return Response({"error": "No video file provided"}, status=status.HTTP_400_BAD_REQUEST)

        video_file = request.FILES["video"]
        if not video_file.name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            return Response({"error": "Invalid video format"}, status=status.HTTP_400_BAD_REQUEST)

        missing_person_name = request.data.get("missing_person_name")
        if not missing_person_name:
            return Response({"error": "Missing person name is required"}, status=status.HTTP_400_BAD_REQUEST)

        # save temp video
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp:
            for chunk in video_file.chunks():
                tmp.write(chunk)
            temp_video_path = tmp.name

        try:
            # process
            detection_results = extract_and_process_faces(temp_video_path, missing_person_name, frame_skip=10)

            detection_record = Detectionmodelresult.objects.create(
                video_filename=video_file.name,
                missing_person_name=missing_person_name,
                found=detection_results["found"],
                frames_detected=detection_results["frames"],
                total_frames=detection_results["total_frames_processed"],
            )

            return Response(
                {
                    "message": "Video processed successfully",
                    "detection_id": detection_record.id,
                    "missing_person_name": missing_person_name,
                    "found": detection_results["found"],
                    "frames_found": detection_results["frames"],
                    "total_frames_processed": detection_results["total_frames_processed"],
                },
                status=status.HTTP_200_OK,
            )
        finally:
            # cleanup temp file
            try:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
            except Exception as e:
                logger.warning("Failed to remove temp video: %s", e)

    except Exception as e:
        logger.exception("Error in detect_in_video: %s", e)
        return Response({"error": f"Error processing video: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(["GET"])
def get_detection_result(request, detection_id=None):
    try:
        if detection_id:
            detection = Detectionmodelresult.objects.get(id=detection_id)
            serializer = DetectionResultSerializer(detection)
            return Response(serializer.data, status=status.HTTP_200_OK)

        name = request.query_params.get("name")
        if name:
            detections = Detectionmodelresult.objects.filter(missing_person_name=name)
            if not detections.exists():
                return Response({"error": "No results found for this name"}, status=status.HTTP_404_NOT_FOUND)
            serializer = DetectionResultSerializer(detections, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)

        detections = Detectionmodelresult.objects.all()
        serializer = DetectionResultSerializer(detections, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    except Detectionmodelresult.DoesNotExist:
        return Response({"error": "Result not found"}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.exception("Error in get_detection_result: %s", e)
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def upload_person_form(request):
    if request.method == "POST":
        name = request.POST.get("name")
        image = request.FILES.get("image")
        if name and image:
            Missingperson.objects.create(name=name, image=image)
            messages.success(request, f"Missing person {name} uploaded successfully!")
            return redirect("upload_person_form")
    return render(request, "upload_person.html")


def detect_video_form(request):
    if request.method == "POST":
        # use the same view function to process
        response = detect_in_video(request)
        # response is a DRF Response object
        if getattr(response, "status_code", None) == 200:
            messages.success(request, "Video processed successfully!")
            return redirect("results_view")
        else:
            # read data safely
            data = getattr(response, "data", {"error": "unknown"})
            messages.error(request, f"Error: {data}")
    return render(request, "detect_video.html")


def results_view(request):
    results = Detectionmodelresult.objects.all().order_by("-created_at")
    return render(request, "results.html", {"results": results})





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
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

detector = None
embedder = None
model = None
label_encoder = None

def get_face_detector_and_embedder():
    global detector, embedder, model, label_encoder
    if detector is None or embedder is None:
        # Lazy import
        from mtcnn import MTCNN
        from keras_facenet import FaceNet
        detector = MTCNN()
        embedder = FaceNet()
        print("Detector and embedder loaded.")

    if model is None or label_encoder is None:
        import pickle
        with open("model/face_detection_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        print("Trained model and label encoder loaded.")

    return detector, embedder, model, label_encoder


def get_embedding(face_img, embedder):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype('float32')
    embedding = embedder.embeddings([face_img])[0]
    return embedding

def save_video_temporarily(video_file):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, video_file.name)
    with open(temp_path, "wb+") as destination:
        for chunk in video_file.chunks():
            destination.write(chunk)
    return temp_path

def cleanup_temp_file(file_path):
    try:
        os.unlink(file_path)
        temp_dir = os.path.dirname(file_path)
        os.rmdir(temp_dir)
    except OSError:
        pass


def extract_and_process_faces(video_path, missing_person_name, frame_skip=10):
    detector, embedder, model, label_encoder = get_face_detector_and_embedder()
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    found_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            results = detector.detect_faces(frame)
            for r in results:
                x, y, w, h = r['box']
                x, y = max(0, x), max(0, y)
                face = frame[y:y+h, x:x+w]
                if face.size == 0:
                    continue

                embedding = get_embedding(face, embedder).reshape(1, -1)
                pred = model.predict(embedding)
                detected_name = label_encoder.inverse_transform(pred)[0]

                if detected_name == missing_person_name:
                    found_frames.append(frame_number)
                    logger.info(f" FOUND {missing_person_name} at frame {frame_number}")

        frame_number += 1

    cap.release()
    return {
        'found': len(found_frames) > 0,
        'frames': found_frames,
        'total_frames_processed': frame_number
    }

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_missing_person(request):
    try:
        serializer = MissingPersonSerializer(data=request.data)
        if serializer.is_valid():
            if request.FILES:
                for field_name, file_obj in request.FILES.items():
                    allowed_formats = ['.jpg', '.jpeg', '.png', '.bmp']
                    if not any(file_obj.name.lower().endswith(fmt) for fmt in allowed_formats):
                        return Response(
                            {'error': f'Invalid image format for {field_name}. Supported: jpeg, jpg, png, bmp'},
                            status=status.HTTP_400_BAD_REQUEST
                        )
                    max_size = 5 * 1024 * 1024
                    if file_obj.size > max_size:
                        return Response(
                            {'error': f'Image file too large for {field_name}. Maximum size: 5MB'},
                            status=status.HTTP_400_BAD_REQUEST
                        )

            person = serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.error(f"Error uploading missing person: {str(e)}")
        return Response({'error': f'Error uploading missing person: {str(e)}'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def detect_in_video(request):
    try:
        if 'video' not in request.FILES:
            return Response({'error': 'No video file provided'}, status=status.HTTP_400_BAD_REQUEST)

        video_file = request.FILES['video']
        if not video_file.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return Response({'error': 'Invalid video format. Supported: .mp4, .avi, .mov, .mkv'},
                            status=status.HTTP_400_BAD_REQUEST)

        missing_person_name = request.data.get('missing_person_name')
        if not missing_person_name:
            return Response({'error': 'Missing person name is required'}, status=status.HTTP_400_BAD_REQUEST)

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as temp_file:
            for chunk in video_file.chunks():
                temp_file.write(chunk)
            temp_video_path = temp_file.name

        try:
            detection_results = extract_and_process_faces(temp_video_path, missing_person_name, frame_skip=10)

            detection_record = Detectionmodelresult.objects.create(
                video_filename=video_file.name,
                missing_person_name=missing_person_name,
                found=detection_results['found'],
                frames_detected=detection_results['frames'],
                total_frames=detection_results['total_frames_processed']
            )

            return Response({
                'message': 'Video processed successfully',
                'detection_id': detection_record.id,
                'missing_person_name': missing_person_name,
                'found': detection_results['found'],
                'frames_found': detection_results['frames'],
                'total_frames_processed': detection_results['total_frames_processed']
            }, status=status.HTTP_200_OK)

        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return Response({'error': f'Error processing video: {str(e)}'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
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
                return Response({'error': 'No results found for this name'}, status=status.HTTP_404_NOT_FOUND)
            serializer = DetectionResultSerializer(detections, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)

        detections = Detectionmodelresult.objects.all()
        serializer = DetectionResultSerializer(detections, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    except Detectionmodelresult.DoesNotExist:
        return Response({'error': 'Result not found'}, status=status.HTTP_404_NOT_FOUND)

def upload_person_form(request):
    if request.method == "POST":
        name = request.POST.get("name")
        image = request.FILES.get("image")
        if name and image:
            Missingperson.objects.create(name=name, image=image)
            messages.success(request, f"Missing person {name} uploaded successfully!")
            return redirect('upload_person_form')
    return render(request, "upload_person.html")

def detect_video_form(request):
    if request.method == "POST":
        response = detect_in_video(request)
        if response.status_code == 200:
            messages.success(request, "Video processed successfully!")
            return redirect("results_view")
        else:
            messages.error(request, f"Error: {response.data}")
    return render(request, "detect_video.html")

def results_view(request):
    results = Detectionmodelresult.objects.all().order_by("-created_at")
    return render(request, "results.html", {"results": results})



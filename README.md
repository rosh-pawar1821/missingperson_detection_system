
# Project Title

Missing Person Detection System

## 📌 Project Description
The Missing Person Detection System is a Django-based web application that detects missing persons using Machine Learning and Deep Learning.  
Users can upload an image, and the system compares it with stored data to identify the person.

This project is built using Django, Python, and Machine Learning,adeeo Learning.

## 📊 Dataset

The model is trained using a dataset of missing person images.

### Dataset Details
- The dataset contains images of persons for training and testing
- Images are stored in folders based on person name and ID
- Used for face recognition and  image classification

### Dataset Source
- Custom dataset created manually  

## 🤖 Download Object Detection Model

This project uses a pre-trained Object Detection model.
Model are placed into "missingperson_detection_system/model
/"



## 📦 Requirements

This project requires the following Python libraries.

All dependencies are listed in the requirements.txt file.

### Install requirements

Run the following command:pip install -r requirements.txt
### Main Libraries Used

- Django
- numpy
- opencv-python
- tensorflow / keras
- scikit-learn
- pillow
- matplotlib

## ⚙️ Setup

Follow the steps below to setup and run the Missing Person Detection System.

### 1️⃣ Clone the repository
https://github.com/rosh-pawar1821/missingperson_detection_system

### 2️⃣ Create virtual environment
python -m env env

Activate environment: Windows:env\Scripts\activate

### 3️⃣ Install requirements
pip install -r requirements.txt

### 4️⃣ Apply migrations
python manage.py migrate

### 5️⃣ Run server
python manage.py runserer

Open browser:


### 6️⃣ Download Model

Download object detection model and place inside: model/


### 7️⃣ Add Dataset

Place dataset inside: dataset/

## 🎯 Inference Demo

After completing the setup, follow the steps below to test the Object Detection / Missing Person Detection system.

### Step 1 – Run Django Server
python manage.py runserver

Open browser:


### Step 2 – Upload Image

- Go to detection page
- Upload image
- Click Detect

### Step 3 – Model Prediction

The system will:

- Load object detection model
- Process uploaded image
- Detect person
- Show result on screen

### Example

Input:media/siddhi.jpg

Output:found/yes
## 🚀 Features
- Upload missing person image
- Face detection using ML model
- Match with stored database
- Show prediction result
- Admin panel to manage records
- User-friendly UI
## 🛠️ Tech Stack
- Python
- Django RestFrameWork
- HTML / CSS 
- SQLite 
- Scikit-learn / TensorFlow / Keras 
- OpenCV


## Screenshots

## 📸 Screenshots

### Upload Page
![Upload](https://raw.githubusercontent.com/rosh-pawar1821/missingperson_detection_system/main/upload.jpeg)

### Detection Page
![Detection](https://raw.githubusercontent.com/rosh-pawar1821/missingperson_detection_system/main/detection.jpeg)

### Result Page
![Result](https://raw.githubusercontent.com/rosh-pawar1821/missingperson_detection_system/main/result.jpeg)


## 📄 High Level Design (HLD) Document

🔗 View HLD:
[click here](https://docs.google.com/document/d/129riBJY2PMn3BeaY5JjCMhtyt3jF5o_q/view)
## Contributer

Roshani S Pawar


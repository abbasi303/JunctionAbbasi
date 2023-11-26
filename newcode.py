import os
print("Current Working Directory: ", os.getcwd())
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import ShortTermFeatures as aF


import cv2
import numpy as np
import face_recognition

import cv2
import face_recognition

def is_blurred(video_path):
    # Read the video
    video_capture = cv2.VideoCapture(video_path)
    ret, frame = video_capture.read()
    if not ret:
        return False, "No frame found in video"

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian
    laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()

    # Set your threshold for blur detection
    laplacian_threshold = 0  # Example threshold ==============================================set this right with dataset
    is_blurred = laplacian_var < laplacian_threshold

    return is_blurred, "Frame processed"

def process_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    ret, frame = video_capture.read()
    if not ret:
        return "No frame found in video"

    face_locations = face_recognition.face_locations(frame)

    if len(face_locations) > 1:
        return "More than one person detected"

    blurred, message = is_blurred(video_path)
    if blurred:
        return "Video is blurred"

    # Additional checks (eyes open, color composition, etc.) go here

    return f"Found {len(face_locations)} face(s) in the video"









#voice analysis

from pyAudioAnalysis import audioSegmentation as aS
import moviepy.editor as mp

def extract_audio_features(video_path):
    with VideoFileClip(video_path) as video:
        audio = video.audio
        audio_path = "temp_audio.wav"
        audio.write_audiofile(audio_path, codec='pcm_s16le')

    # Load audio file
    try:
        [Fs, x] = audioBasicIO.read_audio_file(audio_path)
        # Check if audio length is sufficient for feature extraction
        if len(x) < int(0.050 * Fs):  # Assuming frame size is 0.050 * Fs
            print("Audio file too short for feature extraction")
            return None, None
        features, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
    except ValueError as e:
        print(f"Error in feature extraction: {e}")
        return None, None
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return features, f_names




import cv2
import speech_recognition as sr
import moviepy.editor as mp
import dlib
from scipy.spatial import distance as dist

# Function to extract audio from video
def extract_audio(video_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio_path = "temp_audio.wav"
    audio.write_audiofile(audio_path)
    return audio_path

# Function to process audio for speech
def process_audio_for_speech(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = ""
    return text

def get_mouth_openness(shape):
    mouth = [shape.part(i) for i in range(48, 68)]

    # Check if sufficient landmarks are detected
    if len(mouth) < 20:
        print("Insufficient mouth landmarks detected")
        return None  # or handle this case as needed

    # Convert Dlib points to tuples (x, y)
    mouth_point_2 = (mouth[2].x, mouth[2].y)
    mouth_point_10 = (mouth[10].x, mouth[10].y)
    mouth_point_0 = (mouth[0].x, mouth[0].y)
    mouth_point_6 = (mouth[6].x, mouth[6].y)

    # Calculate the Euclidean distances
    vertical_distance = dist.euclidean(mouth_point_2, mouth_point_10)
    horizontal_distance = dist.euclidean(mouth_point_0, mouth_point_6)

    # Calculate mouth openness
    mouth_openness = vertical_distance / horizontal_distance
    return mouth_openness

# Function to process video for lip movement
def process_video_for_lip_movement(video_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(video_path)
    lip_movements = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            mouth_openness = get_mouth_openness(shape)
            lip_movements.append(mouth_openness)

    cap.release()
    return lip_movements

# Function to synchronize lip movements with spoken words
def synchronize_lip_audio(lip_movements, spoken_words, frame_rate):
    
    if not spoken_words:
        print("No spoken words detected")
        return None  # or handle this case as needed

    words = spoken_words.split()
    word_timing = len(lip_movements) / len(words)  # Use len(words) here
    threshold = 0.2  # Define a threshold for mouth movement detection

    synced_data = []
    for i, word in enumerate(words):
        start_frame = int(i * word_timing)
        end_frame = int((i + 1) * word_timing)
        word_lip_movement = lip_movements[start_frame:end_frame]
        movement_detected = any(movement > threshold for movement in word_lip_movement)
        synced_data.append((word, movement_detected))

    return synced_data


# Main function to check lip sync
def check_lipsync(video_path):
    audio_path = extract_audio(video_path)
    spoken_words = process_audio_for_speech(audio_path)
    lip_movements = process_video_for_lip_movement(video_path)

    # Check if lip_movements is None or empty
    lip_movements = process_video_for_lip_movement(video_path)
    if not lip_movements:
        print("No lip movements detected")
        return False

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not spoken_words:
        print("No spoken words detected")
        return False

    synced_data = synchronize_lip_audio(lip_movements, spoken_words, frame_rate)
    if not synced_data:
        print("No synced data available")
        return False

    lipsync_matches = all(movement for _, movement in synced_data)
    return lipsync_matches


# Example usage
#result = check_lipsync("path_to_your_video.mp4")
#print("Lip Sync Matches:", result)







# eyes and lips open analysis

import dlib
import cv2
import numpy as np

def check_eyes_and_lips_open(frame):
    # Ensure the frame is a valid numpy array
    if not isinstance(frame, np.ndarray):
        return False, "Frame is not a valid numpy array"

    # Load the facial landmark predictor and face detector
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)
    results = []

    for face in faces:
        shape = predictor(gray, face)

        # Check if eyes and lips are open
        # Note: Implement are_eyes_open and is_mouth_open functions
        eyes_open = are_eyes_open(shape)
        lips_open = is_mouth_open(shape)

        results.append((eyes_open, lips_open))

    return results

# Example usage
# frame = ... # Your frame here
# file_path = 'path/to/shape_predictor_68_face_landmarks.dat'
# results = check_eyes_and_lips_open(frame, file_path)



def are_eyes_open(shape, eye_threshold=0.3):
    leftEye = [shape.part(i) for i in range(36, 42)]
    rightEye = [shape.part(i) for i in range(42, 48)]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return ear > eye_threshold

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

def is_mouth_open(shape, mouth_threshold=0.4):
    mouth = [shape.part(i) for i in range(60, 68)]
    mar = mouth_aspect_ratio(mouth)
    return mar > mouth_threshold

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[3], mouth[9])
    C = dist.euclidean(mouth[4], mouth[8])
    D = dist.euclidean(mouth[0], mouth[6])

    mar = (A + B + C) / (2.0 * D)
    return mar

# Example usage
#results = check_eyes_and_lips_open("path_to_your_image.jpg")
#for eyes_open, lips_open in results:
#    print(f"Eyes Open: {eyes_open}, Lips Open: {lips_open}")





#analyze color composition
import torch
import torchvision.transforms as T
from torchvision import models
import cv2
import numpy as np
from sklearn.cluster import KMeans

def load_deeplab_model():
    # Load a pre-trained DeepLabV3 model
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model

def preprocess_image(image):
    # Preprocess the image
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)

def segment_image(image, model):
    # Preprocess and add batch dimension
    input_tensor = preprocess_image(image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    return output_predictions.cpu().numpy()

def get_dominant_colors(image, k=5): # k is 
    pixels = image.reshape((-1, 3))
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    return dominant_colors
import logging

import logging
from PIL import Image
def analyze_video_color_composition(video_path, frame_sampling_rate=30):
    deeplab_model = load_deeplab_model()
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        logging.error("Failed to open video file.")
        return None

    frame_count = 0
    all_background_colors = []
    all_subject_colors = []

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret or frame is None:
            break

        # Process every nth frame
        if frame_count % frame_sampling_rate == 0:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mask = segment_image(frame_rgb, deeplab_model)

                if mask is None or mask.size == 0:
                    logging.warning("Invalid or empty mask for frame number {}".format(frame_count))
                    continue
                else:
                    logging.info("Mask shape before resize: {}, data type: {}, unique values: {}".format(mask.shape, mask.dtype, np.unique(mask)))

                try:
    # Convert mask to a compatible data type for PIL, if necessary
                    mask_compatible = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask

    # Ensure the mask is not empty before resizing
                    if mask_compatible.size > 0:
                        mask_resized = cv2.resize(mask_compatible, (frame_rgb.shape[1], frame_rgb.shape[0]))
                        mask_resized_3d = np.stack((mask_resized,)*3, axis=-1)
                    else:
                        raise ValueError("Empty or incompatible mask cannot be resized")
                except Exception as e:
                    logging.error("Error resizing mask with OpenCV for frame number {}: {}".format(frame_count, e))
                    try:
                        pil_mask = Image.fromarray(mask_compatible)
                        pil_mask_resized = pil_mask.resize((frame_rgb.shape[1], frame_rgb.shape[0]), Image.NEAREST)
                        mask_resized = np.array(pil_mask_resized)
                    except Exception as e:
                        logging.error("Error resizing mask with PIL for frame number {}: {}".format(frame_count, e))
                        continue
                background = np.where(mask_resized_3d == 0, frame_rgb, 0)
                subject = np.where(mask_resized_3d != 0, frame_rgb, 0)

                background_colors = get_dominant_colors(background, 5)
                subject_colors = get_dominant_colors(subject, 5)

                all_background_colors.append(background_colors)
                all_subject_colors.append(subject_colors)

            except Exception as e:
                logging.error("Error processing frame number {}: {}".format(frame_count, e))
                continue

        frame_count += 1
        logging.info("Processing frame number {}. Frame shape: {}".format(frame_count, frame.shape))

    video_capture.release()
    return all_background_colors, all_subject_colors


    # Return or process the aggregated results

# Example usage
#background_colors, subject_colors = analyze_image("path_to_your_image.jpg")
#print("Background Colors:", background_colors)
#print("Subject Colors:", subject_colors)



#glare and blur analysis

import cv2
import numpy as np

def check_blur_and_glare(image, blur_threshold=100.0, glare_threshold=250):
    # Validate the image
    if not isinstance(image, np.ndarray) or image.size == 0:
        print("Invalid image input")
        return None, None

    # Check for blur
    try:
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        is_blurred = laplacian_var < blur_threshold
    except cv2.error as e:
        print(f"Error in blur check: {e}")
        return None, None

    # Check for glare
    has_glare = detect_glare(image, glare_threshold)

    return is_blurred, has_glare

def detect_glare(image, threshold):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to get regions of high intensity
    _, high_intensity_regions = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Check the amount and distribution of high-intensity regions
    # This can be a simple check like the percentage of the image that is overexposed,
    # or a more complex analysis based on the size and shape of regions
    glare_percentage = np.sum(high_intensity_regions == 255) / high_intensity_regions.size

    # Define a threshold for what you consider as glare
    glare_detected = glare_percentage > 0.1  # Example threshold

    return glare_detected

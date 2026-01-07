import cv2
import numpy as np
import os
from ..config import IMG_SIZE, CHANNELS

# Load the classifier (standard path for opencv data)
# We assume standard environment; if it fails, we fall back to full frame
try:
    FACE_CLASSIFIER_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
except AttributeError:
    # Fallback for some environments
    FACE_CLASSIFIER_PATH = 'haarcascade_frontalface_default.xml'

CROP_PADDING = 1.3

def find_largest_face(faces):
    """Finds the largest face from a list of detected faces (x, y, w, h)."""
    largest_face = None
    max_area = 0

    for (x, y, w, h) in faces:
        current_area = w * h
        if current_area > max_area:
            max_area = current_area
            largest_face = (x, y, w, h)

    return largest_face

def extract_frames(video_path, max_frames=20):
    """
    Extracts frames, detects the largest face, crops with padding, and resizes.
    Mimics the 'create_frames' logic from the notebook exactly.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    face_classifier = cv2.CascadeClassifier(FACE_CLASSIFIER_PATH)
    if face_classifier.empty():
        print("Couldnt not load Haar Cascade. Continuing without face cropping.")
        use_face_detection = False
    else:
        use_face_detection = True

    try:
        frame_count = 0
        # extract 1 out of every 10 frames
        extraction_rate = 10 
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % extraction_rate == 0:
                if use_face_detection:
                    #convert to gray for faster detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Detect faces in images
                    faces = face_classifier.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=4,
                        minSize=(100, 100)
                    )
                    
                    #Face selection and Cropping
                    if len(faces) > 0:
                        (x, y, w, h) = find_largest_face(faces)
                        
                        # Apply padding 
                        center_x = x + w // 2
                        center_y = y + h // 2
                        new_size = int(max(w, h) * CROP_PADDING)
                        
                        # Define Crop Boundaries
                        x1 = max(0, center_x - new_size // 2)
                        y1 = max(0, center_y - new_size // 2)
                        x2 = min(frame.shape[1], center_x + new_size // 2)
                        y2 = min(frame.shape[0], center_y + new_size // 2)
                        
                        cropped_frame = frame[y1:y2, x1:x2]
                        
                        # Resize to (224, 224)
                        resized_frame = cv2.resize(cropped_frame, IMG_SIZE)
                        frames.append(resized_frame)
                    else:
                        # Skip frame if no face detected
                        pass 
                else:
                    # Fallback if no classifier
                    resized_frame = cv2.resize(frame, IMG_SIZE)
                    frames.append(resized_frame)
                    
            frame_count += 1
            
    finally:
        cap.release()

    # adjust frames to have a length equal to max_frames
    if len(frames) > 0:
        frames = np.array(frames)
        if len(frames) > max_frames:
            # Uniformly sample
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = frames[indices]
        elif len(frames) < max_frames:
            # Pad with last frame
            diff = max_frames - len(frames)
            last_frame = frames[-1]
            padding = np.array([last_frame] * diff)
            frames = np.concatenate([frames, padding], axis=0)
    else:
        # Return zeros if detection failed completely
        frames = np.zeros((max_frames, *IMG_SIZE, CHANNELS))

    return frames
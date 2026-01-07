import numpy as np
import os
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


from .config import MAX_SEQ_LENGTH, IMG_SIZE
from .preprocessing.video_utils import extract_frames
from .preprocessing.audio_utils import create_spectrogram
from .preprocessing.text_utils import transcribe_audio

def predict_single_video(video_path, model_path='final_model.h5', tokenizer_path='tokenizer.pickle'):
    """
    Runs the full Lie Detection pipeline for single video files.
    """
    if not os.path.exists(video_path):
        print(f"Video file not found at {video_path}")
        return

    print(f"--- Processing Video ---")

    # -------------------------------------------
    # 1. Preprocessing
    # -------------------------------------------
    
    # Visual Features
    print("1. Extracting Frames...")
    frames = extract_frames(video_path)
    # Adjust dimension, Ex: (20, 224, 224, 3) -> (1, 20, 224, 224, 3)
    frames = np.expand_dims(frames, axis=0)

    # Audio Features
    print("2. Generating Audio Spectrogram...")
    spectrogram = create_spectrogram(video_path)
    # Adjust dimension: Ex: (224, 224, 3) -> (1, 224, 224, 3)
    spectrogram = np.expand_dims(spectrogram, axis=0)

    # Text Features
    print("3. Transcribing Audio...")
    text_transcript = transcribe_audio(video_path)

    
    # Tokenization
    if not os.path.exists(tokenizer_path):
        print("Tokenizer not found. Run train.py first.")
        return
        
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    sequences = tokenizer.texts_to_sequences([text_transcript])
    text_data = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH)

    # -------------------------------------------
    # 2. Prediction
    # -------------------------------------------
    print("4. Loading Model...")
    if not os.path.exists(model_path):
        print("Model file not found.")
        return

    model = tf.keras.models.load_model(model_path)
    
    print("5. Predicting...")
    prediction_score = model.predict([frames, spectrogram, text_data])[0][0]
    
    # -------------------------------------------
    # 3. Results
    # -------------------------------------------
    result = "DECEPTIVE" if prediction_score > 0.5 else "TRUTHFUL"
    confidence = prediction_score if result == "DECEPTIVE" else 1 - prediction_score
    
    print("\n" + "="*30)
    print(f"FINAL PREDICTION: {result}")
    print(f"Confidence: {confidence:.2%}")
    print("="*30 + "\n")

if __name__ == "__main__":
    video_file_path = "data/test_video.mp4" 
    
    if not os.path.exists(video_file_path):
        print(f"Video not found at file path")
    else:
        predict_single_video(video_file_path)
import librosa
import numpy as np
import cv2
from ..config import IMG_SIZE, CHANNELS

def create_spectrogram(audio_path):
    """
    Converts audio to a log-mel spectrogram and resizes it to an image.
    """
    # Load audio (sr=None preserves native sampling rate)
    y, sr = librosa.load(audio_path, sr=None)
    
    # Generate Mel Spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Resize to match VGG input size (treating time axis as width)
    img = cv2.resize(spectrogram_db, IMG_SIZE)
    
    # Min-Max Normalization to 0-1 range
    img = (img - img.min()) / (img.max() - img.min())
    
    # Stack to create 3 channels (R,G,B)
    img = np.stack((img,)*CHANNELS, axis=-1)
    
    return img
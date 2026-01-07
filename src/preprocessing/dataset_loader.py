import os
import numpy as np
import cv2
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from ..config import DATA_DIR, MAX_SEQ_LENGTH
from .video_utils import extract_frames
from .audio_utils import create_spectrogram
from .text_utils import transcribe_audio, load_glove_embeddings, create_embedding_matrix

def load_multimodal_data():
    """
    Loads video, audio, and text data from the dataset folders.
    returns 'tokenizer' so it can be saved for prediction.
    """
    frame_data = []
    spectrogram_data = []
    transcription_data = []
    labels = []
    
    classes = ['Truthful', 'Deceptive'] 
    
    print(f"Loading data from {DATA_DIR}...")

    for label_id, label_name in enumerate(classes):
        class_dir = os.path.join(DATA_DIR, label_name)
        
        if not os.path.exists(class_dir):
            print(f"Directory {class_dir} not found. Skipping.")
            continue
            
        video_files = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
        
        for video_file in tqdm(video_files, desc=f"Processing {label_name}"):
            video_path = os.path.join(class_dir, video_file)
            
            try:
                # Visual
                frames = extract_frames(video_path)
                # Audio
                spectrogram = create_spectrogram(video_path)
                # Text
                transcription = transcribe_audio(video_path)
                
                for frame in frames:
                    frame_data.append(frame)

                    #Duplicate so that during Concatenation of the sub-models, there is no shape conflicts between the 3 models
                    spectrogram_data.append(spectrogram)
                    transcription_data.append(transcription)
                
                    #Labels
                    labels.append(label_id)
                
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                continue

    print("Data loading complete. Converting to Numpy arrays...")
    x_frames = np.array(frame_data)
    x_spectrograms = np.array(spectrogram_data)
    y = np.array(labels)
    
    # Text Tokenization
    print("Tokenizing text...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(transcription_data)
    
    sequences = tokenizer.texts_to_sequences(transcription_data)
    X_text = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH)
    
    vocab_size = len(tokenizer.word_index) + 1
    
    # Load Embeddings
    print("Preparing embedding matrix...")
    glove_index = load_glove_embeddings()
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, glove_index, vocab_size)
    
    return x_frames, x_spectrograms, X_text, y, vocab_size, embedding_matrix, tokenizer
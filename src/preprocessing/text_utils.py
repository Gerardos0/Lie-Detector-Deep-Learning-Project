import whisper
import numpy as np
import os
from ..config import GLOVE_PATH, EMBEDDING_DIM

def transcribe_audio(audio_path):
    """
    Uses OpenAI Whisper 'base' model to transcribe speech to text.
    """
    model = whisper.load_model("base")
    # Whisper can process audio directly from video files
    result = model.transcribe(audio_path)
    return result["text"]

def load_glove_embeddings():
    """
    Parses GloVe text file into a dictionary.
    """
    embeddings_index = {}
    if not os.path.exists(GLOVE_PATH):
        print("GloVe file not found. Returning empty embeddings.")
        return {}

    with open(GLOVE_PATH, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            
    return embeddings_index

def create_embedding_matrix(word_index, embeddings_index, vocab_size):
    """
    Creates the weight matrix for the Embedding Layer.
    """
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    
    for word, i in word_index.items():
        if i < vocab_size:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                
    return embedding_matrix
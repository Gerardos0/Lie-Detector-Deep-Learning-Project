import whisper
import numpy as np
import os
import re
import string
from ..config import GLOVE_PATH, EMBEDDING_DIM
WHISPER_MODEL = None
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("openai-whisper not installed.")


def clean_text(text):
    """
    Cleans transcribed texts.
    Removes uppercase letters, audio cues, punctuation, and extra whitespace
    """
    text = text.lower()
    #remove audio cues like [music] or (laughter)
    text = re.sub(r'\[.*?\]|\(.*?\)|\<.*?\>', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove symbols
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

def transcribe_audio(audio_path):
    """
    Uses OpenAI Whisper 'base' model to transcribe audio to text.
    """
    global WHISPER_MODEL
    
    if not WHISPER_AVAILABLE:
        return "silence"

    if WHISPER_MODEL is None:
        print("Loading Whisper Model...")
        WHISPER_MODEL = whisper.load_model("base")

    # Transcribe
    result = WHISPER_MODEL.transcribe(audio_path)
    
    # Clean the text and return
    return clean_text(result["text"])

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
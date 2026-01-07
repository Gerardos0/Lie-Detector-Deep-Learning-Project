from tensorflow.keras import models, Input
from tensorflow.keras.layers import Embedding, Conv1D, GRU, Dense, Dropout
from ..config import MAX_SEQ_LENGTH, EMBEDDING_DIM

def build_transcription_model(vocab_size, embedding_matrix):
    """
    Text Branch: GloVe Embeddings -> CNN -> GRU
    """
    inputs = Input(shape=(MAX_SEQ_LENGTH,))
    
    # 1. Embedding Layer (Pre-trained GloVe weights)
    x = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQ_LENGTH,
        trainable=False
    )(inputs)
    
    x = Conv1D(filters=128, kernel_size=5, activation='relu', name="conv_1d")(x)
    x = GRU(100, dropout=0.2, recurrent_dropout=0.2, name="gru_layer")(x)
    x = Dense(128, activation='relu', name="dense_1")(x)
    x = Dropout(0.5, name="dropout_1")(x)
    x = Dense(64, activation='relu', name="dense_2")(x)
    x = Dropout(0.5, name="dropout_2")(x)
    x = Dense(32, activation='relu', name="dense_3")(x)
    predictions = Dense(1, activation='sigmoid', name="output_dense")(x)

    
    return models.Model(inputs, predictions, name="Transcription_Model")
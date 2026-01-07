from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from ..config import IMG_SIZE

def build_spectrogram_model():
    """
    Audio Branch: VGG16 (on Spectrograms) + Dense Head
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False 
    
    # Renaming layers to prevent name collisions when fusing the VGG16 models
    for layer in base_model.layers:
        layer._name = f"spectrograms_{layer.name}"

    inputs = Input(shape=IMG_SIZE + (3,), name="spectrograms_input")
    x = base_model(inputs)
    x = GlobalAveragePooling2D(name="spectrograms_GAP")(x)
    x = Dense(256, activation='relu', name="spectrograms__Dense_256")(x)
    x = Dropout(0.5, name=f"spectrograms__dropout_1")(x)
    x = Dense(128, activation='relu', name="spectrograms__Dense_128")(x)
    x = Dropout(0.5, name="spectrograms__dropout_2")(x)
    x = Dense(32, activation='relu', name="spectrograms__Dense_32")(x)
    predictions = Dense(1, activation='sigmoid', name="spectrograms__final_output")(x)
    
    return models.Model(inputs, predictions, name="Spectrograms_Model")
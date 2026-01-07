from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from ..config import IMG_SIZE

def build_frame_model(): 
    """
    Visual Branch (Frames): VGG16 + Dense Head
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False
    
    # Renaming layers to prevent name collisions when fusing the VGG16 models
    for layer in base_model.layers:
        layer._name = f"frames_{layer.name}"
    
    inputs = Input(shape=IMG_SIZE + (3,), name="frames_input")
    x = base_model(inputs)
    x = GlobalAveragePooling2D(name="frames_GAP")(x)
    x = Dense(256, activation='relu', name="frames__Dense_256")(x)
    x = Dropout(0.5, name=f"frames__dropout_1")(x)
    x = Dense(128, activation='relu', name="frames__Dense_128")(x)
    x = Dropout(0.5, name="frames__dropout_2")(x)
    x = Dense(32, activation='relu', name="frames__Dense_32")(x)
    predictions = Dense(1, activation='sigmoid', name="frames__final_output")(x)

    return models.Model(inputs, predictions, name="Frames_Model")
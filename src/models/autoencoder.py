from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from ..config import IMG_SIZE

def build_autoencoder(channels=3):
    """
    Constructs the Convolutional Autoencoder for anomaly detection.
    Exact architecture from the notebook.
    """
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], channels)
    input_img = Input(shape=input_shape)

    # --- Encoder ---
    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # --- Decoder ---
    # Block 3 (Reverse)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    
    # Block 2 (Reverse)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    # Block 1 (Reverse)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    # Output
    decoded = Conv2D(channels, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded, name="Anomaly_Detector_Autoencoder")
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder
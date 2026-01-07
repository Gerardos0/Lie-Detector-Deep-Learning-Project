import numpy as np
import tensorflow as tf
from ..models.autoencoder import build_autoencoder
from ..config import IMG_SIZE, BATCH_SIZE

def train_and_filter_anomalies(data, modality_name="Autoencoder", threshold_sigma=3.0):
    """
    Trains an autoencoder on the data and filters out samples with high reconstruction error.
    Returns:
        filtered_data (np.array): Data with anomalies removed.
        valid_indices (list): Indices of the data that were kept (to sync labels).
    """
    print(f"--- Starting Anomaly Detection for {modality_name} ---")
    

    original_shape = data.shape
    if len(data.shape) == 5: 
        # Reshape to (N_vidoes * Frames, H, W, C) for training
        train_data = data.reshape(-1, *data.shape[2:])
    else:
        train_data = data

    # Build and Train Autoencoder
    autoencoder = build_autoencoder(channels=train_data.shape[-1])
    
    print(f"Training Autoencoder on {len(train_data)} samples.")
    autoencoder.fit(
        train_data, train_data,
        epochs=10,
        batch_size=64,
        shuffle=True,
        verbose=1
    )
    
    # Predict and Calculate MSE
    print("Calculating reconstruction errors...")
    reconstructed = autoencoder.predict(train_data, batch_size=64)
    # MSE calculation per sample
    mse = np.mean(np.power(train_data - reconstructed, 2), axis=(1, 2, 3))
    
    # Determine Threshold (Mean + Z * Std)
    mean_error = np.mean(mse)
    std_error = np.std(mse)
    threshold = mean_error + (threshold_sigma * std_error)
    print(f"Anomaly Threshold (MSE): {threshold:.5f}")
    
    # Filter Data
    valid_indices = []
    
    if len(data.shape) == 5:
        # Reshape MSE back to (N_videos, Frames)
        mse_video = mse.reshape(original_shape[0], original_shape[1])
        # Average MSE per video
        mse_avg = np.mean(mse_video, axis=1)
        
        for i, error in enumerate(mse_avg):
            if error <= threshold:
                valid_indices.append(i)
            else:
                pass # Anomaly detected
    else:
        # Spectrograms (Single image per video)
        for i, error in enumerate(mse):
            if error <= threshold:
                valid_indices.append(i)
    
    filtered_data = data[valid_indices]
    
    print(f"Removed {len(data) - len(filtered_data)} anomalies. Kept {len(filtered_data)} samples.")
    return filtered_data, valid_indices
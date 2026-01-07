import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from .config import EPOCHS, BATCH_SIZE

# Import Preprocessing & Anomaly Detection
from .preprocessing.dataset_loader import load_multimodal_data
from .preprocessing.anomaly_detector import train_and_filter_anomalies

# Import Models
from .models.frame_model import build_frame_model
from .models.spectrogram_model import build_spectrogram_model
from .models.transcription_model import build_transcription_model
from .models.fusion_model import build_fusion_model

# Import Utilities
from .utils import plot_training_history, unfreezing_lower_layers

# Training Hyperparameters
TRAIN_EPOCHS = 10 
FINETUNE_EPOCHS = 20     
FINE_TUNE_LR = 0.000001

early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
)

def main():
    print("=== Multimodal Lie Detection System ===")
    
    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    print("\n--- Load Data ---")
    try:
        x_frames, x_spectrograms, x_trainscriptions, y, vocab_size, embedding_matrix, tokenizer = load_multimodal_data()
    except Exception as e:
        print(f"Could not load data: {e}")
        return

    # ---------------------------------------------------------
    # 2. Anomaly Detection
    # ---------------------------------------------------------
    print("\n--- Frames Anomaly Detection ---")

    # Sync data after frames anomaly deletion
    x_frames, valid_idx_frames = train_and_filter_anomalies(x_frames, "Frames")
    x_spectrograms = x_spectrograms[valid_idx_frames]
    x_trainscriptions = x_trainscriptions[valid_idx_frames]
    y = y[valid_idx_frames]

    print("\n--- Spectrograms Anomaly Detection ---")

    #sync data after spectrogram anomaly deletion
    x_spectrograms, valid_idx_spec = train_and_filter_anomalies(x_spectrograms, "Spectrograms")
    x_frames = x_frames[valid_idx_spec]
    x_trainscriptions = x_trainscriptions[valid_idx_spec]
    y = y[valid_idx_spec]

    # ---------------------------------------------------------
    # 3. Data Splitting
    # ---------------------------------------------------------
    print(f"\nFinal Dataset Size: {len(y)}")
    x_frames_train, x_frames_val, x_spec_train, x_spec_val, x_trans_train, x_trans_val, y_train, y_val = train_test_split(
        x_frames, x_spectrograms, x_trainscriptions, y, test_size=0.2, random_state=42
    )

    # ---------------------------------------------------------
    # 4. Build Models
    # ---------------------------------------------------------
    print("\n--- Building Models ---")
    frames_model = build_frame_model()
    spectrograms_model = build_spectrogram_model()
    transcription_model = build_transcription_model(vocab_size, embedding_matrix)
    
    fusion_model = build_fusion_model(frames_model, spectrograms_model, transcription_model)
    fusion_model.summary()

    # ---------------------------------------------------------
    # 5. PHASE 1: Train head only
    # ---------------------------------------------------------
    print(f"\n=== PHASE 1: TRAINING ({TRAIN_EPOCHS} Epochs) ===")
    history_train = fusion_model.fit(
        [x_frames_train, x_spec_train, x_trans_train],
        y_train,
        validation_data=([x_frames_val, x_spec_val, x_trans_val], y_val),
        epochs=TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping]
    )

    # ---------------------------------------------------------
    # 6. PHASE 2: FINE-TUNING
    # ---------------------------------------------------------
    print("\n=== PHASE 2: FINE-TUNING ===")
    
    # Unfreeze higher layers of the sub-models
    unfreezing_lower_layers(frames_model)
    unfreezing_lower_layers(spectrograms_model)
    
    # Re-compile the Fusion Model
    fusion_model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    
    # Train again
    print(f"Starting Fine-Tuning for {FINETUNE_EPOCHS} Epochs...")
    history_finetune = fusion_model.fit(
        [x_frames_train, x_spec_train, x_trans_train],
        y_train,
        validation_data=([x_frames_val, x_spec_val, x_trans_val], y_val),
        epochs=FINETUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping]
    )

    # ---------------------------------------------------------
    # 7. Save & Visualize
    # ---------------------------------------------------------
    print("\nSaving final model...")
    fusion_model.save('final_model.h5')
    
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plot results (plotting the fine-tuning phase)
    plot_training_history(history_finetune)
    print("\nTraining Complete.")

if __name__ == "__main__":
    main()
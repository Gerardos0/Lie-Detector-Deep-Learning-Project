from tensorflow.keras import models
from tensorflow.keras.layers import concatenate, Dense, Dropout

def build_fusion_model(visual_model, audio_model, text_model):
    """
    Fusion: Concatenates the branches and makes final prediction.
    """
    # Concatenate the output of the three sub-models
    combined = concatenate([visual_model.output, audio_model.output, text_model.output])
    
    # Final Classification Head
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid', name='fusion_output')(x)
    
    # Create the full model graph
    model = models.Model(
        inputs=[visual_model.input, audio_model.input, text_model.input], 
        outputs=predictions,
        name="Multimodal_Fusion_Network"
    )
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
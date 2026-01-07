import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

def plot_training_history(history):
    """
    Plots accuracy and loss graphs from the training history.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.show()

def unfreezing_lower_layers(model):
    """
    Unfreezes the upper blocks (4 and 5) of a VGG-based model for fine-tuning.
    """
    print(f"--- Unfreeze layers for {model.name} ---")
    
    for layer in model.layers:
        if 'block1' in layer.name or 'block2' in layer.name or 'block3' in layer.name:
            layer.trainable = False
        elif 'block4' in layer.name or 'block5' in layer.name:
            layer.trainable = True
                
    print(f"Frozen layers adjusted for {model.name}.")
    return model
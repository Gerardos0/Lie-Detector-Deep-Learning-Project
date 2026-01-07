# Multimodal Lie Detection System

## Overview
This project implements a **multimodal lie detection system** leveraging **video, audio, and textual modalities** to classify deceptive versus truthful behavior.

## Repository Structure
- `src/preprocessing/`: Modules for extracting frames, spectrograms, and transcriptions.
- `src/models/`: Dedicated Keras architectures for each modality.
- `src/train.py`: Main training pipeline.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Download GloVe embeddings (`glove.6B.100d.txt`) to the root folder.
3. Run training: `python src/train.py`
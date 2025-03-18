# ASR-Prototyping

## Overview
A foundational project for Automatic Speech Recognition (ASR) using Python, PyTorch, and torchaudio. Focuses on data preprocessing, model design, and evaluation metrics (WER, CER).

## Usage
1. Prepare your audio clips in `data/audio_clips/` and corresponding transcripts in `data/transcriptions/`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train:
   ```bash
   python src/train.py
   ```
4. Evaluate:
   ```bash
   python src/evaluate.py
   ```

## Future Directions
- Incorporate advanced models (Transformer-based ASR).
- Add real-time inference pipeline.
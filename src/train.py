"""
train.py

A simple training script for a CTC-based ASR model using PyTorch and torchaudio.
"""

import os
import torch
import torchaudio
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from model import ASRModel
import torch.nn.functional as F

# ----------------------------
# 1. Hyperparameters & Config
# ----------------------------

BATCH_SIZE = 4
LR = 1e-3
EPOCHS = 5
NUM_WORKERS = 2

# Example char set (small subset: a-z plus space, apostrophe, blank)
# The final "blank" index is typically required for CTC.
VOCAB = [
    "_",  # CTC blank
    " ", "a", "b", "c", "d", "e", "f", "g", "h", "i",
    "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
    "t", "u", "v", "w", "x", "y", "z", "'"
]
NUM_CLASSES = len(VOCAB)  # 30 in this example

# ----------------------------
# 2. Dataset Definition
# ----------------------------

class SimpleASRDataset(Dataset):
    """
    Expects each sample to have:
      - A path to an audio file (e.g., WAV)
      - A corresponding transcript (string)
    You can adapt this to read from CSV or an external manifest.
    """
    def __init__(self, data_dir, is_train=True, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform

        # For demonstration, we assume a structure:
        # data/audio_clips/*.wav
        # data/transcriptions/*.txt (same filename as .wav)
        # In practice, you'd handle this more robustly.
        self.samples = []
        audio_path = os.path.join(data_dir, "audio_clips")
        for f in os.listdir(audio_path):
            if f.endswith(".wav"):
                transcript_file = f.replace(".wav", ".txt")
                transcript_path = os.path.join(data_dir, "transcriptions", transcript_file)
                if os.path.exists(transcript_path):
                    self.samples.append((os.path.join(audio_path, f), transcript_path))

        # Simple shuffle in place
        random.shuffle(self.samples)

        # Split dataset into train/val (90/10 split)
        split_index = int(0.9 * len(self.samples))
        if is_train:
            self.samples = self.samples[:split_index]
        else:
            self.samples = self.samples[split_index:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_file, transcript_file = self.samples[idx]

        waveform, sr = torchaudio.load(audio_file)
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Basic feature extraction: Convert to Mel Spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=80
        )
        mel_spec = mel_transform(waveform)  # shape: (1, n_mels, time)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)  # log scale

        # Transpose to match (time, feature)
        features = mel_spec_db.squeeze(0).transpose(0, 1)  # shape: (time, 80)

        # Read transcript
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript = f.read().strip().lower()

        # Convert transcript to label indices
        labels = [VOCAB.index(ch) for ch in transcript if ch in VOCAB]

        return {
            "features": features,      # (time, feature_dim)
            "labels": torch.tensor(labels, dtype=torch.long),
            "input_length": features.shape[0],   # time dimension
            "label_length": len(labels)
        }


def collate_fn(batch):
    """
    Custom collate function to pad variable-length feature sequences and labels.
    """
    # Sort by descending input_length so we can pack sequences
    batch.sort(key=lambda x: x["input_length"], reverse=True)

    features = [x["features"] for x in batch]
    labels = [x["labels"] for x in batch]
    input_lengths = [x["input_length"] for x in batch]
    label_lengths = [x["label_length"] for x in batch]

    # Pad features
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    # Pad labels
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    return (
        padded_features,            # shape: (batch, max_time, feature_dim)
        torch.tensor(input_lengths),# shape: (batch,)
        padded_labels,              # shape: (batch, max_label_len)
        torch.tensor(label_lengths) # shape: (batch,)
    )

# ----------------------------
# 3. Training Function
# ----------------------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create train/val datasets
    train_dataset = SimpleASRDataset(data_dir="data", is_train=True)
    val_dataset   = SimpleASRDataset(data_dir="data", is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)

    # Initialize model
    model = ASRModel(input_dim=80, hidden_dim=256, num_layers=2,
                     num_classes=NUM_CLASSES, bidirectional=True).to(device)

    # Define optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_data in train_loader:
            features, input_lengths, labels, label_lengths = batch_data
            features = features.to(device)
            labels   = labels.to(device)

            optimizer.zero_grad()
            # Model forward
            logits, output_lengths = model(features, input_lengths)

            # CTC expects shape: (time, batch, num_classes)
            logits_transposed = logits.transpose(0, 1)  # (time, batch, num_classes)

            loss = ctc_loss(logits_transposed, labels, output_lengths, label_lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch_data in val_loader:
                features, input_lengths, labels, label_lengths = batch_data
                features = features.to(device)
                labels   = labels.to(device)

                logits, output_lengths = model(features, input_lengths)
                logits_transposed = logits.transpose(0, 1)
                loss = ctc_loss(logits_transposed, labels, output_lengths, label_lengths)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save final model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/asr_model.pth")
    print("Model saved to checkpoints/asr_model.pth")

if __name__ == "__main__":
    train()

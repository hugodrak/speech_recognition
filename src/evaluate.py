"""
evaluate.py

Loads a saved model and computes Word Error Rate (WER) on the validation set.
Includes a simple greedy decoder for demonstration.
"""

import torch
import os
from torch.utils.data import DataLoader
from model import ASRModel
from train import SimpleASRDataset, collate_fn, VOCAB, NUM_CLASSES

def greedy_decode(logits):
    """
    Greedy decoder for log probabilities (CTC).
    logits: (time, num_classes)
    Returns a list of decoded token indices.
    """
    # Take argmax over the classes
    argmax = torch.argmax(logits, dim=-1)
    # Remove consecutive duplicates and blanks (index 0)
    decoded = []
    prev = None
    for idx in argmax:
        if idx != prev and idx != 0:
            decoded.append(idx.item())
        prev = idx
    return decoded

def indices_to_text(indices):
    return "".join([VOCAB[i] for i in indices if i < len(VOCAB)])

def compute_wer(ref, hyp):
    """
    Computes Word Error Rate between two strings.
    This is a simplistic approach for demonstration.
    """
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    # Using a standard dynamic programming approach
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion    = d[i][j - 1] + 1
                deletion     = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    
    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 0.0
    return wer

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ASRModel(input_dim=80, hidden_dim=256, num_layers=2,
                     num_classes=NUM_CLASSES, bidirectional=True).to(device)
    checkpoint_path = "checkpoints/asr_model.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Model checkpoint not found. Please train the model first.")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Create validation dataset
    val_dataset = SimpleASRDataset(data_dir="data", is_train=False)
    val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    total_wer = 0.0
    sample_count = 0

    for batch_data in val_loader:
        features, input_lengths, labels, label_lengths = batch_data
        features = features.to(device)
        labels   = labels.to(device)

        with torch.no_grad():
            logits, output_lengths = model(features, input_lengths)
            # (batch, time, num_classes) -> (time, batch, num_classes)
            logits_transposed = logits.transpose(0, 1).detach().cpu()

        # Greedy decode each sequence in the batch
        for b_idx in range(features.size(0)):
            time_len = output_lengths[b_idx]
            # Extract the time slice for this item
            item_logits = logits_transposed[:time_len, b_idx, :]
            pred_indices = greedy_decode(item_logits)
            pred_text = indices_to_text(pred_indices)

            # Reference text
            ref_indices = labels[b_idx][:label_lengths[b_idx]].tolist()
            ref_text = indices_to_text(ref_indices)

            wer = compute_wer(ref_text, pred_text)
            total_wer += wer
            sample_count += 1

            # Optional: print some samples for debugging
            # print(f"REF: {ref_text}\nHYP: {pred_text}\nWER: {wer:.2f}\n")

    avg_wer = total_wer / sample_count if sample_count > 0 else 0.0
    print(f"Validation WER: {avg_wer:.2f}")

if __name__ == "__main__":
    evaluate()

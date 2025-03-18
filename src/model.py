"""
model.py

Defines the ASRModel class using an LSTM-based acoustic encoder and a linear
output layer. Uses Connectionist Temporal Classification (CTC) for speech-to-text.
"""

import torch
import torch.nn as nn

class ASRModel(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, num_layers=2, num_classes=30, bidirectional=True):
        """
        Args:
            input_dim (int): Size of the input feature dimension (e.g., mel filter banks).
            hidden_dim (int): LSTM hidden state dimension.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Size of the output vocabulary (+1 for blank symbol in CTC).
            bidirectional (bool): Whether the LSTM is bidirectional.
        """
        super(ASRModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        if bidirectional:
            fc_in_dim = hidden_dim * 2
        else:
            fc_in_dim = hidden_dim

        self.fc = nn.Linear(fc_in_dim, num_classes)

    def forward(self, x, x_lengths):
        """
        Forward pass of the network.

        Args:
            x (Tensor): Acoustic features of shape (batch, time, feature_dim).
            x_lengths (Tensor): Lengths of each sequence in the batch (before any padding).
        
        Returns:
            output (Tensor): Logits of shape (batch, time, num_classes).
            output_lengths (Tensor): Same as input lengths (for CTC).
        """
        # Pack the padded batch
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, x_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed_input)

        # Unpack
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        # Linear projection
        output = self.fc(rnn_out)

        return output, x_lengths

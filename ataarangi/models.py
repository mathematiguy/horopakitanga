import os
import json
import math
import click
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from itertools import chain
from torch.utils.data import DataLoader
from ataarangi.data import RākauDataset, SequenceTokenizer, load_data


class TransformerModel(nn.Module):
    def __init__(
        self,
        tokenizer,
        embed_size,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        max_seq_length,
        dropout=0.1,
        class_successors_json="data/class_successors.json",
        token_to_class_json="data/token_to_class.json",
    ):
        super(TransformerModel, self).__init__()
        self.architecture = "transformer"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.embed_size = embed_size
        self.vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.class_successors_json = class_successors_json
        self.token_to_class_json = token_to_class_json
        self.logits_mask = self.create_logits_mask()

        # Create a positional encoding that is large enough for any sequence you expect to process
        self.register_buffer(
            "positional_encodings",
            self.create_positional_encodings(max_seq_length, embed_size),
        )
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(embed_size, self.vocab_size)

    def create_positional_encodings(self, max_len, embed_size):
        """Create positional encodings for transformer model."""
        pos_enc = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)

    def create_logits_mask(self):

        class_successor_dict = json.load(open(self.class_successors_json))
        token_to_class_dict = json.load(open(self.token_to_class_json))

        class_to_tokens_dict = {
            k: list(chain.from_iterable([token_to_class_dict[l] for l in v]))
            for k, v in class_successor_dict.items()
        }

        source_to_target_token_dict = {
            k2: v1
            for k1, v1 in class_to_tokens_dict.items()
            for k2 in token_to_class_dict[k1]
        }
        source_to_target_index_dict = {
            self.tokenizer.token_map[k]: [self.tokenizer.token_map[item] for item in v]
            for k, v in source_to_target_token_dict.items()
        }

        tuple_list = [
            (k, v) for k, vs in source_to_target_index_dict.items() for v in vs
        ]

        # Initialize the mask matrix with zeros
        mask_matrix = torch.zeros(
            (self.tokenizer.vocab_size, self.tokenizer.vocab_size), dtype=torch.float32
        )

        # Set valid transitions based on the tuple list
        for src, dest in tuple_list:
            mask_matrix[src, dest] = 1

        return mask_matrix.to(self.device)

    def apply_logit_mask(self, logits, mask):
        """
        Apply a mask to logits to set disallowed transitions to a large negative value
        to effectively remove them from consideration during the softmax operation.

        Parameters:
        logits (Tensor): The logits tensor of shape [batch_size, seq_len, vocab_size].
        mask (Tensor): A mask tensor of shape [vocab_size, vocab_size] indicating allowed transitions.

        Returns:
        Tensor: Adjusted logits with disallowed transitions set to a large negative value.
        """
        # Ensure the mask is a floating point tensor for subsequent operations
        adjusted_mask = mask.float()

        # Expand the mask to match the dimensions of the logits tensor
        # [batch_size, seq_len, vocab_size, vocab_size]
        adjusted_mask = adjusted_mask.unsqueeze(0).unsqueeze(0)
        adjusted_mask = adjusted_mask.expand(logits.size(0), logits.size(1), -1, -1)

        # Expand logits for matrix multiplication
        # Adding a dimension at the end for broadcasting
        logits_expanded = logits.unsqueeze(-1)

        # Perform batch matrix multiplication to apply the mask
        result = torch.matmul(adjusted_mask, logits_expanded)

        # Squeeze the last dimension to collapse it back to the original logits shape
        result = result.squeeze(-1)

        # Set disallowed transitions to a large negative number
        # This makes them negligible in a softmax operation
        large_negative = torch.full_like(logits, -1e10)
        adjusted_logits = torch.where(result != 0, logits, large_negative)

        return adjusted_logits

    def forward(self, src, tgt):
        src_pos = self.positional_encodings[:, : src.size(1), :]
        tgt_pos = self.positional_encodings[:, : tgt.size(1), :]
        src = self.embedding(src) + src_pos
        tgt = self.embedding(tgt) + tgt_pos
        output = self.transformer(src, tgt)
        logits = self.fc_out(output)

        if self.logits_mask is not None:
            return self.apply_logit_mask(logits, self.logits_mask)

        return logits

    def generate(self, src_tokens, max_length=50):
        if src_tokens.dim() == 1:
            src_tokens = src_tokens.unsqueeze(0)  # Ensure batch dimension is present

        device = src_tokens.device
        src_tokens = src_tokens.to(dtype=torch.long)  # Ensure src_tokens is long type
        tgt_tokens = src_tokens[:, 1:]

        for _ in range(max_length):

            # Run the forward pass
            output = self.forward(src_tokens, tgt_tokens)

            # Convert logits to next token ID
            next_token_logits = output[
                :, -1, :
            ]  # Only consider the last token for prediction
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # Append the predicted token to the tgt_tokens
            tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)

            # Break if EOS token is generated
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return tgt_tokens


class RNNModel(nn.Module):
    def __init__(
        self, tokenizer, embed_size, hidden_size, num_layers, architecture="rnn"
    ):
        super(RNNModel, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.architecture = architecture

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        if self.architecture == "lstm":
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.vocab_size)

        self.to(self.device)

    def forward(self, src):
        src = self.embedding(src)
        hidden = None  # Initialize hidden state
        if self.architecture == "lstm":
            output, (hidden, cell) = self.rnn(src, hidden)
        else:
            output, hidden = self.rnn(src, hidden)
        output = self.fc(output)
        return output

    def generate(self, input_tokens, max_length=30):
        outputs = []
        hidden = None
        cell = None  # Initialize cell state for LSTM
        input_tokens = torch.tensor(input_tokens, dtype=torch.long).to(self.device)

        # Feed the initial sequence to prime the state
        embedded = self.embedding(input_tokens[:-1]).unsqueeze(0)  # Add batch dimension

        if self.architecture == "lstm":
            output, (hidden, cell) = self.rnn(embedded, None)
        else:
            output, hidden = self.rnn(embedded, None)

        logits = self.fc(output)
        outputs.append(logits)

        # Start from the last known token
        current_token = input_tokens[-1].view(
            1, 1
        )  # Reshape for batch and sequence dimension

        # Generate subsequent tokens
        for _ in range(max_length - len(input_tokens)):
            embedded = self.embedding(current_token)
            if self.architecture == "lstm":
                output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
            else:
                output, hidden = self.rnn(embedded, hidden)
            logits = self.fc(output)
            next_token = logits.argmax(-1)
            outputs.append(logits)
            current_token = next_token

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        outputs = torch.cat(outputs, dim=1)  # Concatenate along sequence dimension
        return outputs

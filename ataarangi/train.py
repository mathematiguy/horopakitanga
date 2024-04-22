import os
import math
import click
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ataarangi.data import encode_world_state, RākauDataset, SequenceTokenizer, load_data, custom_collate_fn, create_mask_matrix


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Create a positional encoding that is large enough for any sequence you expect to process
        self.register_buffer('positional_encodings', self.create_positional_encodings(max_seq_length, embed_size))
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def create_positional_encodings(self, max_len, embed_size):
        """Create positional encodings for transformer model."""
        pos_enc = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)

    def forward(self, src, tgt):
        src_pos = self.positional_encodings[:, :src.size(1), :]
        tgt_pos = self.positional_encodings[:, :tgt.size(1), :]
        src = self.embedding(src) + src_pos
        tgt = self.embedding(tgt) + tgt_pos
        output = self.transformer(src, tgt)
        return self.fc_out(output)

    def generate(self, src_tokens, max_length=50):
        if src_tokens.dim() == 1:
            src_tokens = src_tokens.unsqueeze(0)  # Add a batch dimension if it's not there

        src = src_tokens.to(dtype=torch.long)  # Ensure src is long type
        src_pos = self.positional_encodings[:, :src.size(1), :].to(src.device)
        src = self.embedding(src) + src_pos

        # Assuming SOS_TOKEN_INDEX is defined
        SOS_TOKEN_INDEX = 0  # Define it based on your specific model's vocabulary
        tgt_tokens = src.new_full((src.size(0), 1), fill_value=SOS_TOKEN_INDEX, dtype=torch.long)  # Ensure it's long type

        for i in range(max_length - 1):
            tgt_pos = self.positional_encodings[:, :tgt_tokens.size(1), :].to(src.device)
            tgt = self.embedding(tgt_tokens) + tgt_pos

            output = self.transformer(src, tgt)
            output = self.fc_out(output[:, -1, :])  # Only take the last token's output
            next_token = output.argmax(-1).unsqueeze(-1)
            tgt_tokens = torch.cat([tgt_tokens, next_token], dim=-1)

            # Assuming EOS_TOKEN_INDEX is defined
            EOS_TOKEN_INDEX = 53
            if next_token.item() == EOS_TOKEN_INDEX:
                break

        return tgt_tokens.squeeze(0)  # Remove the batch dimension if originally there was no batch dimension


def train_one_epoch(model, criterion, optimizer, train_dataloader, mask_tensor, device):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        src = batch['input_ids'][:, :-1].to(device)  # Inputs to the model
        tgt = batch['input_ids'][:, 1:].to(device)   # Expected outputs from the model

        optimizer.zero_grad()
        output = model(src, src)  # Assuming you're using the same src for both src and tgt in model forward pass

        # Check the size and apply the cross-entropy loss
        if output.shape[1] != tgt.shape[1]:
            print(f"Adjusting target shape from {tgt.shape} to match output {output.shape[1]}")
            tgt = tgt[:, :output.shape[1]]  # Adjust tgt length to match output length if necessary

        # Calculate loss using the masked loss function
        loss = masked_loss_function(output, tgt, mask_tensor, lambda_penalty=1.0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_dataloader)


def masked_loss_function(outputs, targets, mask_tensor=None, lambda_penalty=1.0):
    """
    outputs: Logits from the model (batch_size, seq_len, vocab_size)
    targets: Ground truth indices for each position in the sequence (batch_size, seq_len)
    mask_tensor: A tensor indicating valid next token classes (vocab_size, vocab_size)
    lambda_penalty: Weighting factor for the penalty term
    """

    batch_size, seq_len, vocab_size = outputs.size()

    if mask_tensor is None:
        # If no mask is provided, create a mask that allows all transitions
        mask_tensor = torch.ones((vocab_size, vocab_size), dtype=torch.float32, device=outputs.device)

    # Apply mask to the output logits
    if mask_tensor.size() != (vocab_size, vocab_size):
        raise ValueError(f"Mask tensor size mismatch. Expected [{vocab_size}, {vocab_size}]")

    # Expand mask tensor to match output dimensions [batch_size, seq_len, vocab_size, vocab_size]
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, vocab_size, vocab_size]
    mask_tensor = mask_tensor.expand(batch_size, seq_len, -1, -1)  # [batch_size, seq_len, vocab_size, vocab_size]

    # Calculate cross-entropy loss using the masked outputs
    # You need to ensure that your mask_tensor is correctly applied to penalize invalid transitions
    ce_loss = F.cross_entropy(outputs.transpose(1, 2), targets, reduction='mean')  # basic cross-entropy loss

    # Modify this to incorporate your mask or transition penalties
    # Example: applying a simple mask that could scale the logits before computing cross-entropy
    # outputs_masked = outputs * mask_tensor (this line needs to be correctly implemented based on your mask logic)

    return ce_loss


def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch['input_ids'][:, :-1].to(device)
            tgt = batch['input_ids'][:, 1:].to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def setup_model(lr, num_layers, embed_size, dim_feedforward, nhead, dropout, batch_size, train_path, dev_path, class_successors_json, token_to_class_json):

    tokenizer = SequenceTokenizer()

    mask_tensor = create_mask_matrix(tokenizer, class_successors_json, token_to_class_json)

    # Load data and initialize tokenizers
    train_data, dev_data = load_data(train_path, dev_path)

    train_tokens = train_data.apply(
        lambda x: tokenizer.tokenize(x.rākau + x.description.split(' ')),
        axis=1)

    dev_tokens = dev_data.apply(
        lambda x: tokenizer.tokenize(x.rākau + x.description.split(' ')),
        axis=1)

    # Prepare datasets and dataloaders
    train_dataset = RākauDataset(train_tokens, tokenizer)
    dev_dataset = RākauDataset(dev_tokens, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(
        vocab_size=tokenizer.vocab_size,
        embed_size=embed_size,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_length=500,
        dropout=dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, train_dataloader, dev_dataloader, criterion, optimizer, mask_tensor, device


@click.command()
@click.option('--lr', default=0.0001, type=float, help='Learning rate for optimizer.')
@click.option('--num_layers', default=2, type=int, help='Number of encoder/decoder layers.')
@click.option('--embed_size', default=512, type=int, help='Size of the embedding.')
@click.option('--dim_feedforward', default=1024, type=int, help='Dimension of the feedforward network.')
@click.option('--nhead', default=8, type=int, help='Number of attention heads.')
@click.option('--dropout', default=0.1, type=float, help='Dropout rate.')
@click.option('--batch_size', default=64, type=int, help='Batch size for training.')
@click.option('--epochs', default=100, type=int, help='Number of epochs to train.')
@click.option('--train_path', type=str, help='Path to the training data file.')
@click.option('--dev_path', type=str, help='Path to the dev data file.')
@click.option('--model_folder', default='models', type=str, help='Folder to save the model checkpoints.')
@click.option('--class_successors_json', default="data/class_successors.json", type=str, help='Path to class_successors.json file.')
@click.option('--token_to_class_json', default="data/token_to_class.json", type=str, help='Path to token_to_class.json file.')
def run_training(lr, num_layers, embed_size, dim_feedforward, nhead, dropout, batch_size, epochs, train_path, dev_path, model_folder, class_successors_json, token_to_class_json):

    model, train_dataloader, dev_dataloader, criterion, optimizer, mask_tensor, device = setup_model(lr, num_layers, embed_size, dim_feedforward, nhead, dropout, batch_size, train_path, dev_path, class_successors_json, token_to_class_json)

    model_name = f'lr={lr}-num_layers={num_layers}-embed_size={embed_size}-nhead={nhead}-dim_ff={dim_feedforward}-dropout={dropout}'

    # Make sure the model folder exists
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Open the history file for writing
    history_path = os.path.join(model_folder, model_name + '-history.csv')
    with open(history_path, 'w') as history_file:
        history_file.write('epoch,train_loss,dev_loss\n')

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, criterion, optimizer, train_dataloader, mask_tensor, device)
            dev_loss = evaluate(model, criterion, dev_dataloader, device)

            # Log the epoch results
            history_file.write(f'{epoch+1},{train_loss},{dev_loss}\n')
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}')

        # Save the model checkpoint
        torch.save(model.state_dict(), os.path.join(model_folder, f'{model_name}.pth'))


if __name__ == '__main__':
    run_training()

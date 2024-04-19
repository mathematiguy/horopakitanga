import os
import math
import click
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ataarangi.data import encode_world_state, TextTokenizer, WorldStateTokenizer, RākauDataset, load_data, custom_collate_fn


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


def train_one_epoch(model, criterion, optimizer, dataloader, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        src = batch['input_ids'][:, :-1].to(device)
        tgt = batch['input_ids'][:, 1:].to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


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


def setup_model(lr, num_layers, embed_size, dim_feedforward, nhead, dropout, batch_size):
    train_path = 'data/train_set.csv'
    dev_path = 'data/dev_set.csv'

    text_tokenizer = TextTokenizer()
    ws_tokenizer = WorldStateTokenizer()

    # Load data and initialize tokenizers
    train_data, dev_data = load_data(train_path, dev_path, text_tokenizer, ws_tokenizer)

    # Prepare datasets and dataloaders
    train_dataset = RākauDataset(train_data['rākau'], train_data['description'], ws_tokenizer, text_tokenizer)
    dev_dataset = RākauDataset(dev_data['rākau'], dev_data['description'], ws_tokenizer, text_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(
        vocab_size=max(text_tokenizer.token_map.values())+1,
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
    return model, train_dataloader, dev_dataloader, criterion, optimizer, device


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
def run_training(lr, num_layers, embed_size, dim_feedforward, nhead, dropout, batch_size, epochs, train_path, dev_path, model_folder):

    model, train_dataloader, dev_dataloader, criterion, optimizer, device = setup_model(lr, num_layers, embed_size, dim_feedforward, nhead, dropout, batch_size)

    model_name = f'lr={lr}-num_layers={num_layers}-embed_size={embed_size}-nhead={nhead}-dim_ff={dim_feedforward}-dropout={dropout}'

    # Make sure the model folder exists
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Open the history file for writing
    history_path = os.path.join(model_folder, model_name + '-history.csv')
    with open(history_path, 'w') as history_file:
        history_file.write('epoch,train_loss,dev_loss\n')

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, criterion, optimizer, train_dataloader, device)
            dev_loss = evaluate(model, criterion, dev_dataloader, device)

            # Log the epoch results
            history_file.write(f'{epoch+1},{train_loss},{dev_loss}\n')
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}')

        # Save the model checkpoint
        torch.save(model.state_dict(), os.path.join(model_folder, f'{model_name}.pth'))


if __name__ == '__main__':
    run_training()

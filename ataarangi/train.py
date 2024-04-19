import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import click

from ataarangi.data import encode_world_state, TextTokenizer, WorldStateTokenizer, RākauDataset


def custom_collate_fn(batch):
    # Extracting input_ids, token_type_ids, and attention_mask from the batch
    input_ids = [item['input_ids'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    # Find the maximum sequence length in this batch
    max_len = max(len(ids) for ids in input_ids)

    # Pad all sequences to this maximum length
    padded_input_ids = torch.stack([torch.cat([ids, torch.zeros(max_len - len(ids), dtype=torch.long)]) for ids in input_ids])
    padded_token_type_ids = torch.stack([torch.cat([ids, torch.zeros(max_len - len(ids), dtype=torch.long)]) for ids in token_type_ids])
    padded_attention_mask = torch.stack([torch.cat([mask, torch.zeros(max_len - len(mask), dtype=torch.long)]) for mask in attention_mask])

    return {
        'input_ids': padded_input_ids,
        'token_type_ids': padded_token_type_ids,
        'attention_mask': padded_attention_mask
    }


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

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(path, params):
        model = cls(**params)
        model.load_state_dict(torch.load(path))
        return model


@click.command()
@click.option('--lr', default=0.001, help='Learning rate for optimizer.')
@click.option('--num_layers', default=2, help='Learning rate for optimizer.')
@click.option('--embed_size', default=512, help='Learning rate for optimizer.')
@click.option('--dim_feedforward', default=1024, help='Learning rate for optimizer.')
@click.option('--nhead', default=8, help='Learning rate for optimizer.')
@click.option('--dropout', default=0.1, help='Learning rate for optimizer.')
@click.option('--batch_size', default=64, help='Batch size for training.')
@click.option('--epochs', default=100, help='Number of epochs to train.')
@click.option('--train_path', default='data/train_set.csv', help='Path to the training data file.')
@click.option('--dev_path', default='data/dev_set.csv', help='Path to the dev data file.')
@click.option('--history_path', default='models/history.csv', help='File path to save the training and development history.')
@click.option('--model_folder', default='models/', help='Folder to save the model checkpoints.')
def train(lr, num_layers, embed_size, dim_feedforward, n_head, dropout, batch_size, epochs, train_path, dev_path, history_path, model_folder):
    # Load training and dev data
    train_data = pd.read_csv(train_path)
    dev_data = pd.read_csv(dev_path)
    
    train_data['rākau'] = train_data['rākau'].apply(json.loads)
    dev_data['rākau'] = dev_data['rākau'].apply(json.loads)

    text_tokenizer = TextTokenizer()
    ws_tokenizer = WorldStateTokenizer()

    train_data['input'] = train_data['rākau'].apply(ws_tokenizer.tokenize)
    train_data['target'] = train_data['description'].apply(text_tokenizer.tokenize)
    dev_data['input'] = dev_data['rākau'].apply(ws_tokenizer.tokenize)
    dev_data['target'] = dev_data['description'].apply(text_tokenizer.tokenize)

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
        num_encoder_layers=embed_size,
        num_decoder_layers=embed_size,
        dim_feedforward=dim_feedforward,
        max_seq_length=500,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_dev_loss = float('inf')

    # Open the history file and write headers
    with open(history_path, 'w') as f:
        f.write('epoch,train_loss,dev_loss\n')
        
        # Training and validation loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for batch in train_dataloader:
                src = batch['input_ids'][:, :-1].to(device)
                tgt = batch['input_ids'][:, 1:].to(device)

                optimizer.zero_grad()
                output = model(src, tgt)
                loss = criterion(output.view(-1, output.size(-1)), tgt.reshape(-1))
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_dataloader)

            # Evaluate on dev set
            model.eval()
            dev_loss = 0
            with torch.no_grad():
                for batch in dev_dataloader:
                    src = batch['input_ids'][:, :-1].to(device)
                    tgt = batch['input_ids'][:, 1:].to(device)

                    output = model(src, tgt)
                    loss = criterion(output.view(-1, output.size(-1)), tgt.reshape(-1))
                    dev_loss += loss.item()

            avg_dev_loss = dev_loss / len(dev_dataloader)

            # Log to console and history file
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}, Dev Loss: {avg_dev_loss}')
            f.write(f'{epoch+1},{avg_train_loss},{avg_dev_loss}\n')

            # Save the model checkpoint
            model_filename = f'{model_folder}/loss={avg_dev_loss:.4f}_epochs={epoch+1}_.pt'
            torch.save(model.state_dict(), model_filename)

            # Optionally, save only the best model based on dev loss
            if avg_dev_loss < best_dev_loss:
                best_dev_loss = avg_dev_loss
                best_model_filename = f'{model_folder}/best_model.pt'
                torch.save(model.state_dict(), best_model_filename)

    return best_dev_loss


if __name__ == '__main__':
    train()

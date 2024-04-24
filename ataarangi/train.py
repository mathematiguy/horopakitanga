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
from ataarangi.models import RNNModel, TransformerModel
from ataarangi.data import (
    encode_world_state,
    RākauDataset,
    SequenceTokenizer,
    load_data,
    rākau_collate_fn,
)


def train_one_epoch(model, criterion, optimizer, dataloader, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids
        )  # Ensure your model outputs logits for each token in the sequence

        # Assume the targets are aligned with output tokens and use token_type_ids to mask
        targets = input_ids  # if targets are aligned with input_ids in your setup
        mask = (
            token_type_ids == 1
        )  # only compute loss where token_type_ids is 1 (targets)

        # Masking outputs and targets
        outputs = outputs[mask]
        targets = targets[mask]

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def cross_entropy_loss(outputs, targets):
    """
    outputs: Logits from the model (batch_size, seq_len, vocab_size)
    targets: Ground truth indices for each position in the sequence (batch_size, seq_len)
    """
    # Calculate the cross-entropy loss directly without any mask
    ce_loss = F.cross_entropy(outputs.transpose(1, 2), targets, reduction="mean")
    return ce_loss


def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            # Forward pass: compute the model output
            outputs = model(input_ids)

            # Mask to focus only on the tokens that should contribute to loss calculation
            mask = (
                token_type_ids == 1
            )  # Only compute loss where token_type_ids is 1 (targets)

            # Masking outputs and targets
            outputs = outputs[mask]
            targets = input_ids[mask]  # Assuming targets are aligned with input_ids

            # Calculate the loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def setup_model(
    lr,
    num_layers,
    embed_size,
    hidden_size,
    dropout,
    batch_size,
    num_batches=-1,
    train_path="data/train_set.csv",
    dev_path="data/dev_set.csv",
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = SequenceTokenizer()

    # Load data and initialize tokenizers
    train_data, dev_data = load_data(train_path, dev_path)

    # Set the number of batches to train over
    if num_batches == -1:
        num_batches = (len(train_data) // batch_size) + 1
    num_samples = min(len(train_data), num_batches * batch_size)

    # Prepare datasets and dataloaders
    train_dataset = RākauDataset(
        srcs=train_data.rākau[:num_samples],
        tgts=train_data.description[:num_samples],
        tokenizer=tokenizer,
    )
    dev_dataset = RākauDataset(
        srcs=dev_data.rākau, tgts=dev_data.description, tokenizer=tokenizer
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=rākau_collate_fn,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=rākau_collate_fn,
    )

    model = RNNModel(
        tokenizer=tokenizer,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        architecture="rnn",
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, train_dataloader, dev_dataloader, criterion, optimizer, device


@click.command()
@click.option(
    "--architecture",
    default="rnn",
    type=str,
    help="Model architecture (rnn, lstm or transformer).",
)
@click.option(
    "--num_batches",
    default=-1,
    type=int,
    help='Number of batches to train over (default: -1, meaning "all").',
)
@click.option("--lr", default=0.0001, type=float, help="Learning rate for optimizer.")
@click.option(
    "--num_layers", default=2, type=int, help="Number of encoder/decoder layers."
)
@click.option("--embed_size", default=512, type=int, help="Size of the embedding.")
@click.option(
    "--hidden_size", default=512, type=int, help="Size of the hidden layers (for RNN)."
)
@click.option("--dropout", default=0.1, type=float, help="Dropout rate.")
@click.option("--batch_size", default=64, type=int, help="Batch size for training.")
@click.option("--epochs", default=100, type=int, help="Number of epochs to train.")
@click.option("--train_path", type=str, help="Path to the training data file.")
@click.option("--dev_path", type=str, help="Path to the dev data file.")
@click.option(
    "--model_folder",
    default="models",
    type=str,
    help="Folder to save the model checkpoints.",
)
@click.option(
    "--class_successors_json",
    default="data/class_successors.json",
    type=str,
    help="Path to class_successors.json file.",
)
@click.option(
    "--token_to_class_json",
    default="data/token_to_class.json",
    type=str,
    help="Path to token_to_class.json file.",
)
def run_training(
    architecture,
    num_batches,
    lr,
    num_layers,
    embed_size,
    hidden_size,
    dropout,
    batch_size,
    epochs,
    train_path,
    dev_path,
    model_folder,
    class_successors_json,
    token_to_class_json,
):

    model, train_dataloader, dev_dataloader, criterion, optimizer, device = setup_model(
        lr, num_layers, embed_size, hidden_size, dropout, batch_size, num_batches
    )

    model_name = f"lr={lr}-num_layers={num_layers}-embed_size={embed_size}-hidden_size={hidden_size}-dropout={dropout}"

    # Make sure the model folder exists
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Open the history file for writing
    history_path = os.path.join(model_folder, model_name + "-history.csv")
    with open(history_path, "w") as history_file:
        history_file.write("epoch,train_loss,dev_loss\n")

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, criterion, optimizer, train_dataloader, device
            )
            dev_loss = evaluate(model, criterion, dev_dataloader, device)

            # Log the epoch results
            history_file.write(f"{epoch+1},{train_loss},{dev_loss}\n")
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Dev Loss: {dev_loss:.6f}"
            )

        # Save the model checkpoint
        torch.save(model.state_dict(), os.path.join(model_folder, f"{model_name}.pth"))


if __name__ == "__main__":
    run_training()

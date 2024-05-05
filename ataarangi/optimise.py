import os
import click
import pandas as pd

from ataarangi.train import train_one_epoch, TransformerModel, evaluate, setup_model
from ataarangi.data import (
    encode_world_state,
    SequenceTokenizer,
    RākauDataset,
    load_data,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import optuna
from optuna.integration import PyTorchLightningPruningCallback


def generate_trial_name(trial):
    # Generate a name based on some important hyperparameters
    lr = trial.params["lr"]
    num_layers = trial.params["num_layers"]
    embed_size = trial.params["embed_size"]
    hidden_size = trial.params["hidden_size"]
    batch_size = trial.params["batch_size"]
    return f"trial_lr={lr:.5f}-layers={num_layers}-embed={embed_size}-hidden_size={hidden_size}-batch_size={batch_size}"


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 10)
    embed_size = trial.suggest_categorical("embed_size", [16, 32, 64, 128, 256])
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512, 768])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    # Optionally set a descriptive name for the trial
    trial_name = generate_trial_name(trial)
    trial.set_user_attr("name", trial_name)

    # Assuming data loaders and model setup is handled separately
    model, train_dataloader, dev_dataloader, criterion, optimizer, device = setup_model(
        lr, num_layers, embed_size, hidden_size, batch_size
    )

    epochs = 100  # You might want to adjust this per trial or pass it as an argument
    train_losses = []
    dev_losses = []
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, criterion, optimizer, train_dataloader, device
        )
        dev_loss = evaluate(model, criterion, dev_dataloader, device)

        train_losses.append(train_loss)
        dev_losses.append(dev_loss)

        # Report the current loss and check for pruning
        trial.report(dev_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    trial_name = f"loss={dev_loss:.4f}-" + trial_name

    # Save model + history
    torch.save(model.state_dict(), os.path.join("models", f"{trial_name}.pth"))
    history_path = os.path.join("models", trial_name + "-history.csv")

    with open(history_path, "w") as history_file:
        history_file.write("epoch,train_loss,dev_loss\n")
        for epoch, losses in enumerate(zip(train_losses, dev_losses)):
            train_loss, dev_loss = losses
            history_file.write(f"{epoch+1},{train_loss},{dev_loss}\n")

    return dev_loss


def save_study_to_csv(study, filename):
    # Extract study results into a dataframe
    trial_data = {
        "trial_number": [trial.number for trial in study.trials],
        "value": [
            trial.value if trial.value is not None else float("nan")
            for trial in study.trials
        ],
        "params": [trial.params for trial in study.trials],
        "state": [trial.state for trial in study.trials],
    }
    df = pd.DataFrame(trial_data)
    df.to_csv(filename, index=False)


def main():
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        objective, n_trials=100
    )  # Adjust number of trials and potentially add a timeout

    save_study_to_csv(study, "study_results.csv")

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()

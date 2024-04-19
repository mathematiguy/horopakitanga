import optuna
from ataarangi.train import train
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial):
    # Define hyperparameter search space using trial object
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    num_layers = trial.suggest_int('num_layers', 2, 6)
    embed_size = trial.suggest_categorical('embed_size', [256, 512, 768])
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [512, 1024, 2048])
    nhead = trial.suggest_categorical('nhead', [4, 8, 16])
    dropout = trial.suggest_uniform('dropout', 0.1, 0.3)

    # Assume model training and validation is defined in this function
    loss = train(embed_size, num_layers, nhead, dropout)
    
    return loss  # A value to minimize

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
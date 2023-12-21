import os
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score


def verbose_model(epoch, avg_loss, avg_metrics, split, verbose):
    if verbose == 2:
        print(
            f"Epoch {epoch}, Avg {split} Loss: {avg_loss:.4f}, MAE: {avg_metrics['MAE']:.4f}, RMSE: {avg_metrics['RMSE']:.4f}, MAPE: {avg_metrics['MAPE']:.4f}, R-squared: {avg_metrics['R_SQUARED']:.4f}, Max Error: {avg_metrics['MAX_ERROR']:.4f}"
        )
    elif verbose == 1:
        print(f"Epoch {epoch}, Avg {split} Loss: {avg_loss:.4f}")


def flatten_config(config):
    flat_config = {}
    for category, params in config.items():
        for param, value in params.items():
            flat_config[f"{category}_{param}"] = value
    return flat_config


def save_plot_loss_and_metrics(loss, metrics, train=True, save_dir="plots"):
    num_epochs = range(1, len(loss) + 1)
    split_value = "Training" if train else "Validation"
    folder_path = os.path.join(save_dir, split_value)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.figure(figsize=(4, 2))
    plt.plot(num_epochs, loss, label=f"{split_value} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{split_value} Loss Across Epochs")
    plt.legend()
    loss_plot_path = os.path.join(folder_path, f"{split_value}_Loss.png")
    plt.savefig(loss_plot_path)
    plt.close()

    for metric in metrics[0].keys():
        values = [m[metric] for m in metrics]
        plt.figure(figsize=(4, 2))
        plt.plot(num_epochs, values, label=metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.title(f"{metric} Across Epochs")
        plt.legend()
        metric_plot_path = os.path.join(folder_path, f"{metric}_Across_Epochs.png")
        plt.savefig(metric_plot_path)
        plt.close()


def calculate_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets)).item()


def calculate_rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def calculate_mape(predictions, targets):
    epsilon = 1e-7  # Small constant to prevent division by zero
    return torch.mean(torch.abs((targets - predictions) / (targets + epsilon))).item()


def calculate_r_squared(predictions, targets):
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    return r2_score(targets, predictions)


def calculate_max_error(predictions, targets):
    return torch.max(torch.abs(predictions - targets)).item()


def compute_metrics(predictions, targets, epoch):
    mae = calculate_mae(predictions, targets)
    rmse = calculate_rmse(predictions, targets)
    mape = calculate_mape(predictions, targets)
    r_squared = calculate_r_squared(predictions, targets)
    max_error = calculate_max_error(predictions, targets)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R_SQUARED": r_squared,
        "MAX_ERROR": max_error,
    }


def log_metrics(writer, mode, metrics, epoch):
    for metrics, value in metrics.items():
        writer.add_scalar(f"{metrics}/{mode}", value, epoch)


class EarlyStopping:
    def __init__(self, patience=5, monitor="val_loss"):
        self.patience = patience
        self.monitor = monitor
        self.best_score = None
        self.epochs_no_improve = 0
        self.should_stop = False

    def __call__(self, current_score):
        if self.best_score is None or current_score < self.best_score:
            self.best_score = current_score
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            self.should_stop = True
